#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/rvelich/repos/Difix3D-oracle
python=/home/rvelich/venvs/difix-experiments/bin/python
run_name=difix_oracle_lpips3x3_rg05_v95_sparsejoint_wrapvalid80_bs1x8_lr2e5_200k_val5k_p4d0_1107a_fix1
output_dir=/home/rvelich/difix_runs/${run_name}
resume_checkpoint=/home/rvelich/difix_runs/difix_oracle_lpips3x3_rg05_v95_sparsejoint_wrapvalid80_bs1x8_lr2e5_200k_val5k_p4d0_1107a/checkpoints/model_5000.pkl
train_manifest=/home/rvelich/difix_data/manifests/train4700_rank0_geometry.jsonl
train_root=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/combined4700
validation_manifest=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/val100_scene_balanced_seed20261028/data.json
validation_root=/mnt/nvme/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42
validation_s3_root=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/val100_scene_balanced_seed20261028
wrapped_sentinel=/home/rvelich/difix_data/wrapped-online-disabled
pixeldit_repo=/mnt/nvme/rvelich/repos/gs-pixel-refiner-3-wrapped-transplant-skips-1029
s3_cache=/home/rvelich/difix_s3_cache
s3_routes=/mnt/nvme/rvelich/pixeldit-s3-routes/routes.json

for required in \
  "$repo/src/train_difix.py" \
  "$resume_checkpoint" \
  "$train_manifest" \
  "${train_manifest}.idx" \
  "$pixeldit_repo/restore/warped_reference.py" \
  "$s3_routes"
do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [[ -n "$gpu_pids" ]]; then
  echo "A100 GPUs are occupied by process IDs:" >&2
  echo "$gpu_pids" >&2
  exit 2
fi

mkdir -p "$output_dir" "$wrapped_sentinel" "$s3_cache"
cp "$repo/GIT_COMMIT" "$output_dir/git_commit.txt"
cp "$0" "$output_dir/launch.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/home/rvelich/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_NVLS_ENABLE=0
export WANDB_ENTITY=royve

cd "$repo"
exec "$python" -m accelerate.commands.launch \
  --mixed_precision=bf16 \
  --multi_gpu \
  --num_machines=1 \
  --num_processes=8 \
  --main_process_port=29641 \
  "$repo/src/train_difix.py" \
  --output_dir="$output_dir" \
  --dataset_path="$train_manifest" \
  --validation_dataset_path="$validation_manifest" \
  --train_split=train \
  --validation_split=test \
  --data_root="$train_root" \
  --validation_data_root="$validation_root" \
  --path_prefix_from="$validation_root" \
  --path_prefix_to="$validation_s3_root" \
  --storage_backend=s3_stream \
  --s3_cache_dir="$s3_cache" \
  --no-s3_cache_images \
  --s3_max_pool_connections=64 \
  --s3_tar_routes_path="$s3_routes" \
  --wrapped_reference_cache_dir="$wrapped_sentinel" \
  --validation_wrapped_reference_cache_dir="$wrapped_sentinel" \
  --wrapped_reference_builder_repo="$pixeldit_repo" \
  --wrapped_reference_depth_cache_dir="$train_root/_geo_depth_cache_B_v2_portable" \
  --validation_wrapped_reference_depth_cache_dir="$validation_root/_geo_depth_cache_B" \
  --wrapped_reference_splat_radius=1 \
  --no-cache_built_wrapped_references \
  --wrapped_reference_fill=black \
  --sparse_mv_attention \
  --view_attention_query_modes=all,all,valid_only \
  --view_attention_matrix="all,all,valid_only;all,all,valid_only;all,all,valid_only" \
  --wrapped_reference_patch_valid_fraction=0.8 \
  --wrapped_reference_denoise_mask=valid_only \
  --wrapped_reference_loss_mask=valid_pixels \
  --render_endpoint_mode=wrapped_patch_override \
  --render_endpoint_override_policy=oracle_gt \
  --render_endpoint_override_patch_size=16 \
  --render_endpoint_override_valid_fraction=0.95 \
  --render_endpoint_oracle_metric=lpips \
  --render_endpoint_oracle_window_patches=3 \
  --render_endpoint_oracle_margin=0.0 \
  --render_endpoint_oracle_relative_gain=0.05 \
  --render_endpoint_oracle_lpips_chunk_size=64 \
  --max_train_steps=200000 \
  --height=576 \
  --width=1024 \
  --learning_rate=2e-5 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --validation_num_workers=4 \
  --gradient_checkpointing \
  --checkpointing_steps=5000 \
  --eval_freq=5000 \
  --viz_freq=100 \
  --num_samples_eval=100 \
  --num_validation_panel_samples=100 \
  --lambda_lpips=1.0 \
  --lambda_l2=1.0 \
  --lambda_gram=1.0 \
  --gram_loss_warmup_steps=2000 \
  --report_to=wandb \
  --tracker_project_name=pixel-dit \
  --tracker_run_name="$run_name" \
  --timestep=199 \
  --seed=42 \
  --mixed_precision=bf16 \
  --allow_tf32 \
  --set_grads_to_none \
  --resume="$resume_checkpoint" \
  --validate_on_start
