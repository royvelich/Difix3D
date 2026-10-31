#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/rvelich-ebs-2/repos/Difix3D-vanilla-our-data
python=/data/rvelich/venvs/geoquery-strict-uv/bin/python
train_manifest=/mnt/rvelich-ebs-2/datasets/difix_manifests/train4700_rank0.jsonl
validation_manifest=/mnt/rvelich-ebs-2/datasets/difix_manifests/historical_val100.jsonl
run_name=difix_vanilla_rank0_train4700_hist100_4gpu_200k_val5k_clearml_ohio_1031b
output_dir=/mnt/rvelich-ebs-2/training_runs/${run_name}

for required_path in \
  "$repo/src/train_difix.py" \
  "$train_manifest" \
  "${train_manifest}.idx" \
  "$validation_manifest" \
  "${validation_manifest}.idx"
do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required file: $required_path" >&2
    exit 1
  fi
done

"$python" -c "import clearml, pyiqa"

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [[ -n "$gpu_pids" ]]; then
  echo "Ohio GPUs are already occupied by process IDs:" >&2
  echo "$gpu_pids" >&2
  exit 2
fi

mkdir -p "$output_dir"
git -C "$repo" rev-parse HEAD >"$output_dir/git_commit.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/home/ubuntu/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "$repo"
exec "$python" -m accelerate.commands.launch \
  --mixed_precision=bf16 \
  --multi_gpu \
  --num_machines=1 \
  --num_processes=4 \
  --main_process_port=29513 \
  "$repo/src/train_difix.py" \
  --output_dir="$output_dir" \
  --dataset_path="$train_manifest" \
  --validation_dataset_path="$validation_manifest" \
  --train_split=train \
  --validation_split=test \
  --data_root=/data/rvelich/datasets/dl3dv_pairs_2x_20261013_train4700_seed42plus44 \
  --validation_data_root=/data/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42 \
  --path_prefix_from=/mnt/nvme/rvelich/datasets \
  --path_prefix_to=/data/rvelich/datasets \
  --max_train_steps=200000 \
  --height=576 \
  --width=1024 \
  --learning_rate=2e-5 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --validation_num_workers=4 \
  --mv_unet \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=5000 \
  --eval_freq=5000 \
  --viz_freq=100 \
  --num_samples_eval=100 \
  --num_validation_panel_samples=100 \
  --lambda_lpips=1.0 \
  --lambda_l2=1.0 \
  --lambda_gram=1.0 \
  --gram_loss_warmup_steps=2000 \
  --report_to=clearml \
  --tracker_project_name="$run_name" \
  --tracker_run_name="$run_name" \
  --timestep=199 \
  --seed=42 \
  --mixed_precision=bf16 \
  --allow_tf32 \
  --set_grads_to_none
