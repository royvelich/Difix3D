#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/rvelich/venvs/difix-experiments/bin/python
TRAIN_SCRIPT=/home/rvelich/repos/Difix3D-oracle/src/train_difix.py
RUN_ROOT=/home/rvelich/difix_runs
LOG_ROOT=/home/rvelich/difix_logs

VANILLA_NAME=difix_sdturbo_vanilla_twoview_bs1x4_lr2e5_200k_val250_2x100allpanels_gram0_p4d0_1108a
ORACLE_NAME=difix_sdturbo_vanilla_twoview_oracle_lpips5x5_rg20_v95_bs1x4_lr2e5_200k_val250_2x100allpanels_gram0_p4d0_1108a

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

COMMON_ARGS=(
    --dataset_path=/home/rvelich/difix_data/manifests/train4700_rank0_geometry.jsonl
    --validation_dataset_path=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/val100_scene_balanced_seed20261028/data.json
    --validation_name=new100
    --secondary_validation_dataset_path=/home/rvelich/difix_data/manifests/historical_val100.json
    --secondary_validation_name=historical100
    --secondary_validation_data_root=/home/rvelich/difix_data/historical_val100_root
    --secondary_validation_path_prefix_from=/mnt/nvme/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42
    --secondary_validation_path_prefix_to=/home/rvelich/difix_data/historical_val100_root
    --train_split=train
    --validation_split=test
    --data_root=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/combined4700
    --validation_data_root=/mnt/nvme/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42
    --path_prefix_from=/mnt/nvme/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42
    --path_prefix_to=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/val100_scene_balanced_seed20261028
    --storage_backend=s3_stream
    --s3_cache_dir=/home/rvelich/difix_s3_cache
    --no-s3_cache_images
    --s3_max_pool_connections=64
    --s3_tar_routes_path=/mnt/nvme/rvelich/pixeldit-s3-routes/routes.json
    --mv_unet
    --no-sparse_mv_attention
    --enable_xformers_memory_efficient_attention
    --max_train_steps=200000
    --height=576
    --width=1024
    --learning_rate=2e-5
    --train_batch_size=1
    --gradient_accumulation_steps=1
    --dataloader_num_workers=8
    --validation_num_workers=4
    --gradient_checkpointing
    --checkpointing_steps=5000
    --eval_freq=250
    --viz_freq=100
    --num_samples_eval=100
    --num_validation_panel_samples=100
    --lambda_lpips=1
    --lambda_l2=1
    --lambda_gram=0
    --gram_loss_warmup_steps=2000
    --no-legacy_difix_optimizer
    --report_to=wandb
    --tracker_project_name=pixel-dit
    --timestep=199
    --seed=42
    --mixed_precision=bf16
    --allow_tf32
    --set_grads_to_none
)

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "${PYTHON_BIN}" \
    -m accelerate.commands.launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=4 \
    --main_process_port=29671 \
    "${TRAIN_SCRIPT}" \
    --output_dir="${RUN_ROOT}/${VANILLA_NAME}" \
    --tracker_run_name="${VANILLA_NAME}" \
    --render_endpoint_mode=original \
    "${COMMON_ARGS[@]}" \
    >"${LOG_ROOT}/vanilla_twoview_gram0_1108a.log" 2>&1 </dev/null &
VANILLA_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup "${PYTHON_BIN}" \
    -m accelerate.commands.launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=4 \
    --main_process_port=29672 \
    "${TRAIN_SCRIPT}" \
    --output_dir="${RUN_ROOT}/${ORACLE_NAME}" \
    --tracker_run_name="${ORACLE_NAME}" \
    --wrapped_reference_cache_dir=/home/rvelich/difix_data/wrapped-online-disabled \
    --validation_wrapped_reference_cache_dir=/home/rvelich/difix_data/wrapped-online-disabled \
    --wrapped_reference_builder_repo=/mnt/nvme/rvelich/repos/gs-pixel-refiner-3-wrapped-transplant-skips-1029 \
    --wrapped_reference_depth_cache_dir=s3://rvelich-data/datasets/dl3dv_pairs_2x_20261013_ohio1700_seed44/metadata/combined4700/_geo_depth_cache_B_v2_portable \
    --validation_wrapped_reference_depth_cache_dir=/mnt/nvme/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42/_geo_depth_cache_B \
    --secondary_validation_wrapped_reference_depth_cache_dir=/mnt/nvme/rvelich/datasets/dl3dv_pairs_2x_20260922_val141_benchmark_s3to10_seed42/_geo_depth_cache_B \
    --wrapped_reference_splat_radius=1 \
    --no-cache_built_wrapped_references \
    --wrapped_reference_fill=black \
    --no-wrapped_reference_model_view \
    --render_endpoint_mode=wrapped_patch_override \
    --render_endpoint_override_policy=oracle_gt \
    --render_endpoint_override_patch_size=16 \
    --render_endpoint_override_valid_fraction=0.95 \
    --render_endpoint_oracle_metric=lpips \
    --render_endpoint_oracle_window_patches=5 \
    --render_endpoint_oracle_margin=0 \
    --render_endpoint_oracle_relative_gain=0.20 \
    --render_endpoint_oracle_lpips_chunk_size=64 \
    "${COMMON_ARGS[@]}" \
    >"${LOG_ROOT}/oracle_twoview_gram0_1108a.log" 2>&1 </dev/null &
ORACLE_PID=$!

printf 'vanilla_pid=%s\noracle_pid=%s\n' "${VANILLA_PID}" "${ORACLE_PID}"
