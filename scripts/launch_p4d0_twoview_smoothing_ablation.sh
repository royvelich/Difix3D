#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/rvelich/venvs/difix-experiments/bin/python
REPO_ROOT=/home/rvelich/repos/Difix3D-smoothing
TRAIN_SCRIPT=${REPO_ROOT}/src/train_difix.py
RUN_ROOT=/home/rvelich/difix_runs
LOG_ROOT=/home/rvelich/difix_logs

K11_NAME=difix_sdturbo_twoview_smoothgauss_k11_s5_bs1x4_lr2e5_200k_val250_2x100allpanels_gram0_p4d0_1110a
K21_NAME=difix_sdturbo_twoview_smoothgauss_k21_s10_bs1x4_lr2e5_200k_val250_2x100allpanels_gram0_p4d0_1110a

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
for run_name in "${K11_NAME}" "${K21_NAME}"; do
    if [[ -e "${RUN_ROOT}/${run_name}" ]]; then
        echo "Refusing to overwrite existing run: ${RUN_ROOT}/${run_name}" >&2
        exit 1
    fi
done

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
    --render_endpoint_mode=original
    --render_endpoint_smoothing_mode=gaussian
    --render_endpoint_smoothing_placement=after_combination
    --render_endpoint_smoothing_strength=1
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

CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON_BIN}" \
    -m accelerate.commands.launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=4 \
    --main_process_port=29671 \
    "${TRAIN_SCRIPT}" \
    --output_dir="${RUN_ROOT}/${K11_NAME}" \
    --tracker_run_name="${K11_NAME}" \
    --render_endpoint_smoothing_kernel_size=11 \
    --render_endpoint_smoothing_sigma=5 \
    "${COMMON_ARGS[@]}" \
    >"${LOG_ROOT}/smoothgauss_k11_s5_1110a.log" 2>&1 &
K11_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 "${PYTHON_BIN}" \
    -m accelerate.commands.launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=4 \
    --main_process_port=29672 \
    "${TRAIN_SCRIPT}" \
    --output_dir="${RUN_ROOT}/${K21_NAME}" \
    --tracker_run_name="${K21_NAME}" \
    --render_endpoint_smoothing_kernel_size=21 \
    --render_endpoint_smoothing_sigma=10 \
    "${COMMON_ARGS[@]}" \
    >"${LOG_ROOT}/smoothgauss_k21_s10_1110a.log" 2>&1 &
K21_PID=$!

printf 'k11_s5_pid=%s\nk21_s10_pid=%s\n' "${K11_PID}" "${K21_PID}"
wait "${K11_PID}" "${K21_PID}"
