#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
if [[ -f "$SCRIPT_DIR/paths.local.sh" ]]; then source "$SCRIPT_DIR/paths.local.sh"; fi
: "${MODEL_NAME:?Set MODEL_NAME in paths.local.sh or your environment}"
: "${REDS_SMALL1:?Set REDS_SMALL1 in paths.local.sh or your environment}"
: "${REDS_SMALL2:?Set REDS_SMALL2 in paths.local.sh or your environment}"
: "${STEP1_CHECKPOINT:?Set STEP1_CHECKPOINT in paths.local.sh or your environment}"
: "${STEP2_VOXEL_VAE:?Set STEP2_VOXEL_VAE in paths.local.sh or your environment}"
: "${BS_ERGB_PATH:?Set BS_ERGB_PATH in paths.local.sh or your environment}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/step3}"
export OUTPUT_LOG="${OUTPUT_LOG:-logs/step3.log}"
export LOG_NAME="${LOG_NAME:-evdiff_step3}"
export DEFAULT_EMBED="${DEFAULT_EMBED:-dataset/default}"
export NULL_EMBED="${NULL_EMBED:-dataset/null}"
mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_LOG")"
LORA_DIR="$STEP1_CHECKPOINT"
INIT_VOXEL_VAE_PATH="$STEP2_VOXEL_VAE"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for training"'
accelerate launch --config_file config/training/accelerate.yaml --num_processes 1 --mixed_precision=no --module training.train_color_vae_raft_v5_EVDiff \
  --train_batch_size=1 --lora_dir="${LORA_DIR}" --init_voxel_vae_path="${INIT_VOXEL_VAE_PATH}" --pretrained_model_name_or_path="${MODEL_NAME}" \
  --output_channels=16 --base_channels=96 --input_channels=5 --use_default_prompt --temporal_fusion="gated" \
  --num_train_epochs=200 --checkpointing_steps=2000 --validation_steps=500 --max_train_steps=60000 --dataloader_num_workers=4 \
  --learning_rate=5e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=500 \
  --seed=43 --use_random_bias --rank=64 \
  --output_dir="${OUTPUT_DIR}" \
  --report_to="${REPORT_TO:-tensorboard}" --log_name="${LOG_NAME}" \
  --gradient_accumulation_steps=1 \
  --resume_from_checkpoint="latest" \
  --E2VID_sequence_length 38 \
  --E2VID_skip 2 "$@" 2>&1 | tee "$OUTPUT_LOG"
