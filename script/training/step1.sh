#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
if [[ -f "$SCRIPT_DIR/paths.local.sh" ]]; then source "$SCRIPT_DIR/paths.local.sh"; fi
: "${MODEL_NAME:?Set MODEL_NAME in paths.local.sh or your environment}"
: "${PLACES365_PATH:?Set PLACES365_PATH in paths.local.sh or your environment}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/step1}"
export OUTPUT_LOG="${OUTPUT_LOG:-logs/step1.log}"
export LOG_NAME="${LOG_NAME:-evdiff_step1}"
export DEFAULT_EMBED="${DEFAULT_EMBED:-dataset/default}"
export NULL_EMBED="${NULL_EMBED:-dataset/null}"
mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_LOG")"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for training"'
accelerate launch --config_file config/training/accelerate.yaml --num_processes 1 --mixed_precision=fp16 --module training.train_no_650 \
  --pretrained_model_name_or_path="${MODEL_NAME}" \
  --default_embedding_dir="${DEFAULT_EMBED}" --null_embedding_dir="${NULL_EMBED}" \
  --train_batch_size=8 --rank=64 --rank_vae=64 --rank_lora=64 \
  --num_train_epochs=200 --checkpointing_steps=5000 --validation_steps=500 --max_train_steps=400000 --dataloader_num_workers=4 \
  --learning_rate=5e-06 --learning_rate_reg=1e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=3000 \
  --seed=43 --use_default_prompt --use_random_bias \
  --output_dir="${OUTPUT_DIR}" \
  --report_to="${REPORT_TO:-tensorboard}" --log_name="${LOG_NAME}" \
  --gradient_accumulation_steps=4 \
  --resume_from_checkpoint="latest" \
  --guidance_scale=7.5 "$@" 2>&1 | tee "$OUTPUT_LOG"
