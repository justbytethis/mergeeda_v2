#!/bin/bash
# LoRA SFT launcher for the AMBA spec Question dataset.
#
# Usage:
#   scripts/train/train.sh <model_name_or_path> [lora_rank] [options]
#
# Options:
#   --epochs N        num_train_epochs (default: 5)
#   --lr LR           learning_rate (default: 2e-5)
#   --batch-size N    per_device_train_batch_size (default: auto)
#   --grad-accum N    gradient_accumulation_steps (default: auto)
#   --max-len N       model_max_length (default: 4096)
#   --dataset NAME    dataset_use token (default: amba_sft)
#   --output-dir DIR  output directory (default: auto)
#
# Env vars:
#   FRESH=1           force a new run instead of auto-resuming
#   RUN_TAG           override the timestamp tag

set -euo pipefail

cd "$(dirname "$0")/../.."

usage() {
  echo "Usage: $0 <model_name_or_path> [lora_rank] [--epochs N] [--lr LR] [--batch-size N] [--grad-accum N] [--max-len N] [--dataset NAME] [--output-dir DIR]" >&2
  exit 2
}

[[ ${#} -lt 1 ]] && usage

MODEL_NAME_OR_PATH="$1"; shift
LORA_RANK=128
EPOCHS=5
LR=2e-5
BATCH_SIZE=""
GRAD_ACCUM=""
MAX_LEN=4096
DATASET="amba_sft"
OUTPUT_DIR_OVERRIDE=""

# Parse positional lora_rank if next arg is a plain integer
if [[ ${#} -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  LORA_RANK="$1"; shift
fi

while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --epochs)     EPOCHS="$2";           shift 2 ;;
    --lr)         LR="$2";               shift 2 ;;
    --batch-size) BATCH_SIZE="$2";       shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2";       shift 2 ;;
    --max-len)    MAX_LEN="$2";          shift 2 ;;
    --dataset)    DATASET="$2";          shift 2 ;;
    --output-dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

LORA_ALPHA=$(( LORA_RANK * 2 ))
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

model_base="$(basename "${MODEL_NAME_OR_PATH%/}")"
slug="${model_base,,}"
slug="${slug%-instruct}"
slug="$(echo "$slug" | sed -E 's/[^a-z0-9._-]+/-/g; s/-+/-/g; s/^-|-$//g')"

# Auto-resume: reuse newest ./outputs/<slug>-lora-<rank>-<dataset>-* with a checkpoint.
# Pass FRESH=1 to force a new run.
latest_resumable=""
if [[ "${FRESH:-0}" != "1" ]]; then
  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    if compgen -G "$d/checkpoint-*" > /dev/null; then
      latest_resumable="$d"; break
    fi
  done < <(ls -1dt "./data/outputs/${slug}-lora-${LORA_RANK}-${DATASET}-"* 2>/dev/null)
fi

if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
elif [[ -n "$latest_resumable" ]]; then
  OUTPUT_DIR="$latest_resumable"
  echo "[train.sh] Resuming into existing dir: $OUTPUT_DIR"
else
  OUTPUT_DIR="./data/outputs/${slug}-lora-${LORA_RANK}-${DATASET}-${RUN_TAG}"
  echo "[train.sh] Fresh run in: $OUTPUT_DIR"
fi

model_lc="${MODEL_NAME_OR_PATH,,}"

# Default batch size / grad accum by model size
if [[ -z "$BATCH_SIZE" ]]; then
  [[ "$model_lc" == *"1.7b"* ]] && BATCH_SIZE=4 || BATCH_SIZE=2
fi
if [[ -z "$GRAD_ACCUM" ]]; then
  [[ "$model_lc" == *"1.7b"* ]] && GRAD_ACCUM=4 || GRAD_ACCUM=8
fi

extra_args=()
if [[ "$model_lc" == *"vl"* ]]; then
  extra_args+=(--tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True)
fi

gpu_count_from_visible_devices() {
  local cvd="${CUDA_VISIBLE_DEVICES:-}"
  [[ -z "$cvd" ]] && return 1
  local cleaned="${cvd// /}"
  [[ "$cleaned" == "-1" ]] && { echo 0; return 0; }
  local count=0
  IFS=',' read -r -a devs <<< "$cleaned"
  for dev in "${devs[@]}"; do [[ -n "$dev" ]] && ((count += 1)); done
  echo "$count"
}

if gpu_count="$(gpu_count_from_visible_devices 2>/dev/null)"; then
  num_gpus="$gpu_count"
else
  num_gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
[[ -z "$num_gpus" || "$num_gpus" -lt 1 ]] && num_gpus=1

launcher=(python)
if [[ "$num_gpus" -gt 1 ]]; then
  launcher=(torchrun --standalone --nproc_per_node "$num_gpus")
  extra_args+=(--ddp_find_unused_parameters False)
fi

"${launcher[@]}" qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_use "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 10 \
  --bf16 \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --model_max_length "$MAX_LEN" \
  --lora_enable True \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout 0.05 \
  --report_to wandb \
  --run_name "${slug}-lora-${LORA_RANK}-${DATASET}-${RUN_TAG}" \
  "${extra_args[@]}"
