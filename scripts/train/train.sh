#!/bin/bash
# LoRA SFT launcher for the AMBA spec Question dataset.
#
# Prereq:
#   1) `python scripts/augmentation/generate_train_data.py` to produce per-file answers and the merged annotation JSON.
#
# Usage:
#   scripts/train/train.sh <model_name_or_path> [lora_rank]
# Examples:
#   scripts/train/train.sh Qwen/Qwen3-VL-4B-Instruct
#   scripts/train/train.sh Qwen/Qwen3-VL-4B-Instruct 64
#
# The AMBA dataset entry (annotation_path + data_path) is registered under the
# token `amba_sft` in qwen-vl-finetune/qwenvl/data/__init__.py.

set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ${#} -lt 1 ]]; then
  echo "Usage: $0 <model_name_or_path> [lora_rank]" >&2
  echo "Example: $0 Qwen/Qwen3-VL-4B-Instruct 64" >&2
  exit 2
fi

MODEL_NAME_OR_PATH="$1"
LORA_RANK="${2:-128}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

if ! [[ "$LORA_RANK" =~ ^[0-9]+$ ]] || [[ "$LORA_RANK" -lt 1 ]]; then
  echo "Invalid lora_rank: $LORA_RANK (must be a positive integer)" >&2
  exit 2
fi

model_base="$(basename "${MODEL_NAME_OR_PATH%/}")"
slug="${model_base,,}"
slug="${slug%-instruct}"
slug="$(echo "$slug" | sed -E 's/[^a-z0-9._-]+/-/g; s/-+/-/g; s/^-|-$//g')"

# Auto-resume: reuse the newest ./outputs/<slug>-lora-<rank>-amba-* dir that
# has a checkpoint-*. Pass FRESH=1 to force a new run instead.
latest_resumable=""
if [[ "${FRESH:-0}" != "1" ]]; then
  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    if compgen -G "$d/checkpoint-*" > /dev/null; then
      latest_resumable="$d"
      break
    fi
  done < <(ls -1dt ./outputs/${slug}-lora-${LORA_RANK}-amba-* 2>/dev/null)
fi

if [[ -n "$latest_resumable" ]]; then
  OUTPUT_DIR="$latest_resumable"
  echo "[train.sh] Resuming into existing dir: $OUTPUT_DIR"
else
  OUTPUT_DIR="./outputs/${slug}-lora-${LORA_RANK}-amba-${RUN_TAG}"
  echo "[train.sh] Fresh run in: $OUTPUT_DIR"
fi

model_lc="${MODEL_NAME_OR_PATH,,}"
is_vl=false
if [[ "$model_lc" == *"vl"* ]]; then
  is_vl=true
fi

per_device_train_batch_size=1
gradient_accumulation_steps=16
if [[ "$model_lc" == *"1.7b"* ]]; then
  per_device_train_batch_size=2
  gradient_accumulation_steps=8
fi

extra_args=()
if $is_vl; then
  # Train LLM side of the VLM by default; keep vision encoder + projector frozen.
  extra_args+=(--tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True)
fi

gpu_count_from_visible_devices() {
  local cvd="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$cvd" ]]; then
    return 1
  fi
  local cleaned="${cvd// /}"
  if [[ "$cleaned" == "-1" ]]; then
    echo 0
    return 0
  fi
  local count=0
  IFS=',' read -r -a devs <<< "$cleaned"
  for dev in "${devs[@]}"; do
    [[ -n "$dev" ]] && ((count += 1))
  done
  echo "$count"
  return 0
}

if gpu_count="$(gpu_count_from_visible_devices 2>/dev/null)"; then
  num_gpus="$gpu_count"
else
  num_gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ -z "$num_gpus" || "$num_gpus" -lt 1 ]]; then
  num_gpus=1
fi

launcher=(python)
if [[ "$num_gpus" -gt 1 ]]; then
  launcher=(torchrun --standalone --nproc_per_node "$num_gpus")
  extra_args+=(--ddp_find_unused_parameters False)
fi

"${launcher[@]}" qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_use "amba_sft" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 5 \
  --per_device_train_batch_size "$per_device_train_batch_size" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.0 \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 10 \
  --bf16 \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --model_max_length 4096 \
  --lora_enable True \
  --lora_r "$LORA_RANK" \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  "${extra_args[@]}"
