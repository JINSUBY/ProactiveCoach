#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TRAIN_JSONL:?Set TRAIN_JSONL to your prepared training manifest}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to the directory containing your videos}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new training-output directory}"
# Select allocated GPUs externally via CUDA_VISIBLE_DEVICES.
export THINKSTREAM_MASK_Q_TILE=4096
export THINKSTREAM_TRITON_MAXBLOCK_X=16384
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train.py \
  --model "${MODEL:-Qwen/Qwen3.5-4B}" --model-type "${MODEL_TYPE:-qwen35}" \
  --train-jsonl "$TRAIN_JSONL" --video-root "$VIDEO_ROOT" --output-dir "$OUTPUT_DIR" \
  --batch-size "${MICROBATCH:-2}" --gradient-accumulation "${ACCUMULATION:-4}" \
  --deepspeed scripts/zero1.json "$@"
