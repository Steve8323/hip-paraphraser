#!/usr/bin/env bash
# Train HIP adapters on additional Qwen3 BASE sizes from the paper, reusing the
# existing normalized data/ (train.jsonl/valid.jsonl). 1.7B already lives in adapters/.
set -euo pipefail
cd "$(dirname "$0")"

echo "############## Qwen3-0.6B-Base ##############"
HIP_BASE_MODEL=Qwen/Qwen3-0.6B-Base HIP_ADAPTER_PATH=adapters_qwen3_0.6b \
  python3 train.py --epochs 3 --batch-size 4 --num-layers 16

echo "############## Qwen3-4B-Base ##############"
HIP_BASE_MODEL=Qwen/Qwen3-4B-Base HIP_ADAPTER_PATH=adapters_qwen3_4b \
  python3 train.py --epochs 3 --batch-size 2 --num-layers 8 --max-seq-length 768

echo "ALL EXTRA TRAININGS DONE"
