#!/usr/bin/env bash
# End-to-end HIP pipeline (data -> train -> evaluate). Local, MLX, M4.
set -euo pipefail
cd "$(dirname "$0")"

: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY first (export ANTHROPIC_API_KEY=sk-ant-...)}"

echo "== 1/4  build (a,h) pairs with Claude Haiku =="
python3 data_prep.py "$@"

echo "== 2/4  convert to mlx_lm completions format =="
python3 make_data.py

echo "== 3/4  LoRA fine-tune Qwen3-1.7B-Base (loss on human span only) =="
python3 train.py

echo "== 4/4  iterative paraphrase + score + HTML report =="
python3 evaluate.py

echo "done."
