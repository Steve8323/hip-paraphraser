"""Stage 2 - Minimal LoRA fine-tuning (paper §3.2).

Lightly adapt the BASE model on a -> h with loss on the human span only, so the
model's native (human-looking) continuation behaviour is preserved as much as
possible.  Wraps `mlx_lm.lora` with a generated YAML config.

The paper uses LoRA rank 128 / scale 128, lr 5e-5 cosine, 1 epoch, eff. batch 16
on 4xL40S.  We keep the spirit (minimal, low-distortion) with a smaller, stable
local config; override via flags.
"""
import argparse, math, os, subprocess, sys

import yaml

import config


def count_lines(path):
    return sum(1 for _ in open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=16, help="-1 = all transformer blocks")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--iters", type=int, default=0, help="0 = derive from epochs")
    args = ap.parse_args()

    train_path = os.path.join(config.DATA_DIR, "train.jsonl")
    if not os.path.exists(train_path):
        sys.exit("Missing data/train.jsonl - run make_data.py first.")
    n_train = count_lines(train_path)
    iters = args.iters or max(20, math.ceil(n_train * args.epochs / args.batch_size))

    cfg = {
        "model": config.BASE_MODEL,
        "train": True,
        "data": config.DATA_DIR,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "mask_prompt": True,                 # loss on the human span only
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": iters,
        "val_batches": args.val_batches,
        "learning_rate": args.lr,
        "steps_per_report": 10,
        "steps_per_eval": max(25, iters // 6),
        "max_seq_length": args.max_seq_length,
        "adapter_path": config.ADAPTER_PATH,
        "save_every": max(50, iters // 3),
        "grad_checkpoint": True,
        "seed": 0,
        "lora_parameters": {
            "rank": args.rank,
            "scale": args.scale,
            "dropout": args.dropout,
        },
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [args.lr, iters, args.lr * 0.1],
            "warmup": min(20, iters // 10),
        },
    }

    os.makedirs(config.ADAPTER_PATH, exist_ok=True)
    cfg_path = os.path.join(config.HERE, "lora_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[train] {n_train} pairs, {args.epochs} epochs -> {iters} iters, "
          f"rank {args.rank}, lr {args.lr}")
    print(f"[train] config -> {cfg_path}")
    subprocess.run(["mlx_lm.lora", "-c", cfg_path], check=True)
    print(f"[train] adapter saved -> {config.ADAPTER_PATH}")


if __name__ == "__main__":
    main()
