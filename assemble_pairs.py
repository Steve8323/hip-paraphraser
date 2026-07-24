"""Subscription path, step 2: assemble agent-written (a,h) pairs.

The agent writes each batch of rewrites to data/pairs_part_*.jsonl
({"a": ai_rewrite, "h": human_original}).  This shuffles, splits off a held-out
eval set, and produces the same pairs.jsonl / eval_pairs.jsonl the rest of the
pipeline expects -- identical downstream to the API path.
"""
import argparse, glob, json, os, random

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = []
    for part in sorted(glob.glob(os.path.join(config.DATA_DIR, "pairs_part_*.jsonl"))):
        for line in open(part):
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p.get("a") and p.get("h") and len(p["a"].split()) >= 8:
                r = len(p["a"].split()) / max(1, len(p["h"].split()))
                if 0.4 <= r <= 2.5:
                    pairs.append({"a": p["a"].strip(), "h": p["h"].strip()})

    random.Random(a.seed).shuffle(pairs)
    ev, tr = pairs[: a.n_eval], pairs[a.n_eval :]

    with open(config.PAIRS_PATH, "w") as f:
        for p in tr:
            f.write(json.dumps(p) + "\n")
    with open(config.EVAL_PATH, "w") as f:
        for p in ev:
            f.write(json.dumps(p) + "\n")

    print(f"assembled {len(pairs)} pairs -> {len(tr)} train, {len(ev)} eval")


if __name__ == "__main__":
    main()
