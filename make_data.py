"""Convert (a, h) pairs into mlx_lm 'completions' format for LoRA training.

Each record is {"prompt": <source/target prefix>, "completion": <human + close tag>}.
With `mlx_lm.lora --mask-prompt`, loss is computed on the completion only -- i.e.
the model is optimized to reconstruct the human passage h conditioned on the AI
paraphrase a, exactly as in the paper (§3.2).
"""
import json, os, random

import config


def main(valid_frac: float = 0.1, seed: int = 0):
    pairs = [json.loads(l) for l in open(config.PAIRS_PATH)]
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * valid_frac))
    val, train = pairs[:n_val], pairs[n_val:]

    def dump(rows, name):
        path = os.path.join(config.DATA_DIR, name)
        with open(path, "w") as f:
            for p in rows:
                f.write(json.dumps({
                    "prompt": config.build_prompt(p["a"]),
                    "completion": config.build_completion(p["h"]),
                }) + "\n")
        return path

    print("[mlx] train ->", dump(train, "train.jsonl"), f"({len(train)})")
    print("[mlx] valid ->", dump(val, "valid.jsonl"), f"({len(val)})")


if __name__ == "__main__":
    main()
