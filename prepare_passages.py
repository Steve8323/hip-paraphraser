"""Subscription path, step 1: select human passages (NO API key).

Loads + filters + dedups the human corpus and writes data/human_passages.jsonl.
The agent (Claude Code, on your subscription) then rewrites each into an
AI-style counterpart, instead of a paid API call. See assemble_pairs.py.
"""
import argparse, json, os, random

import config
import data_prep  # reuse load_wikitext / quality_ok / dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=140)
    ap.add_argument("--source", choices=["wikitext", "local"], default="wikitext")
    ap.add_argument("--file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    passages = data_prep.load_local(a.file) if a.source == "local" else data_prep.load_wikitext()
    passages = [p for p in passages if data_prep.quality_ok(p, a.min_words, a.max_words)]
    passages = data_prep.dedup(passages)
    random.shuffle(passages)
    passages = passages[: a.n]

    path = os.path.join(config.DATA_DIR, "human_passages.jsonl")
    with open(path, "w") as f:
        for i, h in enumerate(passages):
            f.write(json.dumps({"id": i, "h": h}) + "\n")
    print(f"wrote {len(passages)} human passages -> {path}")


if __name__ == "__main__":
    main()
