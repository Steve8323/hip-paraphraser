"""Stage 1 - Data preparation (paper Algorithm 1).

Build paired data D = {(a_i, h_i)} where h_i is a human passage and a_i is an
AI-style rewrite of it.  The direction matters: we will train the base model to
map a_i -> h_i (machine-like back to human).

  human corpus -> filter by length -> normalize/dedup -> quality screen
              -> for each h: AIParaphrase(h)=a with bounded rejection sampling
                 (keep only if anomaly-free AND semantic-preservation >= thresh)

Default human corpus: wikitext-2-raw-v1 (real, ungated, tiny).  Swap with
  --source local --file passages.txt   (one passage per line, or blank-line
  separated paragraphs).
"""
import argparse, json, os, re, hashlib, random

import config
import llm_api


def _clean(p: str) -> str:
    return re.sub(r"\s+", " ", p).strip()


def load_wikitext(n_scan: int = 6000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    for row in ds:
        t = row["text"]
        if not t or t.strip().startswith("="):   # skip headings / blanks
            continue
        out.append(_clean(t))
        if len(out) >= n_scan:
            break
    return out


def load_local(path: str):
    raw = open(path, encoding="utf-8").read()
    chunks = re.split(r"\n\s*\n", raw) if "\n\n" in raw else raw.splitlines()
    return [_clean(c) for c in chunks if c.strip()]


def quality_ok(p: str, lo: int, hi: int) -> bool:
    w = p.split()
    if not (lo <= len(w) <= hi):
        return False
    letters = sum(c.isalpha() for c in p)
    if letters / max(1, len(p)) < 0.6:          # mostly real words
        return False
    if p.count(".") + p.count("!") + p.count("?") < 1:
        return False
    return True


def dedup(passages):
    seen, out = set(), []
    for p in passages:
        key = hashlib.md5(p.lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wikitext", "local"], default="wikitext")
    ap.add_argument("--file", help="path for --source local")
    ap.add_argument("--n", type=int, default=320, help="number of training pairs to build")
    ap.add_argument("--n-eval", type=int, default=16, help="held-out pairs for evaluation")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=160)
    ap.add_argument("--retries", type=int, default=3, help="K: paraphrase attempts per passage")
    ap.add_argument("--sem-thresh", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if args.source == "local":
        if not args.file:
            ap.error("--source local requires --file")
        passages = load_local(args.file)
    else:
        passages = load_wikitext()

    passages = [p for p in passages if quality_ok(p, args.min_words, args.max_words)]
    passages = dedup(passages)
    random.shuffle(passages)
    print(f"[data] {len(passages)} clean human passages available")

    want = args.n + args.n_eval
    pairs = []
    for h in passages:
        if len(pairs) >= want:
            break
        for _ in range(args.retries):                # bounded rejection sampling
            try:
                a = llm_api.ai_paraphrase(h)
            except Exception as e:
                print("[warn] paraphrase failed:", e)
                continue
            if not llm_api.anomaly_free(a, h):
                continue
            if llm_api.semantic_score(h, a) >= args.sem_thresh:
                pairs.append({"a": a, "h": h})
                break
        if len(pairs) % 20 == 0 and pairs:
            print(f"[data] {len(pairs)}/{want} pairs")

    random.shuffle(pairs)
    eval_pairs = pairs[: args.n_eval]
    train_pairs = pairs[args.n_eval :]

    with open(config.PAIRS_PATH, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    with open(config.EVAL_PATH, "w") as f:
        for p in eval_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"[data] wrote {len(train_pairs)} train pairs -> {config.PAIRS_PATH}")
    print(f"[data] wrote {len(eval_pairs)} eval  pairs -> {config.EVAL_PATH}")


if __name__ == "__main__":
    main()
