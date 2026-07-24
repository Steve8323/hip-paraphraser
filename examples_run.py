"""Generate paraphrases of the same source text from SEVERAL base-model HIP
paraphrasers, so the outputs can be pasted into GPTZero and compared.

All paraphrasers are BASE models from the paper (Qwen3 size sweep).
-> paraphraser_examples.txt (human-readable) + paraphraser_examples.json
"""
import argparse, json, os

import config
import hip_gen
from mlx_lm import load

# (hf base model, adapter dir, display name) -- all BASE models per the paper
PARAS = [
    ("Qwen/Qwen3-0.6B-Base", "adapters_qwen3_0.6b", "Qwen3-0.6B-Base (HIP)"),
    ("Qwen/Qwen3-1.7B-Base", "adapters",            "Qwen3-1.7B-Base (HIP)"),
    ("Qwen/Qwen3-4B-Base",   "adapters_qwen3_4b",   "Qwen3-4B-Base (HIP)"),
]


def iterative(model, tok, text, n):
    return hip_gen.iterative(model, tok, text, n)


def _load_sources(path, n):
    if path:
        raw = open(path, encoding="utf-8").read()
        chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
        return chunks[:n] if n else chunks
    return [json.loads(l)["a"] for l in open(config.EVAL_PATH)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n-src", type=int, default=2)
    ap.add_argument("--src-file", help="paragraphs separated by blank lines")
    ap.add_argument("--only", help="comma-separated name substrings to keep, e.g. 0.6B,1.7B")
    ap.add_argument("--out", default="paraphraser_examples", help="output basename")
    args = ap.parse_args()

    srcs = _load_sources(args.src_file, args.n_src)
    paras = PARAS
    if args.only:
        keep = [s.strip() for s in args.only.split(",")]
        paras = [p for p in PARAS if any(k in p[2] for k in keep)]
    results = {"rounds": args.rounds, "sources": srcs, "by_para": {}}

    for base, adap, name in paras:
        if not os.path.isdir(adap):
            print(f"[examples] skip {name} (no adapter at {adap})")
            continue
        print(f"[examples] loading {name}")
        model, tok = load(base, adapter_path=adap)
        results["by_para"][name] = [iterative(model, tok, s, args.rounds) for s in srcs]
        del model, tok
        try:
            import mlx.core as mx; mx.clear_cache()
        except Exception:
            pass
        print(f"[examples] done {name}")

    lines = [
        "Base-model paraphrasers (HIP, arXiv:2605.19516) — paste each block into GPTZero.",
        f"Every paraphraser is a BASE model from the paper. {args.rounds} iterative rounds each.",
        "=" * 72,
    ]
    for i, s in enumerate(srcs):
        lines.append(f"\n\n########## SOURCE {i+1} — ORIGINAL AI TEXT ##########\n{s}")
        for name in results["by_para"]:
            lines.append(f"\n----- paraphraser: {name} -----\n{results['by_para'][name][i]}")

    open(os.path.join(config.HERE, args.out + ".txt"), "w").write("\n".join(lines))
    json.dump(results, open(os.path.join(config.HERE, args.out + ".json"), "w"), indent=2)
    print(f"[examples] wrote {args.out}.txt / {args.out}.json")


if __name__ == "__main__":
    main()
