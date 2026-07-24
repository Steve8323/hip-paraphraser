"""Replicate the paper's Figure 1 (the title claim): base-model continuations of
a human prefix are judged HUMAN by GPTZero, while instruct-model continuations of
the same prefix are judged AI.

  generate : for N human prefixes, sample a continuation from the BASE model and
             from the INSTRUCT model (raw continuation, temp 1.0, top-p 0.95).
             No API key needed. -> continuations.jsonl
  score    : score each continuation with GPTZero -> fig1_results.json + summary.
             Needs GPTZERO_API_KEY.

Usage:
  python replicate_fig1.py generate --n 50
  GPTZERO_API_KEY=... python replicate_fig1.py score
"""
import argparse, json, os, re

import config

INSTRUCT_MODEL = os.environ.get("HIP_INSTRUCT_MODEL", "Qwen/Qwen3-1.7B")  # post-trained sibling
CONT_PATH = os.path.join(config.HERE, "continuations.jsonl")
FIG1_PATH = os.path.join(config.HERE, "fig1_results.json")


def first_sentence(t):
    m = re.search(r"(.+?[.!?])(\s|$)", t)
    return (m.group(1) if m else t[:200]).strip()


def _continuation(model, tok, prefix, max_tokens=200):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=1.0, top_p=0.95)
    out = generate(model, tok, prefix, max_tokens=max_tokens, sampler=sampler, verbose=False)
    return out.strip()


def cmd_generate(args):
    from mlx_lm import load
    src = os.path.join(config.DATA_DIR, "human_passages.jsonl")
    passages = [json.loads(l) for l in open(src)]
    prefixes = [first_sentence(p["h"]) for p in passages if len(p["h"].split()) >= 50]
    prefixes = [p for p in prefixes if len(p.split()) >= 5][: args.n]
    print(f"[fig1] {len(prefixes)} prefixes")

    rows = [{"prefix": p} for p in prefixes]

    print(f"[fig1] generating BASE continuations ({config.BASE_MODEL})")
    bm, bt = load(config.BASE_MODEL)
    for r in rows:
        r["base"] = _continuation(bm, bt, r["prefix"], args.max_tokens)
    del bm, bt
    try:
        import mlx.core as mx; mx.clear_cache()
    except Exception:
        pass

    print(f"[fig1] generating INSTRUCT continuations ({INSTRUCT_MODEL})")
    im, it = load(INSTRUCT_MODEL)
    for r in rows:
        r["instruct"] = _continuation(im, it, r["prefix"], args.max_tokens)

    with open(CONT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[fig1] wrote {len(rows)} -> {CONT_PATH}")
    print("[fig1] next:  GPTZERO_API_KEY=... python replicate_fig1.py score")


def _score_text(text, prefix):
    import gptzero
    # ensure enough length for the detector; prepend the prefix if the
    # continuation alone is short
    t = text if len(text.split()) >= 50 else (prefix + " " + text).strip()
    return gptzero.human_prob(t)


def cmd_score(args):
    rows = [json.loads(l) for l in open(CONT_PATH)][: args.n]
    out = []
    for i, r in enumerate(rows):
        rec = {"prefix": r["prefix"]}
        for which in ("base", "instruct"):
            try:
                rec[which + "_human_prob"] = _score_text(r[which], r["prefix"])
            except Exception as e:
                print(f"[fig1] score failed ({which}, {i}): {e}")
                rec[which + "_human_prob"] = None
        out.append(rec)
        b, ins = rec.get("base_human_prob"), rec.get("instruct_human_prob")
        print(f"[fig1] {i+1}/{len(rows)}: base P(human)={b}  instruct P(human)={ins}")

    def mean(key):
        v = [x[key] for x in out if x.get(key) is not None]
        return sum(v) / len(v) if v else None

    res = {
        "base_model": config.BASE_MODEL,
        "instruct_model": INSTRUCT_MODEL,
        "n": len(out),
        "mean_base_human_prob": mean("base_human_prob"),
        "mean_instruct_human_prob": mean("instruct_human_prob"),
        "items": out,
    }
    json.dump(res, open(FIG1_PATH, "w"), indent=2)
    print("\n===== FIGURE 1 REPLICATION (GPTZero) =====")
    print(f"base    ({config.BASE_MODEL}): mean P(human) = {res['mean_base_human_prob']}")
    print(f"instruct({INSTRUCT_MODEL}): mean P(human) = {res['mean_instruct_human_prob']}")
    print(f"-> {FIG1_PATH}")
    try:
        _render(res)
    except Exception as e:
        print("[fig1] render skipped:", e)


def _render(res):
    import base64, datetime, io, webbrowser
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b, i = res["mean_base_human_prob"], res["mean_instruct_human_prob"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(["base\n(this work)", "instruct\n(this work)"], [b, i],
                  color=["#1b7837", "#762a83"])
    # paper's Llama3-8B human-prefix reference points
    ax.axhline(0.967, ls="--", color="#1b7837", alpha=.5)
    ax.axhline(0.303, ls="--", color="#762a83", alpha=.5)
    ax.text(1.5, 0.967, "paper base 0.97", ha="right", va="bottom", fontsize=7, color="#1b7837")
    ax.text(1.5, 0.303, "paper instruct 0.30", ha="right", va="bottom", fontsize=7, color="#762a83")
    ax.set_ylim(0, 1); ax.set_ylabel("GPTZero P(human)")
    ax.set_title("Fig.1 replication: base looks human, instruct doesn't")
    for bar, v in zip(bars, [b, i]):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=10)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    png = base64.b64encode(buf.getvalue()).decode()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(config.HERE, f"fig1_report_{stamp}.html")
    open(out, "w").write(
        f"<!doctype html><meta charset=utf-8><title>Fig 1 replication</title>"
        f"<body style='font:15px sans-serif;max-width:720px;margin:32px auto'>"
        f"<h2>Base Models Look Human To GPTZero — Figure 1 replication</h2>"
        f"<p>base <code>{config.BASE_MODEL}</code> vs instruct <code>{INSTRUCT_MODEL}</code>, "
        f"n={res['n']} human-prefix continuations, scored by the GPTZero API.</p>"
        f"<img style='max-width:100%' src='data:image/png;base64,{png}'>"
        f"<p>mean GPTZero P(human): <b>base {b:.3f}</b> vs <b>instruct {i:.3f}</b>. "
        f"Paper (Llama3-8B, human prefix): base 0.967 vs instruct 0.303.</p></body>")
    print(f"[fig1] report -> {out}")
    webbrowser.open("file://" + out)


def cmd_probe(args):
    """Quota-safe check: score ONE item's base+instruct, dump the raw GPTZero
    response so a human can verify key/schema/values before the full batch.
    These 2 calls are cached and reused by `score`, so they cost nothing extra."""
    import gptzero
    r = json.loads(open(CONT_PATH).readline())
    for which in ("base", "instruct"):
        text = r[which] if len(r[which].split()) >= 50 else (r["prefix"] + " " + r[which]).strip()
        raw = gptzero.predict(text)
        doc = raw["documents"][0]
        print(f"\n===== PROBE: {which} =====")
        print("text[:200]:", text[:200])
        print("class_probabilities:", doc.get("class_probabilities"))
        print("predicted_class:", doc.get("predicted_class"),
              "| confidence:", doc.get("confidence_category"))
        print("parsed human_prob:", gptzero.human_prob(text))
    print("\n[probe] OK — verify the values look sane, then run: python replicate_fig1.py score")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=50)
    g.add_argument("--max-tokens", type=int, default=200); g.set_defaults(fn=cmd_generate)
    p = sub.add_parser("probe"); p.set_defaults(fn=cmd_probe)
    s = sub.add_parser("score"); s.add_argument("--n", type=int, default=10**9)
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
