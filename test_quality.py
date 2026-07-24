"""Confirm the paraphrasers locally (no GPTZero key):
  - human-likeness: desklib proxy P(human), source vs paraphrase (should rise)
  - meaning kept:   embedding cosine (sentence-transformers) source vs paraphrase
Writes test_quality_results.json + a timestamped HTML report (opened).
"""
import argparse, base64, datetime, html, io, json, os, re, webbrowser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import detector_proxy


def _semantic_fn():
    try:
        from sentence_transformers import SentenceTransformer, util
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        def sim(a, b):
            e = m.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
            return float(util.cos_sim(e[0], e[1]))
        return sim, "embedding cosine (all-MiniLM-L6-v2)"
    except Exception as e:
        print("[test] sentence-transformers unavailable, using word-overlap:", e)
        W = re.compile(r"[a-z]{3,}")
        def sim(a, b):
            sa, sb = set(W.findall(a.lower())), set(W.findall(b.lower()))
            return len(sa & sb) / max(1, len(sa | sb))
        return sim, "content-word Jaccard (fallback)"


def run(examples_path):
    data = json.load(open(examples_path))
    srcs, by = data["sources"], data["by_para"]
    sim, sim_name = _semantic_fn()

    rows = []
    for name, outs in by.items():
        for i, (s, o) in enumerate(zip(srcs, outs)):
            hp_s, hp_o = detector_proxy.human_prob(s), detector_proxy.human_prob(o)
            rows.append({
                "model": name, "src": i, "src_text": s, "out_text": o,
                "phuman_src": hp_s, "phuman_out": hp_o,
                "d_phuman": (hp_o - hp_s) if (hp_s is not None and hp_o is not None) else None,
                "semantic": sim(s, o),
            })

    agg = {}
    for name in by:
        r = [x for x in rows if x["model"] == name]
        def mean(k):
            v = [x[k] for x in r if x[k] is not None]
            return sum(v) / len(v) if v else None
        agg[name] = {"phuman_src": mean("phuman_src"), "phuman_out": mean("phuman_out"),
                     "d_phuman": mean("d_phuman"), "semantic": mean("semantic"), "n": len(r)}

    return {"examples": examples_path, "rounds": data.get("rounds"),
            "semantic_metric": sim_name, "agg": agg, "rows": rows}


def figure(res):
    names = list(res["agg"])
    xs = range(len(names))
    src = [res["agg"][n]["phuman_src"] for n in names]
    out = [res["agg"][n]["phuman_out"] for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 4))
    w = 0.38
    ax.bar([x - w/2 for x in xs], src, w, label="original AI text", color="#9970ab")
    ax.bar([x + w/2 for x in xs], out, w, label="after paraphrase", color="#1b7837")
    ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("proxy P(human)"); ax.set_ylim(0, 1)
    ax.set_title("Humanization (desklib proxy): paraphrase vs original")
    ax.legend(fontsize=8)
    for x, v in zip(xs, out):
        ax.text(x + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def report(res, no_open=False):
    png = figure(res)
    esc = html.escape
    head = ("<tr><th>model</th><th>P(human) src</th><th>P(human) out</th>"
            "<th>ΔP(human)</th><th>semantic</th></tr>")
    arows = ""
    for n, x in res["agg"].items():
        arows += (f"<tr><td>{esc(n)}</td><td>{x['phuman_src']:.3f}</td>"
                  f"<td>{x['phuman_out']:.3f}</td><td>{x['d_phuman']:+.3f}</td>"
                  f"<td>{x['semantic']:.3f}</td></tr>")
    samples = ""
    for r in res["rows"]:
        samples += (f"<div class=card><div class=meta>{esc(r['model'])} · source {r['src']+1} · "
                    f"P(human) {r['phuman_src']:.2f} → <b>{r['phuman_out']:.2f}</b> · "
                    f"semantic {r['semantic']:.2f}</div>"
                    f"<div class=o>{esc(r['out_text'])}</div></div>")
    doc = f"""<!doctype html><meta charset=utf-8><title>HIP quality test</title>
<style>body{{font:15px/1.55 -apple-system,Segoe UI,sans-serif;max-width:900px;margin:30px auto;padding:0 16px}}
table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #ddd;padding:5px 12px;text-align:center}}
img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}
.card{{border:1px solid #e3e3e3;border-radius:8px;padding:10px;margin:10px 0;background:#fafafa}}
.meta{{font-size:12px;color:#1b7837;font-weight:600;margin-bottom:5px}} .o{{white-space:pre-wrap}}
code{{background:#f0f0f0;padding:1px 5px;border-radius:4px}}</style>
<h2>HIP paraphraser — local quality test</h2>
<p class=sub>{res['rounds']} rounds · human-likeness = desklib proxy P(human) (GPTZero stand-in) ·
meaning = {esc(res['semantic_metric'])}. Higher P(human) and higher semantic are both better.</p>
<img src="data:image/png;base64,{png}">
<table>{head}{arows}</table>
<h3>Paraphrased outputs</h3>{samples}"""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(config.HERE, f"quality_report_{stamp}.html")
    open(out, "w").write(doc)
    print("[test] report ->", out)
    if not no_open:
        webbrowser.open("file://" + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="paraphraser_examples_large.json")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    res = run(args.examples)
    json.dump(res, open(os.path.join(config.HERE, "test_quality_results.json"), "w"), indent=2)
    print(f"\n=== HIP quality test ({res['semantic_metric']}) ===")
    print(f"{'model':26}{'P(hum)src':>11}{'P(hum)out':>11}{'dP(hum)':>10}{'semantic':>10}")
    for n, x in res["agg"].items():
        print(f"{n:26}{x['phuman_src']:>11.3f}{x['phuman_out']:>11.3f}{x['d_phuman']:>+10.3f}{x['semantic']:>10.3f}")
    report(res, args.no_open)


if __name__ == "__main__":
    main()
