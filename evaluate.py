"""Evaluate HIP: run iterative paraphrasing on held-out AI inputs and track,
per round, (i) semantic preservation vs the original input (Claude Haiku judge,
0-10) and (ii) human-likeness (local proxy detector, P(human)).

Outputs eval_results.json, a figure, and a timestamped HTML report (opened).
"""
import argparse, base64, datetime, html, io, json, os, webbrowser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import paraphrase
import detector_proxy


def _semantic(original, rewrite, backend):
    if backend == "none":
        return None
    try:
        import llm_api
        return llm_api.semantic_score(original, rewrite)
    except Exception as e:
        print("[eval] semantic scoring unavailable:", e)
        return None


def run(n_inputs, rounds, semantic="none"):
    pairs = [json.loads(l) for l in open(config.EVAL_PATH)][:n_inputs]
    trajectories, sem_by_round, hum_by_round = [], [], []
    href_hum = []  # human-reference detector score, for context

    for k, p in enumerate(pairs):
        x0, h = p["a"], p["h"]
        traj = paraphrase.iterative(x0, n=rounds)
        sem = [_semantic(x0, x, semantic) for x in traj]
        hum = [detector_proxy.human_prob(x) for x in traj]
        trajectories.append({"input": x0, "human_ref": h, "rounds": traj,
                             "semantic": sem, "human_prob": hum})
        sem_by_round.append(sem)
        hum_by_round.append(hum)
        href_hum.append(detector_proxy.human_prob(h))
        print(f"[eval] input {k+1}/{len(pairs)}: "
              f"P(human) {hum[0]} -> {hum[-1]}, semantic {sem[0]} -> {sem[-1]}")

    def col_mean(rows, j):
        vals = [r[j] for r in rows if j < len(r) and r[j] is not None]
        return sum(vals) / len(vals) if vals else None

    R = rounds + 1
    mean_sem = [col_mean(sem_by_round, j) for j in range(R)]
    mean_hum = [col_mean(hum_by_round, j) for j in range(R)]
    ref_hum = [v for v in href_hum if v is not None]
    return {
        "model": config.BASE_MODEL,
        "n_inputs": len(pairs),
        "rounds": rounds,
        "mean_semantic": mean_sem,
        "mean_human_prob": mean_hum,
        "human_ref_mean_prob": (sum(ref_hum) / len(ref_hum)) if ref_hum else None,
        "trajectories": trajectories,
    }


def make_figure(res):
    xs = list(range(res["rounds"] + 1))
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()

    hum = res["mean_human_prob"]
    sem = res["mean_semantic"]
    if any(v is not None for v in hum):
        ax1.plot(xs, [v if v is not None else float("nan") for v in hum],
                 "o-", color="#1b7837", label="P(human) — proxy detector")
    ref = res.get("human_ref_mean_prob")
    if ref is not None:
        ax1.axhline(ref, ls="--", color="#1b7837", alpha=0.5,
                    label=f"human reference ({ref:.2f})")
    if any(v is not None for v in sem):
        ax2.plot(xs, [v if v is not None else float("nan") for v in sem],
                 "s-", color="#762a83", label="semantic (0–10)")

    ax1.set_xlabel("paraphrase round")
    ax1.set_ylabel("P(human)", color="#1b7837")
    ax1.set_ylim(0, 1)
    ax2.set_ylabel("semantic preservation (0–10)", color="#762a83")
    ax2.set_ylim(0, 10)
    ax1.set_title("HIP: humanness rises, semantics decays, with rounds")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_html(res, png_b64, path):
    def esc(s):
        return html.escape(s)

    def fmt(v, nd=2):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    rows = ""
    for j in range(res["rounds"] + 1):
        rows += (f"<tr><td>{j}</td><td>{fmt(res['mean_human_prob'][j])}</td>"
                 f"<td>{fmt(res['mean_semantic'][j], 1)}</td></tr>")

    samples = ""
    for t in res["trajectories"][:3]:
        steps = ""
        for j, x in enumerate(t["rounds"]):
            tag = "input" if j == 0 else f"round {j}"
            hp = fmt(t["human_prob"][j]) if j < len(t["human_prob"]) else "—"
            sm = fmt(t["semantic"][j], 0) if j < len(t["semantic"]) else "—"
            steps += (f"<div class='step'><div class='meta'>{tag} · "
                      f"P(human) {hp} · semantic {sm}</div>"
                      f"<div class='txt'>{esc(x)}</div></div>")
        samples += (f"<div class='traj'><div class='ref'><b>human reference:</b> "
                    f"{esc(t['human_ref'])}</div>{steps}</div>")

    doc = f"""<!doctype html><meta charset=utf-8>
<title>HIP evaluation</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;
      margin:32px auto;padding:0 18px;color:#1a1a1a}}
 h1{{font-size:22px}} .sub{{color:#666}}
 img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}
 table{{border-collapse:collapse;margin:14px 0}} td,th{{border:1px solid #ddd;padding:5px 12px;text-align:center}}
 .traj{{border:1px solid #e3e3e3;border-radius:10px;padding:14px;margin:16px 0;background:#fafafa}}
 .ref{{font-size:13px;color:#444;margin-bottom:10px}}
 .step{{margin:8px 0;padding:8px 10px;background:#fff;border-radius:8px;border:1px solid #eee}}
 .meta{{font-size:12px;color:#1b7837;font-weight:600;margin-bottom:4px}}
 .txt{{font-size:14px;white-space:pre-wrap}}
 code{{background:#f0f0f0;padding:1px 5px;border-radius:4px}}
</style>
<h1>HIP — Humanization by Iterative Paraphrasing</h1>
<p class="sub">Base model <code>{esc(res['model'])}</code> · {res['n_inputs']} held-out AI
inputs · {res['rounds']} rounds · arXiv:2605.19516.<br>
Human-likeness = local desklib proxy P(human) (GPTZero/Pangram stand-in);
semantic = Claude Haiku 0–10 judge vs the original input.</p>
<img src="data:image/png;base64,{png_b64}">
<h3>Per-round means</h3>
<table><tr><th>round</th><th>P(human)</th><th>semantic</th></tr>{rows}</table>
<h3>Sample trajectories</h3>
{samples}
"""
    with open(path, "w") as f:
        f.write(doc)


def recompute_means(res):
    """Rebuild per-round means from (possibly newly filled) trajectories."""
    def col_mean(key, j):
        vals = [t[key][j] for t in res["trajectories"]
                if j < len(t[key]) and t[key][j] is not None]
        return sum(vals) / len(vals) if vals else None
    R = res["rounds"] + 1
    res["mean_semantic"] = [col_mean("semantic", j) for j in range(R)]
    res["mean_human_prob"] = [col_mean("human_prob", j) for j in range(R)]
    return res


def render(res, no_open=False):
    png = make_figure(res)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(config.HERE, f"report_{stamp}.html")
    make_html(res, png, out)
    print(f"[eval] report -> {out}")
    if not no_open:
        webbrowser.open("file://" + out)
    return out


RESULTS_PATH = None  # set in main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=config.N_ROUNDS)
    ap.add_argument("--semantic", choices=["none", "api"], default="none",
                    help="'none' for the subscription path (agent fills scores later)")
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild the report from eval_results.json (after semantics are filled)")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    results_path = os.path.join(config.HERE, "eval_results.json")
    if args.render_only:
        res = recompute_means(json.load(open(results_path)))
    else:
        res = run(args.n_inputs, args.rounds, semantic=args.semantic)
        with open(results_path, "w") as f:
            json.dump(res, f, indent=2)
    render(res, no_open=args.no_open)


if __name__ == "__main__":
    main()
