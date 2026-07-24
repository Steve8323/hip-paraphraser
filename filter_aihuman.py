"""Curate the AI->human pairs with the detector before fine-tuning.

For each (a, h): score desklib P(human) on both and embedding cosine(a, h).
Keep pairs where the human target genuinely reads human AND meaning is preserved,
so the model learns the register that actually moves the detector.

  python filter_aihuman.py                # show distribution + write with defaults
  python filter_aihuman.py --min-hum 0.4 --min-cos 0.80 --n-eval 12
Writes data_aihuman/pairs.jsonl + data_aihuman/eval_pairs.jsonl.
"""
import argparse, glob, json, os, random, statistics as st

import detector_proxy
import test_quality

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_aihuman")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hum", type=float, default=0.4, help="min desklib P(human) of target h")
    ap.add_argument("--min-cos", type=float, default=0.80, help="min embedding cosine(a,h)")
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = []
    for part in sorted(glob.glob(os.path.join(DIR, "aihuman_part_*.jsonl"))):
        for line in open(part):
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[filter] {len(pairs)} raw pairs")

    sim, sim_name = test_quality._semantic_fn()
    scored = []
    for p in pairs:
        a, h = p["a"], p["h"]
        scored.append({"a": a, "h": h,
                       "pa": detector_proxy.human_prob(a),
                       "ph": detector_proxy.human_prob(h),
                       "cos": sim(a, h)})

    pa = [s["pa"] for s in scored]; ph = [s["ph"] for s in scored]; cos = [s["cos"] for s in scored]
    print(f"[filter] desklib P(human)  a: mean {st.mean(pa):.3f}  | h: mean {st.mean(ph):.3f} "
          f"median {st.median(ph):.3f}")
    print(f"[filter] cosine(a,h): mean {st.mean(cos):.3f} min {min(cos):.3f}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        n = sum(1 for s in scored if s["ph"] >= thr and s["cos"] >= args.min_cos)
        print(f"[filter]   keep if ph>={thr} & cos>={args.min_cos}: {n}")

    kept = [s for s in scored if s["ph"] >= args.min_hum and s["cos"] >= args.min_cos]
    random.Random(args.seed).shuffle(kept)
    print(f"[filter] KEEPING {len(kept)} pairs (ph>={args.min_hum}, cos>={args.min_cos})")

    ev, tr = kept[: args.n_eval], kept[args.n_eval :]
    with open(os.path.join(DIR, "pairs.jsonl"), "w") as f:
        for s in tr:
            f.write(json.dumps({"a": s["a"], "h": s["h"]}) + "\n")
    with open(os.path.join(DIR, "eval_pairs.jsonl"), "w") as f:
        for s in ev:
            f.write(json.dumps({"a": s["a"], "h": s["h"]}) + "\n")
    json.dump(scored, open(os.path.join(DIR, "scored_all.json"), "w"), indent=2)
    print(f"[filter] wrote {len(tr)} train + {len(ev)} eval to {DIR}/")


if __name__ == "__main__":
    main()
