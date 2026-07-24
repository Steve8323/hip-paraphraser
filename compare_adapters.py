"""Head-to-head: does the AI->human-trained adapter humanize AI essays better
than the wiki-trained adapter? Both run one-pass best-of-N (desklib selector +
semantic floor) on the held-out AI essays.
"""
import json, os, statistics as st

import config
import hip_gen
import detector_proxy
import test_quality
from mlx_lm import load

ADAPTERS = [
    ("adapters",          "wiki-trained (old)"),
    ("adapters_aihuman",  "AI->human (new)"),
]
K = 10
MIN_SEM = 0.80


def main():
    inputs = [c.strip() for c in open("aihuman_eval_inputs.txt").read().split("\n\n") if c.strip()]
    sim, _ = test_quality._semantic_fn()
    p_in = [detector_proxy.human_prob(x) for x in inputs]

    print(f"{len(inputs)} held-out AI essays | input mean P(human) = {st.mean(p_in):.3f}\n")
    results = {}
    for adap, name in ADAPTERS:
        if not os.path.isdir(adap):
            print("skip", name); continue
        model, tok = load(config.BASE_MODEL, adapter_path=adap)
        outs, sems = [], []
        sample = None
        for x in inputs:
            best, info = hip_gen.best_of_round(model, tok, x, k=K, temp=1.1, top_p=0.97,
                                               scorer=detector_proxy.human_prob,
                                               sem_scorer=sim, min_sem=MIN_SEM)
            outs.append(info["phuman"] if info["phuman"] is not None else 0.0)
            sems.append(info["semantic"] if info["semantic"] is not None else 0.0)
            if sample is None:
                sample = best
        results[name] = {"out": st.mean(outs), "sem": st.mean(sems), "sample": sample}
        del model, tok
        try:
            import mlx.core as mx; mx.clear_cache()
        except Exception:
            pass
        print(f"{name:24} mean P(human) out = {st.mean(outs):.3f} | mean semantic = {st.mean(sems):.3f}")

    print(f"\n(input mean P(human) was {st.mean(p_in):.3f}; best-of-{K}, semantic floor {MIN_SEM})")
    for name, r in results.items():
        print(f"\n===== sample output — {name} (P(human) {r['out']:.2f}) =====\n{r['sample'][:600]}")
    json.dump({"input_mean": st.mean(p_in), "results": {k: {kk: vv for kk, vv in v.items() if kk != 'sample'}
              for k, v in results.items()}}, open("compare_adapters_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
