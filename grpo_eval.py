"""Did GRPO amortize best-of-N into the weights? Compare a SINGLE GREEDY pass
(temp 0, no search) from the wiki adapter vs the GRPO adapter on the wiki-style
inputs. Success = GRPO single-pass P(human) >> wiki single-pass.
"""
import json, statistics as st

import config
import hip_gen
import detector_proxy
import test_quality
from mlx_lm import load

INPUTS = [json.loads(l)["a"] for l in open("data/eval_pairs.jsonl")]
ADAPTERS = [("adapters", "wiki (GRPO start)"), ("adapters_grpo", "GRPO")]


def main():
    sim, _ = test_quality._semantic_fn()
    p_in = [detector_proxy.human_prob(x) for x in INPUTS]
    print(f"{len(INPUTS)} wiki-style inputs | input mean P(human) = {st.mean(p_in):.3f}\n")
    samples = {}
    for adap, name in ADAPTERS:
        model, tok = load(config.BASE_MODEL, adapter_path=adap)
        ph, sm = [], []
        for i, x in enumerate(INPUTS):
            cap = min(400, int(len(tok.encode(x)) * 1.6) + 64)
            y = hip_gen.paraphrase_once(model, tok, x, cap, temp=0.0, top_p=1.0)  # greedy, 1 pass
            ph.append(detector_proxy.human_prob(y))
            sm.append(sim(x, y))
            if i == 1:
                samples[name] = y
        print(f"{name:18} single greedy pass: mean P(human)={st.mean(ph):.3f}  "
              f"semantic={st.mean(sm):.3f}")
        del model, tok
        import mlx.core as mx; mx.clear_cache()
    for name, y in samples.items():
        print(f"\n--- sample, {name} ---\n{y[:500]}")


if __name__ == "__main__":
    main()
