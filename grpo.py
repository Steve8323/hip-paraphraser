"""Minimal GRPO for the HIP paraphraser (MLX, local).

Start from the competent wiki adapter and reward-optimize a SINGLE pass so it
produces human-scoring, faithful paraphrases without best-of-N at inference.

Reward(x, y) = desklib P(human)(y), GATED to 0 if y drifts (cosine(x,y) < floor)
or is degenerate. Advantages are group-normalized over G samples per prompt
(GRPO baseline); update is REINFORCE-with-group-baseline (on-policy, ratio=1).

  python grpo.py --steps 30 --group 6 --batch 4          # real run
  python grpo.py --steps 2 --group 4 --batch 2 --smoke   # quick check
"""
import argparse, json, os, shutil, statistics as st

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

import config
import hip_gen
import detector_proxy
import test_quality

INIT_ADAPTER = "adapters"            # the wiki adapter = "what previously worked"
OUT_ADAPTER = "adapters_grpo"
LORA_CFG = {"rank": 32, "scale": 16.0, "dropout": 0.0}
NUM_LAYERS = 16


def build_policy():
    model, tok = load(config.BASE_MODEL)
    model.freeze()
    linear_to_lora_layers(model, NUM_LAYERS, LORA_CFG)
    w = os.path.join(INIT_ADAPTER, "adapters.safetensors")
    if os.path.exists(w):
        model.load_weights(w, strict=False)
        print(f"[grpo] initialized from {w}")
    model.train()
    return model, tok


def load_prompts():
    # "what previously worked" = the wiki-style AI inputs, where the detector has
    # reward head-room (best-of-N saw 0.03 -> 0.94). Essay prompts are saturated
    # at 0 and give no GRPO signal, so we exclude them.
    xs = []
    for p in ("data/pairs.jsonl", "data/eval_pairs.jsonl"):
        if os.path.exists(p):
            xs += [json.loads(l)["a"] for l in open(p)]
    return xs


def grpo_loss(model, batch, lengths, adv):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs).astype(mx.float32)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
    logp = -nn.losses.cross_entropy(logits, targets)          # [B, T-1] per-token logprob
    seq_logp = (logp * mask).sum(1) / mx.maximum(mask.sum(1), 1)
    return -(adv * seq_logp).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temp", type=float, default=1.05)
    ap.add_argument("--top-p", type=float, default=0.97)
    ap.add_argument("--sem-floor", type=float, default=0.78)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    import random
    rng = random.Random(args.seed)

    model, tok = build_policy()
    sim, _ = test_quality._semantic_fn()
    opt = optim.Adam(learning_rate=args.lr)
    prompts = load_prompts()
    print(f"[grpo] {len(prompts)} training prompts; G={args.group} B={args.batch} steps={args.steps}")
    pad_id = tok.eos_token_id or 0
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    history = []
    for step in range(args.steps):
        xs = rng.sample(prompts, min(args.batch, len(prompts)))
        seqs, lens, advs = [], [], []
        step_rewards = []
        for x in xs:
            cap = min(400, int(len(tok.encode(x)) * 1.6) + 64)
            prompt = config.build_prompt(x)
            comps, rewards = [], []
            for _ in range(args.group):
                raw = generate(model, tok, prompt, max_tokens=cap, sampler=sampler, verbose=False)
                y = hip_gen.clean_output(raw)
                if hip_gen.is_degenerate(y, x) or sim(x, y) < args.sem_floor:
                    r = 0.0
                else:
                    r = detector_proxy.human_prob(y)
                comps.append(y)
                rewards.append(float(r))
            step_rewards += rewards
            mean_r = st.mean(rewards)
            std_r = st.pstdev(rewards)
            if std_r < 1e-4:
                continue                                       # no intra-group signal
            for y, r in zip(comps, rewards):
                full = tok.encode(prompt + y)
                plen = len(tok.encode(prompt))
                if len(full) <= plen:
                    continue
                seqs.append(full)
                lens.append((plen, len(full)))
                advs.append((r - mean_r) / (std_r + 1e-4))

        if not seqs:
            print(f"[grpo] step {step+1}: no signal (mean reward {st.mean(step_rewards):.3f})")
            continue

        W = max(len(s) for s in seqs)
        batch = mx.array([s + [pad_id] * (W - len(s)) for s in seqs])
        lengths = mx.array(lens)
        adv = mx.array(advs)

        lvag = nn.value_and_grad(model, grpo_loss)
        loss, grads = lvag(model, batch, lengths, adv)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

        mr = st.mean(step_rewards)
        history.append(mr)
        print(f"[grpo] step {step+1}/{args.steps}: mean reward {mr:.3f}  "
              f"loss {float(loss):.4f}  (kept {len(seqs)} seqs)")

    # save adapter
    os.makedirs(OUT_ADAPTER, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(os.path.join(OUT_ADAPTER, "adapters.safetensors"), weights)
    if os.path.exists(os.path.join(INIT_ADAPTER, "adapter_config.json")):
        shutil.copy(os.path.join(INIT_ADAPTER, "adapter_config.json"),
                    os.path.join(OUT_ADAPTER, "adapter_config.json"))
    json.dump({"history": history}, open(os.path.join(OUT_ADAPTER, "grpo_history.json"), "w"))
    print(f"[grpo] saved -> {OUT_ADAPTER}  (reward {history[0] if history else 0:.3f} -> "
          f"{history[-1] if history else 0:.3f})")


if __name__ == "__main__":
    main()
