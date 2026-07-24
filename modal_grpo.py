"""GRPO for the HIP paraphraser on Modal, right-sized to a single 24GB A10G
(1.7B model — no H100/L40S needed).

Pipeline in one GPU function:
  1. SFT a LoRA on the wiki AI->human pairs  -> the competent paraphraser init
     ("what previously worked").
  2. GRPO: reward = desklib P(human), GATED to 0 if the rewrite drifts
     (cosine < floor) or is degenerate. Group-normalized advantages,
     REINFORCE-with-group-baseline (on-policy). Amortizes best-of-N into one pass.
  3. Eval: single GREEDY pass P(human), SFT-init vs GRPO, on held-out inputs.

Run:  modal run modal_grpo.py
"""
import modal

app = modal.App("hip-grpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers==4.55.2", "peft==0.15.2", "accelerate==1.6.0",
        "sentence-transformers==3.4.1", "safetensors", "huggingface_hub", "numpy<2",
    )
    .add_local_dir("/Users/steven/hip_paraphraser/data", remote_path="/root/data")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("hip-grpo-adapter", create_if_missing=True)

BASE = "Qwen/Qwen3-1.7B-Base"
SRC_OPEN, SRC_CLOSE = "<source_text>", "</source_text>"
TGT_OPEN, TGT_CLOSE = "<target_text>", "</target_text>"


@app.function(gpu="A10G", image=image, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/grpo_adapter": adapter_vol})
def train_grpo(sft_epochs: int = 3, grpo_steps: int = 60, group: int = 6,
               batch: int = 4, sft_lr: float = 1e-4, grpo_lr: float = 1e-5,
               temp: float = 1.05, top_p: float = 0.97, sem_floor: float = 0.78):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import json, re, random, statistics as st
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer

    dev = "cuda"
    torch.manual_seed(0); random.seed(0)

    def build_prompt(x):
        return f"{SRC_OPEN}\n{x.strip()}\n{SRC_CLOSE}\n\n{TGT_OPEN}\n"

    _word = re.compile(r"[a-z]{3,}")

    def clean_output(raw):
        out = raw.split(TGT_CLOSE)[0].split(SRC_OPEN)[0].split(TGT_OPEN)[0]
        lines = [l.strip() for l in out.splitlines()]
        while lines and len(lines[0].split()) < 4 and not re.search(r"[.?!,]", lines[0]):
            lines.pop(0)
        return " ".join(l for l in lines if l).strip()

    def is_degenerate(t, x):
        if len(t.split()) < 5:
            return True
        nonlatin = sum(1 for c in t if ord(c) > 0x024F and not c.isspace())
        if nonlatin > 0.03 * max(1, len(t)):
            return True
        ow = set(_word.findall(x.lower()))
        return bool(ow) and len(set(_word.findall(t.lower())) & ow) < 0.2 * len(ow)

    # ---- reward models -----------------------------------------------------
    print("[modal] loading detector (desklib) + embedder ...", flush=True)
    DET = "desklib/ai-text-detector-v1.01"

    class Desklib(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = AutoModel.from_config(cfg)
            self.classifier = nn.Linear(cfg.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            out = self.model(input_ids, attention_mask=attention_mask)[0]
            m = attention_mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.classifier(pooled)

    det_tok = AutoTokenizer.from_pretrained(DET)
    det = Desklib(AutoConfig.from_pretrained(DET))
    det.load_state_dict(load_file(hf_hub_download(DET, "model.safetensors")), strict=False)
    det = det.to(dev).eval().to(torch.float32)
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)

    @torch.no_grad()
    def p_human(texts):
        enc = det_tok(texts, truncation=True, max_length=768, padding=True, return_tensors="pt").to(dev)
        logit = det(enc["input_ids"], enc["attention_mask"]).squeeze(-1)
        return (1 - torch.sigmoid(logit)).float().cpu().tolist()

    @torch.no_grad()
    def cosines(x, ys):
        e = embed.encode([x] + list(ys), convert_to_tensor=True, normalize_embeddings=True)
        return (e[1:] @ e[0]).cpu().tolist()

    # ---- policy: base + LoRA ----------------------------------------------
    print("[modal] loading base model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev)
    model.config.use_cache = True
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pad_id = tok.pad_token_id

    # ---- 1) SFT init on wiki pairs ----------------------------------------
    sft = [json.loads(l) for l in open("/root/data/train.jsonl")]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=sft_lr)
    print(f"[modal] SFT on {len(sft)} pairs, {sft_epochs} epochs", flush=True)
    model.train()
    for ep in range(sft_epochs):
        random.shuffle(sft)
        losses = []
        for r in sft:
            p_ids = tok(r["prompt"], add_special_tokens=True).input_ids
            c_ids = tok(r["completion"], add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = torch.tensor([p_ids + c_ids], device=dev)
            labels = ids.clone()
            labels[0, :len(p_ids)] = -100
            loss = model(input_ids=ids, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[modal]  SFT epoch {ep+1}: loss {st.mean(losses):.3f}", flush=True)

    # ---- eval helper: single greedy pass ----------------------------------
    eval_inputs = [json.loads(l)["a"] for l in open("/root/data/eval_pairs.jsonl")]

    @torch.no_grad()
    def eval_greedy(tag):
        model.eval()
        ph, cs, sample = [], [], None
        for x in eval_inputs:
            cap = min(320, int(len(tok(x).input_ids) * 1.6) + 64)
            pids = tok(build_prompt(x), return_tensors="pt").to(dev)
            g = model.generate(**pids, do_sample=False, max_new_tokens=cap,
                               use_cache=True, pad_token_id=pad_id)
            y = clean_output(tok.decode(g[0, pids.input_ids.shape[1]:], skip_special_tokens=True))
            ph += p_human([y]); cs += cosines(x, [y])
            if sample is None:
                sample = y
        print(f"[modal] EVAL {tag}: P(human)={st.mean(ph):.3f} semantic={st.mean(cs):.3f}", flush=True)
        return {"phuman": st.mean(ph), "semantic": st.mean(cs), "sample": sample}

    baseline = eval_greedy("SFT-init (baseline)")

    # ---- 2) GRPO ----------------------------------------------------------
    prompts = [json.loads(l)["a"] for l in open("/root/data/pairs.jsonl")] + eval_inputs
    for g in opt.param_groups:
        g["lr"] = grpo_lr
    print(f"[modal] GRPO: {len(prompts)} prompts, {grpo_steps} steps, G={group} B={batch}", flush=True)
    history = []
    for step in range(grpo_steps):
        xs = random.sample(prompts, min(batch, len(prompts)))
        seqs, lens, advs, step_r = [], [], [], []
        for x in xs:
            cap = min(320, int(len(tok(x).input_ids) * 1.6) + 64)
            pids = tok(build_prompt(x), return_tensors="pt").to(dev)
            P = pids.input_ids.shape[1]
            model.eval()
            with torch.no_grad():
                gen = model.generate(**pids, do_sample=True, temperature=temp, top_p=top_p,
                                     max_new_tokens=cap, num_return_sequences=group,
                                     use_cache=True, pad_token_id=pad_id)
            ys = [clean_output(tok.decode(gen[i, P:], skip_special_tokens=True)) for i in range(group)]
            phs = p_human(ys)
            css = cosines(x, ys)
            rewards = [0.0 if (is_degenerate(y, x) or c < sem_floor) else ph
                       for y, ph, c in zip(ys, phs, css)]
            step_r += rewards
            m, sd = st.mean(rewards), st.pstdev(rewards)
            if sd < 1e-4:
                continue
            p_ids = tok(build_prompt(x), add_special_tokens=True).input_ids
            for y, r in zip(ys, rewards):
                c_ids = tok(y, add_special_tokens=False).input_ids
                if not c_ids:
                    continue
                seqs.append(p_ids + c_ids)
                lens.append((len(p_ids), len(p_ids) + len(c_ids)))
                advs.append((r - m) / (sd + 1e-4))

        if not seqs:
            history.append(st.mean(step_r) if step_r else 0.0)
            print(f"[modal] step {step+1}: no signal (reward {history[-1]:.3f})", flush=True)
            continue

        model.train()
        opt.zero_grad()
        nseq = len(seqs)
        CHUNK = 2                                     # micro-batch: bound peak memory
        for i0 in range(0, nseq, CHUNK):
            js = list(range(i0, min(i0 + CHUNK, nseq)))
            ss = [seqs[j] for j in js]
            W = max(len(s) for s in ss)
            ids = torch.full((len(ss), W), pad_id, dtype=torch.long, device=dev)
            for k, s in enumerate(ss):
                ids[k, :len(s)] = torch.tensor(s, device=dev)
            attn = (ids != pad_id).long()
            Pl = torch.tensor([lens[j][0] for j in js], device=dev)
            Tl = torch.tensor([lens[j][1] for j in js], device=dev)
            adv = torch.tensor([advs[j] for j in js], device=dev, dtype=torch.float32)

            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, :-1, :]
            targets = ids[:, 1:]
            V = logits.size(-1)
            ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                 reduction="none").view(targets.shape)   # fused, no big log_softmax
            logp = -ce
            steps = torch.arange(1, W, device=dev).unsqueeze(0)
            mask = ((steps >= Pl.unsqueeze(1)) & (steps <= (Tl.unsqueeze(1) - 1))).float()
            seq_logp = (logp * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = -(adv * seq_logp).mean() * (len(js) / nseq)           # scale for grad accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        mr = st.mean(step_r)
        history.append(mr)
        print(f"[modal] step {step+1}/{grpo_steps}: reward {mr:.3f} loss {loss.item():.4f} "
              f"(kept {len(seqs)})", flush=True)

    grpo = eval_greedy("GRPO")

    # persist the trained adapter so the run survives a closed laptop
    model.save_pretrained("/root/grpo_adapter")
    json.dump({"baseline": baseline, "grpo": grpo, "reward_history": history},
              open("/root/grpo_adapter/results.json", "w"), indent=2)
    adapter_vol.commit()
    print("[modal] adapter + results saved to volume 'hip-grpo-adapter'", flush=True)

    return {"baseline": baseline, "grpo": grpo, "reward_history": history,
            "gpu": "A10G", "grpo_steps": grpo_steps}


@app.local_entrypoint()
def main(grpo_steps: int = 80):
    # .spawn() = fire-and-forget: the job runs in Modal's cloud independent of
    # this laptop (survives network drops / closing the lid). Results land in the
    # 'hip-grpo-adapter' volume (results.json) when it finishes.
    call = train_grpo.spawn(grpo_steps=grpo_steps)
    print(f"SPAWNED function call id: {call.object_id}")
    print("Runs in the cloud independently. When done, fetch results with:")
    print("  modal volume get hip-grpo-adapter results.json -")
