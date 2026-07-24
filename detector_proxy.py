"""Local human-likeness proxy detector.

We can't query GPTZero / Pangram (paid, closed).  As a stand-in we use
desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, a RAID-benchmark leader that
empirically tracks GPTZero on humanized text).  Reports human_prob = 1 - P(AI).

Adapted from ~/ai_humanizer/paraphraser/detector.py (handles a transformers-5.x
weight-tying quirk).  Degrades gracefully to None if torch/model unavailable.
"""
_MODEL = "desklib/ai-text-detector-v1.01"
_tok = _mdl = None
_failed = False


def _load():
    global _tok, _mdl, _failed
    if _mdl is not None or _failed:
        return _tok, _mdl
    try:
        import torch, torch.nn as nn
        from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
        from huggingface_hub import hf_hub_download

        class _Desklib(PreTrainedModel):
            config_class = AutoConfig
            _tied_weights_keys = []

            def __init__(self, cfg):
                super().__init__(cfg)
                self.model = AutoModel.from_config(cfg)
                self.classifier = nn.Linear(cfg.hidden_size, 1)

            def forward(self, input_ids, attention_mask=None):
                out = self.model(input_ids, attention_mask=attention_mask)[0]
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(pooled)

        try:
            _tok = AutoTokenizer.from_pretrained(_MODEL)
        except Exception:
            _tok = AutoTokenizer.from_pretrained(_MODEL, use_fast=False)
        _mdl = _Desklib(AutoConfig.from_pretrained(_MODEL))
        from safetensors.torch import load_file
        _mdl.load_state_dict(load_file(hf_hub_download(_MODEL, "model.safetensors")), strict=False)
        _mdl.eval()
        torch.set_grad_enabled(False)
    except Exception as e:  # missing torch, no network, etc.
        print(f"[detector] unavailable ({type(e).__name__}: {e}); skipping detector scores")
        _failed = True
        _tok = _mdl = None
    return _tok, _mdl


def human_prob(text, max_len=768):
    """P(human) in [0,1] per the proxy detector, or None if unavailable."""
    import torch
    tok, mdl = _load()
    if mdl is None:
        return None
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
    logit = float(mdl(enc["input_ids"], enc["attention_mask"]).item())
    return 1.0 - float(torch.sigmoid(torch.tensor(logit)).item())


if __name__ == "__main__":
    import sys
    t = open(sys.argv[1]).read() if len(sys.argv) > 1 else "This is a test sentence."
    print("P(human) =", human_prob(t))
