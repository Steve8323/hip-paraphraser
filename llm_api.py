"""Anthropic (Claude Haiku) backend for HIP.

Two roles, both played by GPT-5-nano in the paper, by Claude Haiku here:
  - ai_paraphrase(h)        : human passage -> AI-style rewrite a   (builds pairs)
  - semantic_score(x0, x)   : 0..10 meaning-preservation judge      (evaluation)

Reads ANTHROPIC_API_KEY from the environment (or ~/.config/anthropic/key).
"""
import os, re, time, functools

_REFUSAL = re.compile(r"\b(I can't|I cannot|I'm unable|I will not|as an AI)\b", re.I)
_PREFACE = re.compile(
    r"^\s*(here(?:'s| is)[^:\n]*:|sure[,!]?|certainly[,!]?|rewrite:|paraphrase:)\s*",
    re.I,
)


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        path = os.path.expanduser("~/.config/anthropic/key")
        if os.path.exists(path):
            key = open(path).read().strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.  Run:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "or write the key to ~/.config/anthropic/key"
        )
    return key


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic(api_key=_api_key())


def _complete(system: str, user: str, max_tokens: int = 1024, temperature: float = 1.0) -> str:
    from config import ANTHROPIC_MODEL
    last = None
    for attempt in range(5):
        try:
            msg = _client().messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        except Exception as e:  # rate limit / transient
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Anthropic call failed after retries: {last}")


# --- AI-side pair synthesis -------------------------------------------------
_AI_SYSTEM = (
    "You are a careful editor. Rewrite the user's passage in clear, polished, "
    "well-structured prose. Preserve every fact and the full meaning, and keep "
    "the length close to the original. Do not add commentary, headings, or "
    "preamble. Output ONLY the rewritten passage."
)


def ai_paraphrase(human_text: str, temperature: float = 1.0) -> str:
    """Human passage h -> AI-style rewrite a (the 'machine-like' source side)."""
    out = _complete(_AI_SYSTEM, human_text, max_tokens=1024, temperature=temperature)
    out = _PREFACE.sub("", out).strip().strip('"').strip()
    return out


def anomaly_free(a: str, h: str) -> bool:
    """Cheap structural screen (paper's AnomalyFree): non-empty, not a refusal,
    length within a sane band of the source."""
    if not a or len(a.split()) < 8:
        return False
    if _REFUSAL.search(a):
        return False
    r = len(a.split()) / max(1, len(h.split()))
    return 0.5 <= r <= 2.0


# --- semantic judge ---------------------------------------------------------
_SEM_SYSTEM = (
    "You compare two passages and rate how completely the SECOND preserves the "
    "meaning of the FIRST, on an integer scale from 0 to 10. 10 = all key "
    "information and claims preserved with no contradictions; 0 = unrelated or "
    "meaning destroyed. Reply with ONLY the integer."
)


def semantic_score(original: str, rewrite: str) -> int:
    user = f"FIRST (original):\n{original}\n\nSECOND (rewrite):\n{rewrite}"
    out = _complete(_SEM_SYSTEM, user, max_tokens=8, temperature=0.0)
    m = re.search(r"\d+", out)
    if not m:
        return 0
    return max(0, min(10, int(m.group())))


if __name__ == "__main__":
    # tiny self-check (needs a key)
    h = "The library closes at nine on weekdays and is shut on public holidays."
    a = ai_paraphrase(h)
    print("AI rewrite:", a)
    print("semantic  :", semantic_score(h, a))
