"""Robust generation helpers shared by paraphrase.py and examples_run.py.

Iterative paraphrasing on small base models can derail two ways:
  (a) non-termination -> the model never emits </target_text> and repeats /
      emits script garbage until the token cap;
  (b) semantic diffusion -> after several rounds the text drifts off-topic.
These helpers cap tokens by the ORIGINAL length (constant across rounds),
truncate repeats, and stop+return the last good round on degeneration.
"""
import re

import config
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

_WORD = re.compile(r"[a-z]{3,}")


def _content_words(t):
    return set(_WORD.findall(t.lower()))


def _truncate_repeats(text):
    """Cut at the first repeated sentence (kills runaway loops)."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    seen, out = set(), []
    for s in sents:
        k = s.strip().lower()
        if len(k) > 15 and k in seen:
            break
        if k:
            seen.add(k)
        out.append(s)
    return " ".join(out).strip()


def clean_output(raw):
    out = raw.split(config.STOP)[0].split(config.SRC_OPEN)[0].split(config.TGT_OPEN)[0]
    # drop leading fragment lines (e.g. "North\nSouth") then collapse to one paragraph
    lines = [l.strip() for l in out.splitlines()]
    while lines and len(lines[0].split()) < 4 and not re.search(r"[.?!,]", lines[0]):
        lines.pop(0)
    out = " ".join(l for l in lines if l)
    return _truncate_repeats(out).strip()


def is_degenerate(text, original):
    if len(text.split()) < 5:
        return True
    nonlatin = sum(1 for c in text if ord(c) > 0x024F and not c.isspace())
    if nonlatin > 0.03 * max(1, len(text)):           # foreign-script garbage
        return True
    ow = _content_words(original)
    if ow and len(_content_words(text) & ow) < 0.2 * len(ow):  # drifted off-topic
        return True
    return False


def paraphrase_once(model, tok, text, cap, temp=None, top_p=None):
    out = generate(
        model, tok, config.build_prompt(text), max_tokens=cap,
        sampler=make_sampler(
            temp=config.TEMPERATURE if temp is None else temp,
            top_p=config.TOP_P if top_p is None else top_p),
        verbose=False,
    )
    return clean_output(out)


def _ok(cand, original):
    return (not is_degenerate(cand, original)
            and len(cand.split()) <= 1.8 * len(original.split()))


def best_of_round(model, tok, text, k=12, temp=1.1, top_p=0.97, scorer=None,
                  target=None, max_k=40, sem_scorer=None, min_sem=0.0):
    """ONE paraphrase round, detector-guided. Sample candidates, discard any that
    fall below the meaning floor (sem_scorer vs original < min_sem), and keep the
    most-human survivor. If `target` is set, keep drawing (up to max_k) until
    P(human) >= target so a single round reaches 'no AI'; else draw k and return
    the best. The semantic floor stops the detector from rewarding degraded text."""
    text = text.strip()
    cap = min(768, int(len(tok.encode(text)) * 1.6) + 64)
    limit = max_k if target is not None else k
    best, best_p, best_sem, kept, drawn = None, -1.0, None, 0, 0
    while drawn < limit:
        c = paraphrase_once(model, tok, text, cap, temp, top_p)
        drawn += 1
        if not _ok(c, text):
            continue
        s = sem_scorer(text, c) if sem_scorer else None
        if s is not None and s < min_sem:               # too much meaning lost
            continue
        kept += 1
        p = scorer(c) if scorer else 1.0
        if p > best_p:
            best_p, best, best_sem = p, c, s
        if scorer is None or (target is not None and best_p >= target):
            break
    if best is None:                                    # no faithful candidate
        return text, {"phuman": scorer(text) if scorer else None,
                      "semantic": None, "drawn": drawn, "kept": 0}
    return best, {"phuman": best_p if scorer else None, "semantic": best_sem,
                  "drawn": drawn, "kept": kept}


def iterative(model, tok, text, n, temp=None, top_p=None, return_traj=False):
    text = text.strip()
    cap = min(768, int(len(tok.encode(text)) * 1.6) + 64)   # fixed, from ORIGINAL
    x, last_good, traj = text, text, [text]
    for _ in range(n):
        nx = None
        for _try in range(3):                                # resample a derailed round
            cand = paraphrase_once(model, tok, x, cap, temp, top_p)
            if _ok(cand, text):
                nx = cand
                break
        if nx is None:                                       # couldn't get a clean round
            break
        last_good = nx
        traj.append(nx)
        x = nx
    return traj if return_traj else last_good
