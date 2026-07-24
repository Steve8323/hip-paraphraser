"""One-pass, detector-guided humanizer.

A SINGLE paraphrase round: sample k candidates and return the one the local
desklib proxy rates most human (while keeping the meaning). Gets text to
'human' in the first round instead of iterating.

  echo "AI text" | python humanize_best.py --k 12
  python humanize_best.py --file draft.txt
  python humanize_best.py --src-file sources_large.txt    # one block per paragraph
"""
import argparse, os, sys

import config
import hip_gen
import detector_proxy

_model = _tok = None
_sem = None


def _load():
    global _model, _tok
    if _model is None:
        from mlx_lm import load
        adapter = config.ADAPTER_PATH if os.path.isdir(config.ADAPTER_PATH) else None
        _model, _tok = load(config.BASE_MODEL, adapter_path=adapter)
    return _model, _tok


def _sem_scorer():
    """Embedding-cosine meaning check (keeps best-of-N from picking degraded text)."""
    global _sem
    if _sem is None:
        import test_quality
        _sem, _ = test_quality._semantic_fn()
    return _sem


def humanize(text, k=12, temp=1.1, top_p=0.97, target=0.9, max_k=40, min_sem=0.85):
    model, tok = _load()
    best, info = hip_gen.best_of_round(model, tok, text, k=k, temp=temp, top_p=top_p,
                                       scorer=detector_proxy.human_prob,
                                       target=target, max_k=max_k,
                                       sem_scorer=_sem_scorer(), min_sem=min_sem)
    return best, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--src-file", help="multiple inputs, separated by blank lines")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--target", type=float, default=0.9,
                    help="keep sampling (up to --max-k) until P(human) >= target")
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--min-sem", type=float, default=0.85,
                    help="reject candidates below this embedding cosine vs the original")
    ap.add_argument("--temp", type=float, default=1.1)
    ap.add_argument("--top-p", type=float, default=0.97)
    ap.add_argument("--quiet", action="store_true", help="print only the humanized text")
    args = ap.parse_args()

    if args.src_file:
        raw = open(args.src_file, encoding="utf-8").read()
        inputs = [c.strip() for c in raw.split("\n\n") if c.strip()]
    elif args.text:
        inputs = [args.text]
    elif args.file:
        inputs = [open(args.file, encoding="utf-8").read().strip()]
    else:
        inputs = [sys.stdin.read().strip()]

    for i, text in enumerate(inputs):
        if not text:
            continue
        best, info = humanize(text, args.k, args.temp, args.top_p, args.target,
                              args.max_k, args.min_sem)
        if args.quiet:
            print(best)
            continue
        p_in = detector_proxy.human_prob(text)
        sem = info.get("semantic")
        print(f"\n===== input {i+1} =====")
        print(f"P(human): in {p_in:.3f}  ->  out {info['phuman']:.3f}  "
              f"| semantic {sem:.3f}" if sem is not None else
              f"P(human): in {p_in:.3f}  ->  out {info['phuman']:.3f}")
        print(f"(1 round, {info['kept']} faithful of {info['drawn']} drawn)")
        print(best)


if __name__ == "__main__":
    main()
