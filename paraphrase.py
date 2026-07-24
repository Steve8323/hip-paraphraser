"""Stage 3 - Iterative paraphrasing (paper Algorithm 2).

Load the BASE model + the HIP LoRA adapter and apply the paraphraser N times.
Each round formats the previous text as a <source_text>...</target_text> prompt
and lets the base model continue with the (humanized) rewrite.

CLI:
  python paraphrase.py --file input.txt --rounds 10
  echo "some AI text" | python paraphrase.py
"""
import argparse, os, sys

import config

_model = _tok = None


def _ensure():
    global _model, _tok
    if _model is None:
        from mlx_lm import load
        adapter = config.ADAPTER_PATH if os.path.isdir(config.ADAPTER_PATH) else None
        if adapter is None:
            print("[warn] no adapter found; using the raw base model", file=sys.stderr)
        _model, _tok = load(config.BASE_MODEL, adapter_path=adapter)
    return _model, _tok


def iterative(text, n=None, temperature=None, top_p=None):
    import hip_gen
    model, tok = _ensure()
    n = config.N_ROUNDS if n is None else n
    return hip_gen.iterative(model, tok, text, n, temp=temperature, top_p=top_p,
                             return_traj=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--rounds", type=int, default=config.N_ROUNDS)
    ap.add_argument("--temp", type=float, default=config.TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=config.TOP_P)
    ap.add_argument("--final", action="store_true", help="print only the final humanized text")
    args = ap.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = open(args.file, encoding="utf-8").read()
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        sys.exit("No input text.")

    traj = iterative(text, n=args.rounds, temperature=args.temp, top_p=args.top_p)
    if args.final:
        print(traj[-1])
        return
    for i, x in enumerate(traj):
        tag = "INPUT (round 0)" if i == 0 else f"ROUND {i}"
        print(f"\n===== {tag} =====\n{x}")


if __name__ == "__main__":
    main()
