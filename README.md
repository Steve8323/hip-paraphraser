# HIP — an AI paraphraser on a base LLM

Implements **Humanization by Iterative Paraphrasing (HIP)** from
Xu, Zhong, Raghunathan, Fang, Kolter, *"Base Models Look Human To AI Detectors"*
(arXiv:2605.19516), on a **base** open-source small LLM (`Qwen/Qwen3-1.7B-Base`).

## The idea (why a *base* model)

Detectors like GPTZero/Pangram largely flag artifacts of **instruction tuning**,
not some invariant "machine-ness." The paper's finding: a base model's
continuations of human text are judged *overwhelmingly human*. HIP exploits this:

1. **Data** — take human passages `h`; have an AI rewrite each into AI-style text
   `a`. Pairs `(a, h)`. (Paper: GPT-5-nano. Here: Claude Haiku.)
2. **Minimal fine-tune** — LoRA-tune the **base** model to map `a → h` using a
   plain continuation format (no chat template), loss on the human span only.
   This teaches the *paraphrase* behavior while preserving the base model's
   native human-looking continuation style ("low distortion").

   ```
   <source_text>
   {a}
   </source_text>

   <target_text>
   {h}            <- the model learns to produce this
   </target_text>
   ```
3. **Iterate** — apply the paraphraser N≈10 times; each round rewrites the last
   output, progressively replacing AI-origin context with human-looking context.

## Files

| file | role |
|---|---|
| `config.py` | model id, the source/target format, generation + round params |
| `llm_api.py` | Claude Haiku: AI-side pair synthesis + 0–10 semantic judge |
| `data_prep.py` | Alg. 1 — human corpus → filter/dedup → `(a,h)` with rejection sampling |
| `make_data.py` | pairs → mlx_lm `completions` jsonl (prompt masked in loss) |
| `train.py` | Alg. 2 setup — wraps `mlx_lm.lora` (`--mask-prompt`) → `adapters/` |
| `paraphrase.py` | Stage 3 — iterative paraphrasing, base model + adapter |
| `detector_proxy.py` | local desklib P(human) proxy (GPTZero/Pangram stand-in) |
| `evaluate.py` | run on held-out inputs → JSON + figure + HTML report |
| `run.sh` | full pipeline |

## Run

### A) On your Claude subscription — no API key (the path that was used)

The LLM steps (writing the AI-side pairs, judging meaning) are done by Claude
Code itself instead of a paid API call. The scripts do everything non-LLM; the
agent does the rewriting in between.

```bash
cd ~/hip_paraphraser
python3 prepare_passages.py --n 140        # pick human passages (no key)
#   -> agent/subagents rewrite each into AI-style text, writing
#      data/pairs_part_*.jsonl   (one batch per subagent)
python3 assemble_pairs.py --n-eval 16      # parts -> pairs.jsonl + eval_pairs.jsonl
python3 normalize_pairs.py                 # de-tokenize wikitext surface artifacts
python3 make_data.py
python3 train.py --epochs 3                # LoRA -> adapters/
python3 evaluate.py --n-inputs 12 --semantic none   # trajectories + detector
#   -> agent fills semantic scores into eval_results.json
python3 evaluate.py --render-only          # rebuild + open the HTML report
```

### B) With an Anthropic API key (fully scripted)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd ~/hip_paraphraser
./run.sh                       # data → train → evaluate (opens an HTML report)
# fewer/more pairs:  ./run.sh --n 200 --n-eval 16
```

Use the trained paraphraser on your own text:

```bash
# One-pass, detector-guided (default). Draws candidates and keeps the most-human
# one above the meaning floor, escalating until P(human) >= target — in ONE round.
echo "Your AI-sounding paragraph." | ./humanize.sh
./humanize.sh --target 0.95 --max-k 60 --min-sem 0.85 --file draft.txt

# Raw iterative paraphrasing (the paper's N-round loop), no detector in the loop:
python3 paraphrase.py --file draft.txt --rounds 10
```

The one-pass humanizer (`humanize_best.py`) is best-of-N with the local desklib
P(human) as selector and an embedding-cosine **semantic floor** so it can't win by
degrading the text. On movable inputs it takes 0.03 -> ~0.90 P(human) in a single
round with ~0.99 semantic; on essay-style prose desklib is saturated and won't move
(see caveats).

## Notes / honest caveats

- **Detector is a proxy.** The paper evaluates on the commercial GPTZero/Pangram
  APIs; we use a local desklib DeBERTa detector as a stand-in. Trends should
  track, absolute numbers won't match.
- Everything runs locally on Apple Silicon via MLX; only the AI-side synthesis
  and the semantic judge hit the Anthropic API.
- This is a **detector-evasion / humanization** research reproduction. Use it to
  study how detectors fail, not to misrepresent authorship.
