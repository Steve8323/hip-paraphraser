"""Central config for HIP (Humanization by Iterative Paraphrasing).

Method: Xu, Zhong, Raghunathan, Fang, Kolter, "Base Models Look Human To AI
Detectors" (arXiv:2605.19516).  A *base* LLM is minimally LoRA-tuned to map an
AI-style passage back to its human original, using a plain-text continuation
format (NOT a chat template) so the base model's native human-looking
continuation behavior is preserved.  Applied iteratively, its output drifts
toward what detectors judge as human.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- model -----------------------------------------------------------------
# A BASE model is the whole point: instruct-tuning is what detectors flag.
BASE_MODEL   = os.environ.get("HIP_BASE_MODEL", "Qwen/Qwen3-1.7B-Base")
ADAPTER_PATH = os.environ.get("HIP_ADAPTER_PATH", os.path.join(HERE, "adapters"))

# --- paths -----------------------------------------------------------------
DATA_DIR    = os.environ.get("HIP_DATA_DIR", os.path.join(HERE, "data"))
PAIRS_PATH  = os.path.join(DATA_DIR, "pairs.jsonl")        # (a, h) training pairs
EVAL_PATH   = os.path.join(DATA_DIR, "eval_pairs.jsonl")   # held-out AI inputs + human refs

# --- HIP continuation format (paper §3.2) ----------------------------------
# The prompt prefix is everything up to and including the <target_text> open
# tag; the model completes the human passage and the closing tag.  Loss is on
# the completion span only (mlx_lm --mask-prompt).
SRC_OPEN, SRC_CLOSE = "<source_text>", "</source_text>"
TGT_OPEN, TGT_CLOSE = "<target_text>", "</target_text>"
STOP = TGT_CLOSE

def build_prompt(source_text: str) -> str:
    return f"{SRC_OPEN}\n{source_text.strip()}\n{SRC_CLOSE}\n\n{TGT_OPEN}\n"

def build_completion(target_text: str) -> str:
    return f"{target_text.strip()}\n{TGT_CLOSE}"

# --- generation (paper: temperature 1.0, top-p 0.95, N=10 rounds) ----------
TEMPERATURE = float(os.environ.get("HIP_TEMP", "1.0"))
TOP_P       = float(os.environ.get("HIP_TOP_P", "0.95"))
N_ROUNDS    = int(os.environ.get("HIP_ROUNDS", "10"))

# --- Anthropic (AI-side pair synthesis + semantic judge) -------------------
# Paper uses GPT-5-nano for both; we substitute Claude Haiku.
ANTHROPIC_MODEL = os.environ.get("HIP_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
