"""Deterministically de-tokenize wikitext surface artifacts in the pairs so the
model learns to emit clean prose (and the detector eval isn't gamed by the
corpus's pre-tokenized ' , ' / '@-@' style).  Applies to both a and h.
"""
import json, re

import config


def normalize(t: str) -> str:
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r"\s+([,.;:!?%)\]])", r"\1", t)      # no space before closing punct
    t = re.sub(r"([(\[])\s+", r"\1", t)              # no space after opening bracket
    t = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", t)
    t = t.replace(" 's", "'s").replace("`` ", '"').replace(" ''", '"')
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    for path in (config.PAIRS_PATH, config.EVAL_PATH):
        rows = [json.loads(l) for l in open(path)]
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({"a": normalize(r["a"]), "h": normalize(r["h"])}) + "\n")
        print(f"normalized {len(rows)} rows in {path}")


if __name__ == "__main__":
    main()
