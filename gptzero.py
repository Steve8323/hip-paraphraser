"""GPTZero API client (the paper's actual detector).

Needs a GPTZero API key (free tier: gptzero.me -> dashboard -> API). Set it via
  export GPTZERO_API_KEY=...
or write it to ~/.config/gptzero/key

human_prob(text) -> P(human) in [0,1], matching the quantity the paper reports.
"""
import hashlib, json, os, time, urllib.request, urllib.error

ENDPOINT = "https://api.gptzero.me/v2/predict/text"
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gptzero_cache.json")


def _load_cache():
    if os.path.exists(_CACHE_PATH):
        try:
            return json.load(open(_CACHE_PATH))
        except Exception:
            return {}
    return {}


def _cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key():
    k = os.environ.get("GPTZERO_API_KEY", "").strip()
    if not k:
        p = os.path.expanduser("~/.config/gptzero/key")
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        raise RuntimeError(
            "GPTZERO_API_KEY not set. Get a free key at gptzero.me -> dashboard -> "
            "API, then: export GPTZERO_API_KEY=...  (or write ~/.config/gptzero/key)"
        )
    return k


def predict(text, use_cache=True):
    """Call GPTZero once. Results are cached on disk by text hash so the same
    text is NEVER billed twice (probe calls are reused by the batch)."""
    key = _cache_key(text)
    cache = _load_cache()
    if use_cache and key in cache:
        return cache[key]

    body = json.dumps({"document": text}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"x-api-key": _key(), "Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            cache[key] = resp
            json.dump(cache, open(_CACHE_PATH, "w"))
            return resp
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1)); continue
            raise RuntimeError(f"GPTZero error: {last}")
        except Exception as e:  # transient network
            last = str(e); time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GPTZero request failed: {last}")


def human_prob(text):
    doc = predict(text)["documents"][0]
    cp = doc.get("class_probabilities") or {}
    if "human" in cp:
        return float(cp["human"])
    if doc.get("completely_generated_prob") is not None:
        return 1.0 - float(doc["completely_generated_prob"])
    if doc.get("average_generated_prob") is not None:
        return 1.0 - float(doc["average_generated_prob"])
    raise RuntimeError("unexpected GPTZero schema: " + json.dumps(doc)[:300])


if __name__ == "__main__":
    import sys
    t = open(sys.argv[1]).read() if len(sys.argv) > 1 else (
        "I went to the store and the weather was nice so I walked the long way home.")
    print("P(human) =", human_prob(t))
