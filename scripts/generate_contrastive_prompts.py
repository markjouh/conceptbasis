"""Generate contrastive prompt sets for each dictionary concept:
  positive: short descriptions of an object WITH the property
  negative: short descriptions of an object WITHOUT it / with the opposite

Direction = mean(text emb of positives) - mean(negatives). Replaces the
'an object that is X' - 'an object' construction, which is below chance for
semantic/world-knowledge concepts (e.g. 'manmade': AUROC 0.41 vs 0.97 with a
real contrastive negative) — such axes need an opposing prompt set, not a
neutral base.

Local qwen (LM Studio), text-only. Output: data/contrastive_prompts.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_URL = os.environ.get("VLM_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "qwen/qwen3.6-35b-a3b")
N = 8

SYSTEM = "You write short CLIP prompts describing objects. Output strict JSON only."


def build_prompt(c: dict) -> str:
    mem = ", ".join(c["members"][:6])
    neg = f" Known opposite terms: {', '.join(c['negative_pole'])}." if c.get("negative_pole") else ""
    return (
        f"Property: '{c['name']}' (also phrased as: {mem}).{neg}\n"
        f"Write {N} POSITIVE prompts: short phrases (2-6 words) describing a generic object "
        f"that clearly HAS this property, e.g. for 'manmade': \"a manmade object\", "
        f"\"an artificial manufactured item\".\n"
        f"Write {N} NEGATIVE prompts: same style but an object that clearly LACKS it or has "
        f"the OPPOSITE, e.g. for 'manmade': \"a natural object\", \"something from nature\".\n"
        f"Vary wording. Do not name specific object types.\n"
        f'Return ONLY JSON: {{"positive": [...], "negative": [...]}}'
    )


def _arr(t: str, key: str) -> list[str]:
    m = t.find(f'"{key}"')
    lb = t.find("[", m) if m >= 0 else -1
    if lb < 0:
        return []
    seg = t[lb:t.find("]", lb) + 1]
    return [s for s in re.findall(r'"([^"]{3,60})"', seg)]


def gen_one(c: dict, retries: int = 5) -> dict:
    payload_base = {
        "model": MODEL, "max_tokens": 600, "reasoning_effort": "none",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_prompt(c)}],
    }
    for k in range(retries):
        try:
            r = requests.post(API_URL, json={**payload_base, "temperature": 0.4 if k == 0 else 0.8},
                              timeout=120)
            r.raise_for_status()
            t = re.sub(r"```(?:json)?", "", r.json()["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(1.0 + k)
            continue
        pos, neg = _arr(t, "positive")[:N], _arr(t, "negative")[:N]
        if len(pos) >= 4 and len(neg) >= 4:
            return {"positive": pos, "negative": neg}
    return {"positive": [], "negative": []}


def main():
    d = json.load(open(os.path.join(ROOT, "data", "dictionary.json")))
    path = os.path.join(ROOT, "data", "contrastive_prompts.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    todo = [c for c in d if len(out.get(c["name"], {}).get("positive", [])) < 4]
    print(f"{len(todo)} concepts to generate")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(gen_one, c): c["name"] for c in todo}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            r = fut.result()
            with lock:
                if r["positive"]:
                    out[futs[fut]] = r
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    thin = [c["name"] for c in d if len(out.get(c["name"], {}).get("positive", [])) < 4]
    print(f"wrote {path} | thin/missing: {thin or 'none'}")
    print("\nsample (manmade):", json.dumps(out.get("manmade", {}), indent=1)[:400])


if __name__ == "__main__":
    main()
