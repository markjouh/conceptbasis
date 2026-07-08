"""VLM-verified concept anchors: Qwen judges candidate pos/neg images per
concept (yes/no), verified sets become image-anchored mu+ - mu- directions.

Candidates come from the current soft-label ranking (top band = proposed
positives, bottom band = proposed negatives) over the CC0-mining + train split.
Resumable JSONL: data/concept_judgments.jsonl

  python scripts/verify_concepts.py --per-side 24 --workers 8
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")
API_URL = os.environ.get("VLM_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "qwen/qwen3.6-35b-a3b")
MAX_SIDE = 512

SYSTEM = "You judge visual properties of objects in photos. Answer with exactly one word: yes or no."


def encode_image(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def judge(path: str, concept: str, members: list[str], retries: int = 4) -> bool | None:
    gloss = f" (i.e. {', '.join(members[:3])})" if len(members) > 1 else ""
    q = (f"Look at the main object in this photo. Is it '{concept}'{gloss}? "
         f"Judge the object itself. Answer exactly one word: yes or no.")
    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": 5, "reasoning_effort": "none",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": [
                         {"type": "text", "text": q},
                         {"type": "image_url", "image_url": {"url": encode_image(path)}}]}],
    }
    for k in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            a = r.json()["choices"][0]["message"]["content"].strip().lower()
        except Exception:
            time.sleep(1 + k)
            continue
        if re.match(r"^yes", a):
            return True
        if re.match(r"^no", a):
            return False
    return None


def build_tasks(per_side: int, seed: int = 0):
    d = json.load(open(os.path.join(ROOT, "data/dictionary.json")))
    ids = json.load(open(os.path.join(ROOT, "data/image_ids.json")))
    df = pd.read_parquet(os.path.join(ROOT, "data/labels.parquet"))
    tr = (df.split == "train").to_numpy()
    scols = [c for c in df.columns if c.startswith("s_")]
    S = df[scols].to_numpy(dtype=np.float32)
    tr_idx = np.where(tr)[0]
    rng = np.random.default_rng(seed)
    tasks = []
    for k, c in enumerate(d):
        order = tr_idx[np.argsort(-S[tr_idx, k])]
        top = order[:200]                      # sample within bands (diversity)
        bot = order[-200:]
        pos = rng.choice(top, size=min(per_side, len(top)), replace=False)
        neg = rng.choice(bot, size=min(per_side, len(bot)), replace=False)
        for i in pos:
            tasks.append({"concept": c["name"], "members": c["members"],
                          "image_id": ids[i], "proposed": "pos"})
        for i in neg:
            tasks.append({"concept": c["name"], "members": c["members"],
                          "image_id": ids[i], "proposed": "neg"})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-side", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/concept_judgments.jsonl")
    ap.add_argument("--only-concepts", nargs="*", default=None)
    args = ap.parse_args()

    tasks = build_tasks(args.per_side)
    if args.only_concepts:
        tasks = [t for t in tasks if t["concept"] in set(args.only_concepts)]
    out_path = os.path.join(ROOT, args.out)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                if r.get("verdict") is not None:
                    done.add((r["concept"], r["image_id"]))
    todo = [t for t in tasks if (t["concept"], t["image_id"]) not in done]
    print(f"{len(tasks)} judgments planned, {len(done)} done, {len(todo)} to run")

    lock = threading.Lock()
    def work(t):
        v = judge(os.path.join(IMG_DIR, t["image_id"]), t["concept"], t["members"])
        return {**{k: t[k] for k in ("concept", "image_id", "proposed")}, "verdict": v}

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush()

    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    ok = [r for r in rows if r["verdict"] is not None]
    agree = np.mean([(r["verdict"] is True) == (r["proposed"] == "pos") for r in ok])
    print(f"done. {len(ok)} verdicts | CLIP-proposal agreement: {agree:.2f} "
          f"(disagreements are the interesting set)")


if __name__ == "__main__":
    main()
