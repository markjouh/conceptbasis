"""Caption THINGS training images with the local VLM (LM Studio).

One grounded ~15-30 word sentence per image; the concept label (folder name)
is passed as the subject. Instance-level detail (color, material, pose,
setting) is the point — it feeds the contrastive loss so residual dims learn
what the 256 concepts don't cover. Resumable JSONL.

  VLM_API_URL=http://127.0.0.1:1234/v1/chat/completions \
  VLM_MODEL=qwen/qwen3.6-35b-a3b \
  python scripts/caption_images.py --n-images 24            # sample
  python scripts/caption_images.py                          # all 26k
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_URL = os.environ.get("VLM_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "qwen/qwen3.6-35b-a3b")
LOCAL = "127.0.0.1" in API_URL or "localhost" in API_URL
MAX_SIDE = int(os.environ.get("VLM_IMG_SIDE", "512"))

SYSTEM = "You write concise, concrete image captions. One sentence, plain prose."
PROMPT = (
    "This photo shows a {subject}. Write ONE sentence (15-30 words) describing the "
    "{subject} ITSELF: its color(s), material, texture or finish, shape, distinctive "
    "visual details, and condition. Do NOT describe the background, scene, or location. "
    "Concrete and visual, no opinions, no 'image of'."
)


def encode_image(path: str) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def subject_of(rel: str) -> str:
    return re.sub(r"\d+$", "", os.path.dirname(rel).split(os.sep)[-1]).replace("_", " ")


def caption_one(api_key: str, path: str, subject: str, retries: int = 4) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL, "temperature": 0.3, "max_tokens": 90,
        **({"reasoning_effort": "none"} if LOCAL else {}),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT.format(subject=subject)},
                {"type": "image_url", "image_url": {"url": encode_image(path)}},
            ]},
        ],
    }
    for k in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")
            if len(txt.split()) >= 8:
                return txt
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="data/raw/object_images")
    ap.add_argument("--out", default="data/captions.jsonl")
    ap.add_argument("--n-images", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY", "lm-studio")

    img_dir = os.path.join(ROOT, args.img_dir)
    rels = []
    for cur, _, files in os.walk(img_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                rels.append(os.path.relpath(os.path.join(cur, fn), img_dir))
    rels.sort()
    if args.n_images:
        random.Random(args.seed).shuffle(rels)
        rels = rels[:args.n_images]

    out_path = os.path.join(ROOT, args.out)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                if r.get("caption"):
                    done.add(r["image_id"])
    todo = [r for r in rels if r not in done]
    print(f"endpoint {API_URL} | {len(done)} done, {len(todo)} to caption")

    lock = threading.Lock()
    fails = {"n": 0}

    def work(rel):
        cap = caption_one(api_key, os.path.join(img_dir, rel), subject_of(rel))
        return {"image_id": rel, "caption": cap}

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, rel) for rel in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            row = fut.result()
            if not row["caption"]:
                fails["n"] += 1
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush()
    print(f"done. failures: {fails['n']}")


if __name__ == "__main__":
    main()
