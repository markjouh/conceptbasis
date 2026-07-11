"""Mine visual attributes per image with a VLM (OpenRouter). Bottom-up concept
dictionary: pool these across images, then cluster/dedup and select prevalent +
diverse ones as concept directions.

  export OPENROUTER_API_KEY=...
  python scripts/data/mine_attributes.py --img-dir data/raw/<unzipped> --n-images 300
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from conceptbasis.splits import image_class, load_split_manifest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
API_URL = os.environ.get("VLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "google/gemma-4-26b-a4b-it")
LOCAL = "127.0.0.1" in API_URL or "localhost" in API_URL

SYSTEM = ("You list general properties of objects — both how they look and what kind of "
          "thing they are. Output a JSON array of strings only.")
PROMPT = (
    "List 10-15 GENERAL properties of it that could apply to many DIFFERENT kinds of "
    "objects (not just this one), across both appearance AND kind/function: "
    "appearance (color, material, finish, texture, shape, transparency, reflectivity, "
    "rigidity) and kind/function (broad category and use, e.g. animal, plant, food, tool, "
    "container, vehicle, furniture, clothing; natural vs manmade; edible, wearable, "
    "handheld, electronic). "
    "Only properties you can judge from the image. Do NOT list object-specific parts, the "
    "object's exact name, or non-visual facts (price, brand, history). "
    '(For a laptop, good: ["black","metallic","rectangular","rigid","manmade","electronic","portable"]; '
    'bad: ["trackpad","keys","laptop"].) '
    "State each property POSITIVELY (avoid 'non-...' / 'not ...'). "
    "Each property 1-2 words. Return ONLY a JSON array of strings."
)


def label_from_id(image_id: str) -> str:
    name = re.sub(r"\d+$", "", os.path.splitext(image_id)[0])   # bow2 -> bow
    return name.replace("_", " ").strip()


MAX_SIDE = int(os.environ.get("VLM_IMG_SIDE", "512"))


def encode_image(path: str) -> str:
    """Downscale to MAX_SIDE before base64 — vision tokens scale with pixels,
    so this cuts prompt-processing time a lot at no quality loss for attributes."""
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def attrs_one(api_key: str, path: str, subject: str = "", retries: int = 4) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    text = (f"This is a photo of a {subject}. " if subject else "") + PROMPT
    payload = {
        "model": MODEL, "temperature": 0.3, "max_tokens": 300,
        **({"reasoning_effort": "none"} if LOCAL else {}),   # disable qwen thinking on LM Studio
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": encode_image(path)}},
            ]},
        ],
    }
    for k in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(1.5 * (k + 1)); continue
        t = re.sub(r"```(?:json)?", "", content)                 # strip markdown fence
        i, j = t.find("["), t.rfind("]")
        arr = None
        if i != -1 and j > i:
            try:
                arr = json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                arr = None
        if arr is None:
            arr = re.findall(r'"([^"]+)"', t)                    # fallback
        vals = [str(s).strip().lower() for s in arr]
        vals = [v for v in vals if re.fullmatch(r"[a-z][a-z /-]{1,30}", v) and 1 <= len(v.split()) <= 4]
        if len(vals) >= 5:
            return vals[:20]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--split", choices=("train", "dev", "test"), default="train")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("tagging test requires --allow-test")
    if args.out is None:
        args.out = (
            "data/heldout/attributes_test.jsonl"
            if args.split == "test"
            else f"data/attributes_{args.split}.jsonl"
        )
    api_key = os.environ.get("OPENROUTER_API_KEY", "lm-studio")
    if not LOCAL and api_key == "lm-studio":
        sys.exit("set OPENROUTER_API_KEY")
    print(f"endpoint: {API_URL}  model: {MODEL}")

    img_dir = os.path.join(ROOT, args.img_dir)
    manifest = load_split_manifest(ROOT, args.split_manifest)
    paths = []
    for cur, _, files in os.walk(img_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cur, fn))
    paths.sort()
    paths = [
        path
        for path in paths
        if manifest["classes"][image_class(os.path.relpath(path, img_dir))] == args.split
    ]
    random.Random(args.seed).shuffle(paths)
    if args.n_images is not None:
        paths = paths[:args.n_images]
    print(f"found images, sampling {len(paths)}")

    out_path = os.path.join(ROOT, args.out)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                if r.get("attributes"):
                    done.add(r["image_id"])
    todo = [p for p in paths if os.path.relpath(p, img_dir) not in done]
    print(f"{len(done)} done, {len(todo)} to label")

    lock = threading.Lock()
    def work(p):
        iid = os.path.relpath(p, img_dir)
        return {"image_id": iid, "attributes": attrs_one(api_key, p, label_from_id(iid))}

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush()

    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    allattrs = [a for r in rows for a in r.get("attributes", [])]
    from collections import Counter
    print(f"\ntotal attribute mentions: {len(allattrs)} | unique: {len(set(allattrs))}")
    print("most common:", Counter(allattrs).most_common(20))


if __name__ == "__main__":
    main()
