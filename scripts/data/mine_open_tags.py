"""Stage 1b — Mine open-set image tags used to discover dictionary concepts.

Each image receives a short, positive, nonredundant list of visible or commonly
associated properties. Train-split tags are the bottom-up input to dictionary
construction; they are not the later exhaustive fixed-dictionary labels.

The JSONL job is append-only and resumable. Use
``scripts/vllm/mine_open_tags.sh`` for the pinned local Gemma configuration.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from conceptbasis.splits import image_class, load_split_manifest
from conceptbasis.vlm import (
    GEMMA_MODEL_ID,
    LOCAL_VLLM_API_URL,
    api_key_for,
    check_or_write_run_metadata,
    image_reference,
    is_local_endpoint,
    load_completed_image_ids,
    ordered_ids_sha256,
    recursive_image_paths,
    session,
    sha256_file,
    sha256_text,
    validate_image_transport,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
API_URL = os.environ.get("VLM_API_URL", LOCAL_VLLM_API_URL)
MODEL = os.environ.get("VLM_MODEL", GEMMA_MODEL_ID)

SYSTEM = (
    "You list reusable, visually grounded properties of objects—both how they look and "
    "what kind of thing they are. Every property must be reasonably unambiguous and "
    "non-vacuous. Output a JSON array of strings only."
)
PROMPT = (
    "List 5-50 GENERAL properties of it that could apply to many DIFFERENT kinds of "
    "objects (not just this one), across both appearance AND kind/function: "
    "appearance (color, material, finish, texture, shape, transparency, reflectivity, "
    "rigidity) and kind/function (broad category and use, e.g. animal, plant, food, tool, "
    "container, vehicle, furniture, clothing; natural vs manmade; edible, wearable, "
    "handheld, electronic). "
    "Every proposed property must pass BOTH tests: (1) UNAMBIGUOUS—a consistent annotator "
    "can understand one reasonably clear meaning and judge whether it applies; avoid vague or "
    "multiply interpretable words. (2) NON-VACUOUS—it meaningfully distinguishes this object "
    "from many ordinary objects; do not state properties that are almost automatically true of "
    "physical objects or IRL photographs. Bad vacuous or underspecified tags include solid, "
    "three-dimensional, physical, visible, visual, various shapes/textures, ... "
    "Only properties you can judge from the image. Do NOT list object-specific parts, the "
    "object's exact name, or non-visual facts (price, brand, history). "
    '(For a laptop, good examples, ...: ["black","metallic","rectangular","rigid","manmade","electronic","portable"]; '
    'bad examples, ...: ["trackpad","keys","laptop","solid","three-dimensional"].) '
    "Cover distinct appearance and kind/function dimensions; never add a vague, ambiguous, "
    "marginal, or vacuous tag merely to fill the list. "
    "Every tag must assert a property that IS PRESENT in the depicted object. If a property "
    "does not apply, omit it—never encode its absence or negation (for example, never write "
    "'non-...', 'not ...', 'without ...', or '*-absent'). "
    "Keep the tags mutually non-redundant and prefer reusable atomic properties. Do not add a "
    "compound or near-duplicate that merely restates tags already in the list; for example, if "
    "you list 'blue' and 'denim', do not also list 'blue denim'. "
    "Each property 1-2 words. Return ONLY a JSON array of strings."
)

MIN_ATTRIBUTES = 5
MAX_ATTRIBUTES = 50
RETRY_PROMPT = (
    "Your previous response failed validation. Return 5-50 UNIQUE positive properties, "
    "keep every property to 1-2 words, and preserve all unambiguous/non-vacuous rules above."
)

NEGATED_ATTRIBUTE = re.compile(
    r"^(?:no|not|non|without|lacking)(?:[ -]|$)|(?:^|[ -])(?:absent|missing|free)$"
)


def label_from_id(image_id: str) -> str:
    # Full THINGS paths are nested as ``class/image.jpg`` whereas the CC0
    # representatives are flat ``class.jpg`` files.  Always derive the known
    # subject from the split unit, not from the exemplar filename.
    name = re.sub(r"\d+$", "", image_class(image_id))   # bow2 -> bow
    return name.replace("_", " ").strip()


MAX_SIDE = int(os.environ.get("VLM_IMG_SIDE", "768"))
MAX_OUTPUT_TOKENS = int(os.environ.get("VLM_MAX_OUTPUT_TOKENS", "192"))
RETRY_MAX_OUTPUT_TOKENS = int(os.environ.get("VLM_RETRY_MAX_OUTPUT_TOKENS", "300"))


def encode_image(path: str, transport: str = "base64") -> str:
    """Use an exact local file URI or a bounded JPEG for remote APIs."""
    return image_reference(path, transport=transport, max_side=MAX_SIDE)


def validated_attributes(content: str) -> list[str] | None:
    """Normalize attributes and enforce the minimum count and word limits."""
    text = re.sub(r"```(?:json)?", "", content)
    start, end = text.find("["), text.rfind("]")
    values = None
    if start != -1 and end > start:
        try:
            candidate = json.loads(text[start : end + 1])
            if isinstance(candidate, list):
                values = candidate
        except json.JSONDecodeError:
            pass
    if values is None:
        values = re.findall(r'"([^"]+)"', text)
    normalized = [str(value).strip().lower() for value in values]
    normalized = [
        value
        for value in normalized
        if re.fullmatch(r"[a-z][a-z /-]{1,30}", value)
        and 1 <= len(value.split()) <= 2
        and not NEGATED_ATTRIBUTE.search(value)
    ]
    unique = list(dict.fromkeys(normalized))
    return unique if MIN_ATTRIBUTES <= len(unique) <= MAX_ATTRIBUTES else None


def build_run_metadata(
    run_id: str,
    *,
    image_transport: str,
    max_output_tokens: int,
    retry_max_output_tokens: int,
    image_dir: str,
    split: str,
    split_manifest: str,
    selected_ids: list[str],
    selection_seed: int,
    api_url: str = API_URL,
    model: str = MODEL,
) -> dict:
    prompt_template = (
        PROMPT
        + "\nThis is a photo of a {subject}."
        + "\nInvalid-output retry instruction: "
        + RETRY_PROMPT
    )
    split_manifest_path = Path(ROOT, split_manifest)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "model": model,
        "api_url": api_url,
        "model_revision": os.environ.get("VLM_MODEL_REVISION"),
        "server_profile": os.environ.get("VLM_SERVER_PROFILE"),
        "prompt_sha256": sha256_text(f"{SYSTEM}\n{prompt_template}"),
        "image_transport": image_transport,
        "client_transform": (
            "none" if image_transport == "file" else "resize-max-side-jpeg-q88"
        ),
        "max_image_side": MAX_SIDE if image_transport == "base64" else None,
        "max_output_tokens": max_output_tokens,
        "retry_max_output_tokens": retry_max_output_tokens,
        "temperature": 0.3,
        "image_dir": image_dir,
        "split": split,
        "split_manifest": split_manifest,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "source_image_count": len(selected_ids),
        "selection_seed": selection_seed,
        "selection_sha256": ordered_ids_sha256(selected_ids),
    }


def attrs_one(
    api_key: str,
    path: str,
    subject: str = "",
    retries: int = 4,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    retry_max_output_tokens: int = RETRY_MAX_OUTPUT_TOKENS,
    image_transport: str = "base64",
    api_url: str = API_URL,
    model: str = MODEL,
) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    text = PROMPT + (f"\nThis is a photo of a {subject}." if subject else "")
    payload = {
        "model": model, "temperature": 0.3, "max_tokens": max_output_tokens,
        **({"reasoning_effort": "none"} if is_local_endpoint(api_url) else {}),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(path, image_transport)},
                },
            ]},
        ],
    }
    for k in range(retries):
        try:
            r = session().post(api_url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                if payload["max_tokens"] < retry_max_output_tokens:
                    payload["max_tokens"] = retry_max_output_tokens
                continue
        except Exception:
            time.sleep(1.5 * (k + 1)); continue
        vals = validated_attributes(content)
        if vals is not None:
            return vals
        payload["messages"][1]["content"][0]["text"] = f"{text}\n{RETRY_PROMPT}"
        if payload["max_tokens"] < retry_max_output_tokens:
            payload["max_tokens"] = retry_max_output_tokens
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--img-dir", default="data/raw/object_images_CC0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--image-transport",
        choices=("base64", "file"),
        default=os.environ.get("VLM_IMAGE_TRANSPORT", "base64"),
    )
    ap.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    ap.add_argument(
        "--retry-max-output-tokens", type=int, default=RETRY_MAX_OUTPUT_TOKENS
    )
    ap.add_argument("--split-manifest", default="data/splits.json")
    ap.add_argument("--split", choices=("train", "dev", "test"), default="train")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.workers < 1 or args.max_output_tokens < 1:
        ap.error("--workers and --max-output-tokens must be positive")
    if args.retry_max_output_tokens < args.max_output_tokens:
        ap.error("--retry-max-output-tokens must be >= --max-output-tokens")
    if args.run_id is not None and not args.run_id.strip():
        ap.error("--run-id must not be empty")
    try:
        validate_image_transport(args.api_url, args.image_transport)
    except ValueError as error:
        ap.error(str(error))
    if args.split == "test" and not args.allow_test:
        raise ValueError("tagging test requires --allow-test")
    if args.out is None:
        args.out = (
            "data/heldout/attributes_test.jsonl"
            if args.split == "test"
            else f"data/attributes_{args.split}.jsonl"
        )
    try:
        api_key = api_key_for(args.api_url)
    except ValueError as error:
        ap.error(str(error))
    print(f"endpoint: {args.api_url}  model: {args.model}")

    img_dir = os.path.join(ROOT, args.img_dir)
    manifest = load_split_manifest(ROOT, args.split_manifest)
    paths = [str(path) for path in recursive_image_paths(img_dir)]
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
    if args.run_id is not None:
        check_or_write_run_metadata(
            Path(out_path),
            build_run_metadata(
                args.run_id,
                image_transport=args.image_transport,
                max_output_tokens=args.max_output_tokens,
                retry_max_output_tokens=args.retry_max_output_tokens,
                image_dir=args.img_dir,
                split=args.split,
                split_manifest=args.split_manifest,
                selected_ids=[os.path.relpath(path, img_dir) for path in paths],
                selection_seed=args.seed,
                api_url=args.api_url,
                model=args.model,
            ),
        )
    done = load_completed_image_ids(out_path, required_field="attributes")
    todo = [p for p in paths if os.path.relpath(p, img_dir) not in done]
    print(f"{len(done)} done, {len(todo)} to label")

    lock = threading.Lock()
    current_rows = []
    started = time.monotonic()
    def work(p):
        iid = os.path.relpath(p, img_dir)
        return {
            "image_id": iid,
            "attributes": attrs_one(
                api_key,
                p,
                label_from_id(iid),
                max_output_tokens=args.max_output_tokens,
                retry_max_output_tokens=args.retry_max_output_tokens,
                image_transport=args.image_transport,
                api_url=args.api_url,
                model=args.model,
            ),
        }

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            row = fut.result()
            current_rows.append(row)
            with lock:
                if row["attributes"]:
                    f.write(json.dumps(row) + "\n"); f.flush()
                else:
                    error_path = out_path + ".errors.jsonl"
                    with open(error_path, "a") as error_file:
                        error_file.write(
                            json.dumps(
                                {
                                    "image_id": row["image_id"],
                                    "error": "no_valid_attributes_after_retries",
                                }
                            )
                            + "\n"
                        )

    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    allattrs = [a for r in rows for a in r.get("attributes", [])]
    from collections import Counter
    print(f"\ntotal attribute mentions: {len(allattrs)} | unique: {len(set(allattrs))}")
    print("most common:", Counter(allattrs).most_common(20))
    elapsed = time.monotonic() - started
    successes = sum(bool(row.get("attributes")) for row in current_rows)
    failures = len(current_rows) - successes
    print(
        f"completed={successes} failed={failures} "
        f"wall_seconds={elapsed:.3f} "
        f"effective_images_per_second={successes / max(elapsed, 1e-9):.3f}"
    )


if __name__ == "__main__":
    main()
