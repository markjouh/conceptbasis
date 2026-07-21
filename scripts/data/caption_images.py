"""Stage 1a — Caption THINGS images for contrastive adapter training.

One grounded ~8-30 word sentence per image; the concept label (folder name)
is passed as the subject. Instance-level detail (color, material, pose,
setting) is the point — it feeds the contrastive loss so residual dims learn
what the 256 concepts don't cover. Resumable JSONL.

Use ``scripts/vllm/caption_images.sh`` so the model, transport, concurrency,
run identity, and output path are explicit.
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
    sha256_text,
    validate_image_transport,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
API_URL = os.environ.get("VLM_API_URL", LOCAL_VLLM_API_URL)
MODEL = os.environ.get("VLM_MODEL", GEMMA_MODEL_ID)
MAX_SIDE = int(os.environ.get("VLM_IMG_SIDE", "768"))
MAX_OUTPUT_TOKENS = int(os.environ.get("VLM_MAX_OUTPUT_TOKENS", "64"))
RETRY_MAX_OUTPUT_TOKENS = int(os.environ.get("VLM_RETRY_MAX_OUTPUT_TOKENS", "90"))

SYSTEM = "You write concise, concrete image captions. One sentence, plain prose."
LEGACY_PROMPT = (
    "This photo shows a {subject}. Write ONE sentence (15-30 words) describing the "
    "{subject} ITSELF: its color(s), material, texture or finish, shape, distinctive "
    "visual details, and condition. Do NOT describe the background, scene, or location. "
    "Concrete and visual, no opinions, no 'image of'."
)
PROMPT = (
    "Write one concise, natural sentence describing the depicted {subject}. Include the "
    "subject noun and the most salient instance-specific visual details that are clearly "
    "supported by the image, such as color, shape, material, texture, pose, viewpoint, or "
    "condition. Select only applicable details; do not mechanically cover every category or "
    "speculate. Ignore incidental background unless it helps disambiguate the object. Do not "
    "mention brands, opinions, or 'image/photo of'. Aim for 8-30 words."
)
CLIP_GROUNDED_PROMPT = PROMPT
PROMPT_VARIANTS = {"legacy": LEGACY_PROMPT, "clip-grounded": PROMPT}
PROMPT_WORD_BOUNDS = {"legacy": (15, 30), "clip-grounded": (8, 30)}


def encode_image(path: str, transport: str = "base64") -> str:
    return image_reference(path, transport=transport, max_side=MAX_SIDE)


def subject_of(rel: str) -> str:
    return re.sub(r"\d+$", "", os.path.dirname(rel).split(os.sep)[-1]).replace("_", " ")


def valid_caption(text: str, min_words: int = 8, max_words: int = 30) -> bool:
    """Accept exactly one sentence within the selected prompt's word bounds."""
    normalized = " ".join(text.split())
    if not min_words <= len(normalized.split()) <= max_words:
        return False
    unquoted = normalized.rstrip("\"')]}")
    if not re.search(r"[.!?]$", unquoted):
        return False
    # Reject a likely additional sentence boundary while allowing punctuation
    # inside tokens such as ``3.5-inch`` and abbreviations such as ``U.S.`` or
    # ``Brgy. Rima``. The prompt already asks for one sentence, so a short list
    # of ordinary sentence starters is more robust than treating every period
    # before a capital as a boundary.
    body = unquoted[:-1]
    return not re.search(
        r"[.!?]+\s*(?=(?:It|The|This|That|These|Those|A|An|He|She|They|We|I)\b)",
        body,
    )


def build_run_metadata(
    run_id: str,
    *,
    image_transport: str,
    max_output_tokens: int,
    retry_max_output_tokens: int,
    image_dir: str,
    selected_ids: list[str],
    selection_seed: int,
    api_url: str = API_URL,
    model: str = MODEL,
    prompt_template: str = PROMPT,
    min_words: int = 8,
    max_words: int = 30,
) -> dict:
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
        "caption_word_bounds": [min_words, max_words],
        "image_dir": image_dir,
        "source_image_count": len(selected_ids),
        "selection_seed": selection_seed,
        "selection_sha256": ordered_ids_sha256(selected_ids),
    }


def caption_one(
    api_key: str,
    path: str,
    subject: str,
    retries: int = 4,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    retry_max_output_tokens: int = RETRY_MAX_OUTPUT_TOKENS,
    image_transport: str = "base64",
    api_url: str = API_URL,
    model: str = MODEL,
    prompt_template: str = PROMPT,
    min_words: int = 8,
    max_words: int = 30,
) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "temperature": 0.3, "max_tokens": max_output_tokens,
        **({"reasoning_effort": "none"} if is_local_endpoint(api_url) else {}),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_template.format(subject=subject)},
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
            txt = " ".join(choice["message"]["content"].split())
            if choice.get("finish_reason") == "length":
                if payload["max_tokens"] < retry_max_output_tokens:
                    payload["max_tokens"] = retry_max_output_tokens
                continue
            if valid_caption(txt, min_words=min_words, max_words=max_words):
                return txt
            if len(txt.split()) < min_words and payload["max_tokens"] < retry_max_output_tokens:
                payload["max_tokens"] = retry_max_output_tokens
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--img-dir", default="data/raw/object_images")
    ap.add_argument(
        "--out",
        default="data/captions_vllm_gemma4_nvfp4_clip_grounded_v2.jsonl",
    )
    ap.add_argument("--n-images", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
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
    ap.add_argument(
        "--prompt-variant", choices=tuple(PROMPT_VARIANTS), default="clip-grounded"
    )
    args = ap.parse_args()
    prompt_template = PROMPT_VARIANTS[args.prompt_variant]
    min_words, max_words = PROMPT_WORD_BOUNDS[args.prompt_variant]
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
    try:
        api_key = api_key_for(args.api_url)
    except ValueError as error:
        ap.error(str(error))

    img_dir = os.path.join(ROOT, args.img_dir)
    rels = [path.relative_to(img_dir).as_posix() for path in recursive_image_paths(img_dir)]
    if args.n_images:
        random.Random(args.seed).shuffle(rels)
        rels = rels[:args.n_images]

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
                selected_ids=rels,
                selection_seed=args.seed,
                api_url=args.api_url,
                model=args.model,
                prompt_template=prompt_template,
                min_words=min_words,
                max_words=max_words,
            ),
        )
    done = load_completed_image_ids(out_path, required_field="caption")
    todo = [r for r in rels if r not in done]
    print(
        f"endpoint {args.api_url} model={args.model} | "
        f"{len(done)} done, {len(todo)} to caption"
    )

    lock = threading.Lock()
    fails = {"n": 0}
    started = time.monotonic()

    def work(rel):
        cap = caption_one(
            api_key,
            os.path.join(img_dir, rel),
            subject_of(rel),
            max_output_tokens=args.max_output_tokens,
            retry_max_output_tokens=args.retry_max_output_tokens,
            image_transport=args.image_transport,
            api_url=args.api_url,
            model=args.model,
            prompt_template=prompt_template,
            min_words=min_words,
            max_words=max_words,
        )
        return {"image_id": rel, "caption": cap}

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, rel) for rel in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            row = fut.result()
            if not row["caption"]:
                fails["n"] += 1
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush()
    elapsed = time.monotonic() - started
    successes = len(todo) - fails["n"]
    print(
        f"done. failures={fails['n']} wall_seconds={elapsed:.3f} "
        f"effective_images_per_second={successes / max(elapsed, 1e-9):.3f}"
    )


if __name__ == "__main__":
    main()
