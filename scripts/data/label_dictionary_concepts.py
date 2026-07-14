"""Label images against the frozen concept dictionary with a local VLM.

Production recipe (validated 2026-07-11):
  - google/gemma-4-26b-a4b-qat in LM Studio
  - temperature 0, reasoning disabled, 512 px maximum image side
  - exact-name JSON output with present and uncertain concepts
  - four concurrent requests against an explicitly loaded parallel-4 model

The output is append-only and resumable. A sidecar records the exact prompt,
dictionary, and inference settings; incompatible resumes fail rather than mix
annotations produced under different semantics.

Load the model before a production run:

  lms unload -a
  lms load google/gemma-4-26b-a4b-qat --gpu max \
    --context-length 8192 --parallel 4 --ttl 28800 \
    --identifier google/gemma-4-26b-a4b-qat -y

Then run:

  python scripts/data/label_dictionary_concepts.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image
from tqdm import tqdm

from conceptbasis.splits import image_class, load_split_manifest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PROMPT_VERSION = "gemma26-literal-names-v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-qat"
DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_OUTPUT_TOKENS = 1600

POLICY = """Use each concept's ordinary literal meaning. Mark YES only when it
is clearly visible, defining of the labeled object type, or an explicitly
depicted state/context. A component counts only when it is prominent and the
concept naturally describes the object. Ignore other objects and background;
do not infer incidental traits. Avoid opposing labels unless they clearly
describe different visible regions. True tautologies count."""

SYSTEM = "Return precise structured visual labels."
_thread_state = threading.local()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def display_path(path: Path) -> str:
    """Use a stable repo-relative path when possible, otherwise an absolute one."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session() -> requests.Session:
    if not hasattr(_thread_state, "session"):
        _thread_state.session = requests.Session()
    return _thread_state.session


def encode_image(path: Path, max_side: int) -> str:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def strip_fences(content: str) -> str:
    return re.sub(r"```(?:json)?", "", content).replace("```", "").strip()


def parse_response(
    content: str,
    concept_names: set[str],
) -> tuple[list[str], list[str], list[str]] | None:
    """Return validated present/uncertain/unknown names, or None on bad JSON."""
    cleaned = strip_fences(content)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    raw_present = value.get("yes", value.get("present", []))
    raw_uncertain = value.get("uncertain", [])
    if not isinstance(raw_present, list) or not isinstance(raw_uncertain, list):
        return None
    strings = [item.strip() for item in raw_present + raw_uncertain if isinstance(item, str)]
    unknown = sorted({name for name in strings if name not in concept_names})
    present = sorted({name for name in raw_present if name in concept_names})
    present_set = set(present)
    uncertain = sorted(
        {name for name in raw_uncertain if name in concept_names and name not in present_set}
    )
    if not present:
        return None
    return present, uncertain, unknown


def build_prompt(concepts: list[str], object_label: str) -> str:
    concept_list = "\n".join(f"- {name}" for name in concepts)
    # Put the per-image label last so the long invariant prefix is cacheable.
    return f"""{POLICY}

For all concepts, return only JSON using exact names from the list:
{{"yes": [concept names], "uncertain": [concept names]}}
Every omitted concept is NO.

Concepts:
{concept_list}

The labeled main object is: {object_label}"""


def structured_response_format(concepts: list[str]) -> dict[str, Any]:
    """Constrain vLLM output to canonical, bounded ConceptBasis labels."""
    label = {"type": "string", "enum": concepts}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "concept_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "yes": {
                        "type": "array",
                        "items": label,
                        "minItems": 1,
                        "maxItems": 64,
                    },
                    "uncertain": {
                        "type": "array",
                        "items": label,
                        "maxItems": 32,
                    },
                },
                "required": ["yes", "uncertain"],
                "additionalProperties": False,
            },
        },
    }


def request_one(
    *,
    api_url: str,
    api_key: str | None,
    model: str,
    path: Path,
    image_id: str,
    object_label: str,
    prompt: str,
    concept_names: set[str],
    max_side: int,
    timeout: int,
    retries: int,
    temperature: float = 0,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    retry_repeat_penalty_step: float = 0,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(path, max_side)},
                    },
                ],
            },
        ],
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format is not None:
        payload["response_format"] = response_format
    errors = []
    for attempt in range(1, retries + 1):
        effective_repeat_penalty = (
            repeat_penalty + retry_repeat_penalty_step * (attempt - 1)
            if repeat_penalty is not None
            else None
        )
        if effective_repeat_penalty is not None:
            payload["repeat_penalty"] = effective_repeat_penalty
        started = time.monotonic()
        try:
            response = session().post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            elapsed = time.monotonic() - started
            if not response.ok:
                errors.append(f"attempt {attempt}: HTTP {response.status_code}")
            else:
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                parsed = parse_response(message.get("content") or "", concept_names)
                if parsed is None:
                    errors.append(f"attempt {attempt}: invalid response JSON")
                else:
                    present, uncertain, unknown = parsed
                    usage = body.get("usage") or {}
                    return {
                        "schema_version": SCHEMA_VERSION,
                        "image_id": image_id,
                        "object_class": image_class(image_id),
                        "object_label": object_label,
                        "status": "ok",
                        "present": present,
                        "uncertain": uncertain,
                        "unknown_names": unknown,
                        "model": body.get("model", model),
                        "response_id": body.get("id"),
                        "finish_reason": choice.get("finish_reason"),
                        "latency_seconds": round(elapsed, 3),
                        "usage": usage,
                        "attempts": attempt,
                        "sampling": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "repeat_penalty": effective_repeat_penalty,
                            "retry_repeat_penalty_step": retry_repeat_penalty_step,
                        },
                        "created_at": utc_now(),
                    }
        except Exception as error:  # keep the overnight run alive
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        if attempt < retries:
            time.sleep(1.5 * attempt)
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": image_id,
        "object_class": image_class(image_id),
        "object_label": object_label,
        "status": "error",
        "errors": errors,
        "model": model,
        "created_at": utc_now(),
    }


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_successful_ids(output: Path) -> set[str]:
    done: set[str] = set()
    if not output.exists():
        return done
    with output.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {output}:{line_number}") from error
            if row.get("status") != "ok":
                continue
            image_id = row["image_id"]
            if image_id in done:
                raise ValueError(f"duplicate successful image_id in {output}: {image_id}")
            done.add(image_id)
    return done


def check_or_write_meta(
    meta_path: Path,
    expected: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> None:
    compatibility_keys = {
        "schema_version",
        "prompt_version",
        "prompt_sha256",
        "dictionary_sha256",
        "split_manifest_sha256",
        "selection_sha256",
        "source_image_count",
        "model",
        "temperature",
        "top_p",
        "repeat_penalty",
        "retry_repeat_penalty_step",
        "structured_output",
        "reasoning_effort",
        "max_output_tokens",
        "max_image_side",
        "split",
        "image_dir",
    }
    legacy_defaults = {
        # The completed Gemma run predates these explicit fields. Their missing
        # values are equivalent to the current production defaults.
        "top_p": None,
        "repeat_penalty": None,
        "retry_repeat_penalty_step": 0.04,
        "structured_output": False,
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        mismatches = {
            key: (existing.get(key, legacy_defaults.get(key)), expected.get(key))
            for key in compatibility_keys
            if existing.get(key, legacy_defaults.get(key)) != expected.get(key)
        }
        if mismatches:
            raise ValueError(f"refusing incompatible resume; metadata differs: {mismatches}")
        return
    if output_path is not None and output_path.exists() and output_path.stat().st_size:
        raise ValueError(
            f"refusing to adopt existing output without metadata sidecar: {output_path}"
        )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(expected, indent=2) + "\n")


def preflight_local_lm_studio(api_url: str, model: str, workers: int) -> None:
    """Refuse an accidental LM Studio autoload with unvalidated settings."""
    parsed = urlsplit(api_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return
    native_url = f"{parsed.scheme}://{parsed.netloc}/api/v1/models"
    response = requests.get(native_url, timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    instances = [
        instance
        for entry in models
        if entry.get("key") == model
        for instance in entry.get("loaded_instances", [])
        if instance.get("id") == model
    ]
    if not instances:
        raise RuntimeError(
            f"{model} is not explicitly loaded in LM Studio; use the load command "
            "in this script's docstring before labeling"
        )
    parallel = max(instance.get("config", {}).get("parallel", 1) for instance in instances)
    if parallel < workers:
        raise RuntimeError(
            f"LM Studio loaded parallel={parallel}, but --workers={workers}; reload the "
            "model with matching --parallel or lower --workers"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="data/raw/object_images")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", default="data/dictionary_labels_train_gemma26.jsonl")
    parser.add_argument("--api-url", default=os.environ.get("VLM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--retry-repeat-penalty-step", type=float, default=0.04)
    parser.add_argument(
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain vLLM output to canonical names with JSON-schema decoding",
    )
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fsync-every", type=int, default=50)
    parser.add_argument("--skip-local-preflight", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise ValueError("labeling the sealed test split requires --allow-test")
    if args.workers < 1 or args.max_side < 1 or args.retries < 1:
        raise ValueError("workers, max-side, and retries must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if args.repeat_penalty is not None and args.repeat_penalty <= 0:
        raise ValueError("repeat-penalty must be positive")
    if args.retry_repeat_penalty_step < 0:
        raise ValueError("retry-repeat-penalty-step must be nonnegative")
    if not args.skip_local_preflight:
        preflight_local_lm_studio(args.api_url, args.model, args.workers)

    img_dir = resolve(ROOT, args.img_dir)
    dictionary_path = resolve(ROOT, args.dictionary)
    split_manifest_path = resolve(ROOT, args.split_manifest)
    output = resolve(ROOT, args.out)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    error_path = output.with_suffix(output.suffix + ".errors.jsonl")

    dictionary = json.loads(dictionary_path.read_text())
    concepts = [row["name"] for row in dictionary]
    if len(concepts) != len(set(concepts)):
        raise ValueError("dictionary concept names must be unique")
    concept_names = set(concepts)
    prompt_template = build_prompt(concepts, "{object_label}")

    manifest = load_split_manifest(str(ROOT), str(split_manifest_path))
    paths = sorted(
        path
        for path in img_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and manifest["classes"][image_class(str(path.relative_to(img_dir)))] == args.split
    )
    if args.n_images is not None:
        random.Random(args.seed).shuffle(paths)
        paths = paths[: args.n_images]

    selected_ids = [str(path.relative_to(img_dir)) for path in paths]
    selection_sha256 = sha256_text("\n".join(selected_ids) + "\n")

    expected_meta = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM,
        "prompt": prompt_template,
        "prompt_sha256": sha256_text(SYSTEM + "\n" + prompt_template),
        "dictionary": display_path(dictionary_path),
        "dictionary_sha256": sha256(dictionary_path),
        "split_manifest": display_path(split_manifest_path),
        "split_manifest_sha256": sha256(split_manifest_path),
        "image_dir": display_path(img_dir),
        "split": args.split,
        "source_image_count": len(paths),
        "selection_sha256": selection_sha256,
        "model": args.model,
        "api_url": args.api_url,
        "temperature": 0,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "retry_repeat_penalty_step": args.retry_repeat_penalty_step,
        "structured_output": args.structured_output,
        "reasoning_effort": "none",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_image_side": args.max_side,
        "workers": args.workers,
        "created_at": utc_now(),
    }
    check_or_write_meta(meta_path, expected_meta, output_path=output)

    done = load_successful_ids(output)
    work = []
    for path in paths:
        image_id = str(path.relative_to(img_dir))
        if image_id not in done:
            object_class = image_class(image_id)
            work.append((path, image_id, object_class.replace("_", " ")))

    print(
        f"model={args.model} endpoint={args.api_url} workers={args.workers} "
        f"split={args.split} images={len(paths)} done={len(done)} remaining={len(work)}",
        flush=True,
    )
    if not work:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("VLM_API_KEY")

    def annotate(item: tuple[Path, str, str]) -> dict[str, Any]:
        path, image_id, object_label = item
        return request_one(
            api_url=args.api_url,
            api_key=api_key,
            model=args.model,
            path=path,
            image_id=image_id,
            object_label=object_label,
            prompt=build_prompt(concepts, object_label),
            concept_names=concept_names,
            max_side=args.max_side,
            timeout=args.timeout,
            retries=args.retries,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            retry_repeat_penalty_step=args.retry_repeat_penalty_step,
            response_format=structured_response_format(concepts)
            if args.structured_output
            else None,
        )

    successes = failures = writes_since_sync = 0
    started = time.monotonic()
    with (
        output.open("a", buffering=1) as output_file,
        error_path.open("a", buffering=1) as error_file,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures = [executor.submit(annotate, item) for item in work]
        for future in tqdm(as_completed(futures), total=len(futures)):
            row = future.result()
            target = output_file if row["status"] == "ok" else error_file
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
            if row["status"] == "ok":
                successes += 1
                writes_since_sync += 1
                if writes_since_sync >= args.fsync_every:
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    writes_since_sync = 0
            else:
                failures += 1
        output_file.flush()
        os.fsync(output_file.fileno())

    elapsed = time.monotonic() - started
    print(
        f"completed={successes} failed={failures} wall_seconds={elapsed:.1f} "
        f"effective_seconds_per_success={elapsed / max(1, successes):.3f}",
        flush=True,
    )
    if failures:
        raise SystemExit(f"{failures} rows failed; rerun to retry them")


if __name__ == "__main__":
    main()
