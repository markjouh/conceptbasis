"""Stage 3 — Label every image against the finalized fixed dictionary.

The production wrapper uses the pinned Gemma 4 26B NVFP4 vLLM runtime,
temperature zero, reasoning disabled, exact dictionary names, and a strictly
validated ``concept: YES/NO`` checklist.

The output is append-only and resumable. A sidecar records the exact prompt,
dictionary, and inference settings; incompatible resumes fail rather than mix
annotations produced under different semantics.

Unlike open-tag mining, this is closed-set and exhaustive. Use
``scripts/vllm/serve_vlm.sh`` and ``scripts/vllm/label_fixed_dictionary.sh``
for the validated local configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from conceptbasis.splits import image_class, load_split_manifest
from conceptbasis.vlm import (
    GEMMA_MODEL_ID,
    LOCAL_VLLM_API_URL,
    check_or_write_metadata,
    image_reference,
    load_completed_image_ids,
    recursive_image_paths,
    session,
    sha256_file as sha256,
    sha256_text,
    validate_image_transport,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PROMPT_VERSION = "gemma26-canonical-leaders-v2"
NUMBERED_PROMPT_VERSION = "gemma26-exhaustive-natural-predicate-numbered-v8"
NAMED_BINARY_PROMPT_VERSION = (
    "gemma26-object-grounded-named-binary-v10"
)
DEFAULT_MODEL = GEMMA_MODEL_ID
DEFAULT_API_URL = LOCAL_VLLM_API_URL
MAX_OUTPUT_TOKENS = 1600

POLICY = """Use each listed concept's ordinary literal meaning. The merged
member phrases used to construct the dictionary are not additional labeling
targets. Mark YES only when the listed concept itself is clearly visible,
defining of the labeled object type, or an explicitly depicted state/context.
A component counts only when it is prominent and the concept naturally
describes the object. Ignore other objects and background; do not infer
incidental traits or merely associated categories. Avoid opposing labels
unless they clearly describe different visible regions. True tautologies
count."""

SYSTEM = "Return precise structured visual labels."

NUMBERED_SYSTEM = (
    "Be exhaustive over natural object properties; reject technicalities and "
    "class-inappropriate predicates. Complete the exact numbered checklist."
)

NAMED_BINARY_SYSTEM = (
    "Label only the named object. Complete every checklist item."
)

NUMBERED_POLICY = """Be exhaustive: answer YES for every genuine property of the
depicted object or subtype, including uncommon, secondary, and non-salient
appearance, color, material, condition, function, category, and characteristic-use
properties. Labels are independent; include specific and broad labels when both
apply.

Answer NO only for a strained technicality or merely possible association, a
predicate unnatural for the semantic class, a hidden-part trait projected onto
the whole, or a scene fact. Animals are not weatherproof; living or wild animals
are not edible unless treated or depicted as food. Heterogeneous requires a
visibly salient, natural whole-object mixture. Perforated requires visible holes
characterizing the object. Various shapes requires a whole or group with distinct
shape types, not merely body parts. Indoor/outdoor may describe characteristic
use, not photo setting. Categories must fit the depicted subtype. Judge stable
surface color, not lighting or glare. Judge each item independently."""

NAMED_BINARY_POLICY = """Mark YES when the concept clearly and naturally describes
the labeled object as shown or its ordinary subtype. Be exhaustive, but judge
only that object: ignore the background, other objects, and the photo setting.
Do not transfer properties from its source, contents, or an associated whole.
Reject technicalities, incidental possibilities, hidden-part guesses, and
descriptions unnatural for this kind of object. Characteristic use may count;
color must belong to the object rather than lighting or glare."""


def display_path(path: Path) -> str:
    """Use a stable repo-relative path when possible, otherwise an absolute one."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_image(path: Path, max_side: int, transport: str = "base64") -> str:
    return image_reference(path, transport=transport, max_side=max_side)


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


def build_numbered_prompt(concepts: list[str], object_label: str) -> str:
    concept_list = "\n".join(
        f"{index}. {name}" for index, name in enumerate(concepts, start=1)
    )
    return f"""{NUMBERED_POLICY}

Return exactly one line per proposed concept in the same order, using this form:
1. exact concept name: YES
2. exact concept name: NO

Repeat each exact concept name. Use only YES or NO. Do not add commentary, omit
items, combine items, or reorder them.

Proposed concepts:
{concept_list}

The labeled main object is: {object_label}"""


def build_named_binary_prompt(concepts: list[str], object_label: str) -> str:
    concept_list = "\n".join(f"- {name}" for name in concepts)
    return f"""{NAMED_BINARY_POLICY}

For every proposed concept, in order, return one line in this form:
exact concept name: YES
exact concept name: NO

Repeat each exact name once. Add no numbering or commentary; omit and reorder
nothing.

Proposed concepts:
{concept_list}

The labeled main object is: {object_label}"""


def named_binary_regex(concepts: list[str]) -> str:
    """Exact xgrammar-compatible checklist with a binary choice per leader."""
    return "\n".join(re.escape(name) + ": (YES|NO)" for name in concepts)


def parse_numbered_response(
    content: str,
    concepts: list[str],
) -> tuple[list[str], list[str], list[str]] | None:
    lines = [
        line.strip()
        for line in re.sub(r"```(?:text)?", "", content)
        .replace("```", "")
        .splitlines()
        if line.strip()
    ]
    if len(lines) != len(concepts):
        return None
    present: list[str] = []
    for index, (line, expected_name) in enumerate(zip(lines, concepts), start=1):
        match = re.fullmatch(r"(\d+)\.\s+(.+):\s+(YES|NO)", line)
        if (
            match is None
            or int(match.group(1)) != index
            or match.group(2) != expected_name
        ):
            return None
        if match.group(3) == "YES":
            present.append(expected_name)
    return present, [], []


def parse_named_binary_response(
    content: str,
    concepts: list[str],
) -> tuple[list[str], list[str], list[str]] | None:
    lines = [
        line.strip()
        for line in re.sub(r"```(?:text)?", "", content)
        .replace("```", "")
        .splitlines()
        if line.strip()
    ]
    if len(lines) != len(concepts):
        return None
    present: list[str] = []
    for line, expected_name in zip(lines, concepts):
        if line == f"{expected_name}: YES":
            present.append(expected_name)
        elif line != f"{expected_name}: NO":
            return None
    return present, [], []


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
                        "minItems": 0,
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
    max_output_tokens: int,
    image_transport: str,
    timeout: int,
    retries: int,
    temperature: float = 0,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    retry_repeat_penalty_step: float = 0,
    response_format: dict[str, Any] | None = None,
    structured_outputs: dict[str, Any] | None = None,
    system_prompt: str = SYSTEM,
    response_mode: Literal["json", "numbered", "named_binary"] = "json",
    ordered_concepts: list[str] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(path, max_side, image_transport)
                        },
                    },
                ],
            },
        ],
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if response_format is not None:
        payload["response_format"] = response_format
    if structured_outputs is not None:
        payload["structured_outputs"] = structured_outputs
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
                errors.append(
                    f"attempt {attempt}: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            else:
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                if response_mode == "numbered" and ordered_concepts is not None:
                    parsed = parse_numbered_response(content, ordered_concepts)
                elif response_mode == "named_binary" and ordered_concepts is not None:
                    parsed = parse_named_binary_response(content, ordered_concepts)
                else:
                    parsed = parse_response(content, concept_names)
                if parsed is None:
                    errors.append(f"attempt {attempt}: invalid {response_mode} response")
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
        "run_id",
        "model",
        "model_revision",
        "server_profile",
        "temperature",
        "top_p",
        "repeat_penalty",
        "retry_repeat_penalty_step",
        "structured_output",
        "reasoning_effort",
        "max_output_tokens",
        "max_image_side",
        "client_transform",
        "image_transport",
        "review_mode",
        "constrain_checklist",
        "skip_labeled_from",
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
        "image_transport": "base64",
        "client_transform": "resize-max-side-jpeg-q88",
        "run_id": None,
        "model_revision": None,
        "server_profile": None,
        "review_mode": "direct",
        "constrain_checklist": False,
        "skip_labeled_from": [],
    }
    check_or_write_metadata(
        meta_path,
        expected,
        output_path=output_path,
        compatibility_keys=compatibility_keys,
        legacy_defaults=legacy_defaults,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--img-dir", default="data/raw/object_images")
    parser.add_argument(
        "--dictionary",
        default="data/dictionary_usage_profile_v8.json",
    )
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--out",
        default=(
            "data/dictionary_labels_train_vllm_gemma4_nvfp4_"
            "usage_profile_v8_object_grounded_v11.jsonl"
        ),
    )
    parser.add_argument("--api-url", default=os.environ.get("VLM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--review-mode",
        choices=("direct", "numbered", "named_binary"),
        default="direct",
    )
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--retry-repeat-penalty-step", type=float, default=0.04)
    parser.add_argument(
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Constrain vLLM output to canonical names with JSON-schema decoding",
    )
    parser.add_argument(
        "--constrain-checklist",
        action="store_true",
        help="grammar-constrain named_binary output to exact names and YES/NO values",
    )
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument(
        "--image-transport",
        choices=("base64", "file"),
        default=os.environ.get("VLM_IMAGE_TRANSPORT", "base64"),
    )
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument(
        "--image-id-list",
        default=None,
        help="optional JSON list of exact image IDs to label, preserving its order",
    )
    parser.add_argument(
        "--skip-labeled-from",
        action="append",
        default=[],
        help=(
            "successful-label JSONL whose image IDs are excluded from this shard; "
            "repeat for multiple immutable predecessor shards"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fsync-every", type=int, default=50)
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise ValueError("labeling the sealed test split requires --allow-test")
    if args.run_id is not None and not args.run_id.strip():
        raise ValueError("run-id must not be empty")
    if (
        args.workers < 1
        or args.max_side < 1
        or args.max_output_tokens < 1
        or args.retries < 1
    ):
        raise ValueError("workers, max-side, max-output-tokens, and retries must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if args.repeat_penalty is not None and args.repeat_penalty <= 0:
        raise ValueError("repeat-penalty must be positive")
    if args.retry_repeat_penalty_step < 0:
        raise ValueError("retry-repeat-penalty-step must be nonnegative")
    validate_image_transport(args.api_url, args.image_transport)
    if args.review_mode in {"numbered", "named_binary"} and args.structured_output:
        raise ValueError(f"{args.review_mode} review mode uses a validated text checklist")
    if args.constrain_checklist and args.review_mode != "named_binary":
        raise ValueError("constrain-checklist requires review-mode named_binary")
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
    prompt_builders = {
        "direct": build_prompt,
        "numbered": build_numbered_prompt,
        "named_binary": build_named_binary_prompt,
    }
    system_prompts = {
        "direct": SYSTEM,
        "numbered": NUMBERED_SYSTEM,
        "named_binary": NAMED_BINARY_SYSTEM,
    }
    prompt_versions = {
        "direct": PROMPT_VERSION,
        "numbered": NUMBERED_PROMPT_VERSION,
        "named_binary": NAMED_BINARY_PROMPT_VERSION,
    }
    prompt_builder = prompt_builders[args.review_mode]
    system_prompt = system_prompts[args.review_mode]
    prompt_version = prompt_versions[args.review_mode]
    prompt_template = prompt_builder(concepts, "{object_label}")

    manifest = load_split_manifest(str(ROOT), str(split_manifest_path))
    if args.image_id_list:
        requested_ids = json.loads(resolve(ROOT, args.image_id_list).read_text())
        if not isinstance(requested_ids, list) or not all(
            isinstance(image_id, str) for image_id in requested_ids
        ):
            raise ValueError("--image-id-list must contain a JSON string list")
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("--image-id-list contains duplicates")
        paths = [img_dir / image_id for image_id in requested_ids]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"requested images are missing: {missing[:5]}")
        wrong_split = [
            image_id
            for image_id in requested_ids
            if manifest["classes"][image_class(image_id)] != args.split
        ]
        if wrong_split:
            raise ValueError(
                f"requested images are outside split {args.split}: {wrong_split[:5]}"
            )
    else:
        paths = [
            path
            for path in recursive_image_paths(img_dir)
            if manifest["classes"][image_class(str(path.relative_to(img_dir)))] == args.split
        ]
    skip_sources = []
    skipped_ids: set[str] = set()
    for value in args.skip_labeled_from:
        skip_path = resolve(ROOT, value)
        source_ids = load_completed_image_ids(skip_path, required_status="ok")
        skipped_ids.update(source_ids)
        skip_sources.append(
            {
                "path": display_path(skip_path),
                "sha256": sha256(skip_path),
                "successful_rows": len(source_ids),
            }
        )
    if skipped_ids:
        paths = [
            path
            for path in paths
            if str(path.relative_to(img_dir)) not in skipped_ids
        ]
    if args.n_images is not None:
        random.Random(args.seed).shuffle(paths)
        paths = paths[: args.n_images]

    selected_ids = [str(path.relative_to(img_dir)) for path in paths]
    selection_sha256 = sha256_text("\n".join(selected_ids) + "\n")

    expected_meta = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "prompt": prompt_template,
        "prompt_sha256": sha256_text(system_prompt + "\n" + prompt_template),
        "dictionary": display_path(dictionary_path),
        "dictionary_sha256": sha256(dictionary_path),
        "split_manifest": display_path(split_manifest_path),
        "split_manifest_sha256": sha256(split_manifest_path),
        "image_dir": display_path(img_dir),
        "split": args.split,
        "source_image_count": len(paths),
        "selection_sha256": selection_sha256,
        "run_id": args.run_id,
        "model": args.model,
        "model_revision": os.environ.get("VLM_MODEL_REVISION"),
        "server_profile": os.environ.get("VLM_SERVER_PROFILE"),
        "api_url": args.api_url,
        "temperature": 0,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "retry_repeat_penalty_step": args.retry_repeat_penalty_step,
        "structured_output": args.structured_output,
        "review_mode": args.review_mode,
        "constrain_checklist": args.constrain_checklist,
        "skip_labeled_from": skip_sources,
        "reasoning_effort": "none",
        "max_output_tokens": args.max_output_tokens,
        "client_transform": (
            "none" if args.image_transport == "file" else "resize-max-side-jpeg-q88"
        ),
        "max_image_side": (
            args.max_side if args.image_transport == "base64" else None
        ),
        "image_transport": args.image_transport,
        "workers": args.workers,
        "created_at": utc_now(),
    }
    check_or_write_meta(meta_path, expected_meta, output_path=output)

    done = load_completed_image_ids(output, required_status="ok")
    work = []
    for path in paths:
        image_id = str(path.relative_to(img_dir))
        if image_id not in done:
            object_class = image_class(image_id)
            work.append((path, image_id, object_class.replace("_", " ")))

    print(
        f"model={args.model} endpoint={args.api_url} workers={args.workers} "
        f"split={args.split} images={len(paths)} skipped={len(skipped_ids)} "
        f"done={len(done)} remaining={len(work)}",
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
            prompt=prompt_builder(concepts, object_label),
            concept_names=concept_names,
            max_side=args.max_side,
            max_output_tokens=args.max_output_tokens,
            image_transport=args.image_transport,
            timeout=args.timeout,
            retries=args.retries,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            retry_repeat_penalty_step=args.retry_repeat_penalty_step,
            response_format=structured_response_format(concepts)
            if args.structured_output
            else None,
            structured_outputs=(
                {"regex": named_binary_regex(concepts)}
                if args.constrain_checklist
                else None
            ),
            system_prompt=system_prompt,
            response_mode=(
                args.review_mode
                if args.review_mode in {"numbered", "named_binary"}
                else "json"
            ),
            ordered_concepts=(
                concepts
                if args.review_mode in {"numbered", "named_binary"}
                else None
            ),
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
        f"effective_seconds_per_success={elapsed / max(1, successes):.3f} "
        f"effective_images_per_second={successes / max(elapsed, 1e-9):.3f}",
        flush=True,
    )
    if failures:
        raise SystemExit(f"{failures} rows failed; rerun to retry them")


if __name__ == "__main__":
    main()
