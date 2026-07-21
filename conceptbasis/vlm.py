"""Shared utilities for reproducible OpenAI-compatible VLM data jobs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
from urllib.parse import urlsplit

import requests
from PIL import Image


LOCAL_ENDPOINT_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
GEMMA_MODEL_ID = "nvidia/Gemma-4-26B-A4B-NVFP4"
LOCAL_VLLM_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_THREAD_STATE = threading.local()


def is_local_endpoint(api_url: str) -> bool:
    """Return whether an API URL resolves explicitly to the local machine."""
    return urlsplit(api_url).hostname in LOCAL_ENDPOINT_HOSTS


def validate_image_transport(api_url: str, transport: str) -> None:
    """Reject server-side file references that a remote endpoint cannot read."""
    if transport == "file" and not is_local_endpoint(api_url):
        raise ValueError(
            "--image-transport=file requires a localhost, 127.0.0.1, or ::1 endpoint"
        )


def api_key_for(api_url: str) -> str:
    """Resolve local/remote authentication without silently calling a remote."""
    import os

    key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if is_local_endpoint(api_url):
        return "local"
    raise ValueError("set VLM_API_KEY for a remote endpoint")


def recursive_image_paths(image_dir: str | Path) -> list[Path]:
    """Return supported images in stable relative-path order."""
    root = Path(image_dir)
    return sorted(
        (path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def load_completed_image_ids(
    output: str | Path,
    *,
    required_field: str | None = None,
    required_status: str | None = None,
) -> set[str]:
    """Load unique successful IDs from an append-only JSONL artifact."""
    path = Path(output)
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if required_status is not None and row.get("status") != required_status:
                continue
            if required_field is not None and not row.get(required_field):
                continue
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"missing image_id at {path}:{line_number}")
            if image_id in done:
                raise ValueError(f"duplicate completed image_id in {path}: {image_id}")
            done.add(image_id)
    return done


def image_reference(
    path: str | Path,
    *,
    transport: str,
    max_side: int,
    jpeg_quality: int = 88,
) -> str:
    """Return a local file URI or a bounded base64 JPEG data URI."""
    if max_side < 1 or not 1 <= jpeg_quality <= 100:
        raise ValueError("max_side and jpeg_quality must be positive and valid")
    image_path = Path(path)
    if transport == "file":
        return image_path.resolve().as_uri()
    if transport != "base64":
        raise ValueError(f"unsupported image transport: {transport}")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def session() -> requests.Session:
    """Reuse one HTTP session per worker thread."""
    if not hasattr(_THREAD_STATE, "session"):
        _THREAD_STATE.session = requests.Session()
    return _THREAD_STATE.session


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ordered_ids_sha256(values: Sequence[str]) -> str:
    """Hash an ordered selection with an unambiguous trailing delimiter."""
    return sha256_text("\n".join(values) + "\n")


def check_or_write_metadata(
    metadata_path: Path,
    expected: Mapping[str, Any],
    *,
    output_path: Path | None = None,
    compatibility_keys: Collection[str] | None = None,
    legacy_defaults: Mapping[str, Any] | None = None,
) -> None:
    """Guard append-only outputs against incompatible resumptions."""
    defaults = legacy_defaults or {}
    keys = compatibility_keys or expected.keys()
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        mismatches = {
            key: (
                existing.get(key, defaults.get(key)),
                expected.get(key, defaults.get(key)),
            )
            for key in keys
            if existing.get(key, defaults.get(key))
            != expected.get(key, defaults.get(key))
        }
        if mismatches:
            raise ValueError(
                f"refusing incompatible resume; metadata differs: {mismatches}"
            )
        return
    if output_path is not None and output_path.exists() and output_path.stat().st_size:
        raise ValueError(
            f"refusing to adopt existing output without metadata sidecar: {output_path}"
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(dict(expected), indent=2) + "\n")


def check_or_write_run_metadata(output_path: Path, expected: Mapping[str, Any]) -> Path:
    """Write the conventional ``<output>.meta.json`` resume guard."""
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    check_or_write_metadata(metadata_path, expected, output_path=output_path)
    return metadata_path
