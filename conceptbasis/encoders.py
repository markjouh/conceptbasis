"""Encoder presets and provenance-safe cache helpers.

The historical PE cache remains available as ``project`` for compatibility.
The selected SigLIP2 Giant release uses an isolated, checksummed namespace so
it cannot be confused with those legacy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from conceptbasis import BACKBONE, BACKBONE_PRETRAINED


MANIFEST_SCHEMA = "conceptbasis.encoder-cache/v1"
MANIFEST_NAME = "encoder_manifest.json"
ENCODER_ARTIFACT_NAMES = (
    "image_embeddings.npy",
    "image_embeddings_cc0.npy",
    "image_ids.json",
    "concept_directions_initial.npy",
    "labels.parquet",
)


@dataclass(frozen=True)
class EncoderSpec:
    """A requested OpenCLIP model release.

    ``revision`` pins the Hugging Face source when the OpenCLIP pretrained tag
    resolves to a Hub repository.  It is part of cache identity even when it is
    omitted, making a floating release visible in provenance.
    """

    name: str
    model: str
    pretrained: str
    revision: str | None = None

    @property
    def cache_key(self) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        revision = self.revision[:7] if self.revision else "floating"
        return f"{base}@{revision}"

    def as_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "model": self.model,
            "pretrained": self.pretrained,
            "revision": self.revision,
        }


PROJECT_ENCODER = EncoderSpec(
    name="pe-core-bigg-p14-448",
    model=BACKBONE,
    pretrained=BACKBONE_PRETRAINED,
)

SIGLIP2_GIANT = EncoderSpec(
    name="siglip2-gopt-p16-384",
    model="ViT-gopt-16-SigLIP2-384",
    pretrained="webli",
    revision="ad3410bee2c3373be5ed01e7c4e7fcd2bf95a183",
)

ENCODER_PRESETS: Mapping[str, EncoderSpec] = {
    "project": PROJECT_ENCODER,
    "siglip2-giant": SIGLIP2_GIANT,
}


class EncoderCacheMismatch(ValueError):
    """Raised rather than reusing an encoder cache with different provenance."""


def select_encoder(
    preset: str,
    *,
    model: str | None = None,
    pretrained: str | None = None,
    revision: str | None = None,
) -> EncoderSpec:
    """Resolve a preset or an explicit model/pretrained pair.

    Requiring both explicit values prevents an accidental combination such as
    a SigLIP architecture with the project's ``meta`` pretrained tag.
    """

    if preset not in ENCODER_PRESETS:
        raise ValueError(f"unknown encoder preset: {preset}")
    if (model is None) != (pretrained is None):
        raise ValueError("--model and --pretrained must be supplied together")
    base = ENCODER_PRESETS[preset]
    if model is None:
        return EncoderSpec(
            name=base.name,
            model=base.model,
            pretrained=base.pretrained,
            revision=revision or base.revision,
        )
    return EncoderSpec(
        name=model,
        model=model,
        pretrained=pretrained,
        revision=revision,
    )


def encoder_output_dir(
    root: Path,
    spec: EncoderSpec,
    requested: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a cache directory while protecting the production ``data`` dir."""

    root = root.resolve()
    production = (root / "data").resolve()
    if requested is None:
        output = (
            production
            if spec == PROJECT_ENCODER
            else root / "outputs" / "encoder_candidates" / spec.cache_key
        )
    else:
        candidate = Path(requested)
        output = candidate if candidate.is_absolute() else root / candidate
    output = output.resolve()
    if spec != PROJECT_ENCODER and (
        output == production or production in output.parents
    ):
        raise ValueError(
            "candidate encoders cannot write to data/; choose an isolated --output-dir"
        )
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible containers for comparison."""
    return json.loads(json.dumps(value, sort_keys=True))


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array)
    temporary.replace(path)


def release_visual_tower(model: Any, device: str) -> None:
    """Release an unused OpenCLIP visual tower before text-only inference."""
    if not hasattr(model, "visual"):
        return
    model.visual = None
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def cache_identity(
    spec: EncoderSpec,
    *,
    precision: str,
    image_ids: Sequence[str],
    cc0_image_ids: Sequence[str],
    dictionary_path: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    """Return the fields that must match before cached arrays may be reused."""

    return {
        "encoder": spec.as_dict(),
        "precision": precision,
        "inputs": {
            "full_images": {
                "count": len(image_ids),
                "ordered_ids_sha256": sha256_json(list(image_ids)),
            },
            "cc0_images": {
                "count": len(cc0_image_ids),
                "ordered_ids_sha256": sha256_json(list(cc0_image_ids)),
            },
            "dictionary_sha256": sha256_file(dictionary_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
        },
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    return record


def validate_cache_manifest(
    output_dir: Path,
    identity: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
    allow_legacy: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any] | None:
    """Validate candidate cache identity and any existing declared artifacts."""

    manifest_path = output_dir / MANIFEST_NAME
    existing_artifacts = [
        output_dir / name
        for name in ENCODER_ARTIFACT_NAMES
        if (output_dir / name).exists()
    ]
    if not manifest_path.exists():
        if existing_artifacts and not allow_legacy:
            names = ", ".join(path.name for path in existing_artifacts)
            raise EncoderCacheMismatch(
                f"refusing unmanifested encoder cache in {output_dir}: {names}"
            )
        return None

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EncoderCacheMismatch(f"unsupported encoder manifest: {manifest_path}")
    if manifest.get("identity") != dict(identity):
        raise EncoderCacheMismatch(
            f"encoder cache identity does not match requested run: {manifest_path}"
        )
    if source is not None and manifest.get("source") != canonical_json(dict(source)):
        raise EncoderCacheMismatch(
            f"encoder cache runtime/preprocess source does not match: {manifest_path}"
        )
    if verify_hashes:
        for name, expected in manifest.get("artifacts", {}).items():
            path = output_dir / name
            # A missing artifact can be regenerated.  An existing but changed
            # artifact must never be consumed as if it came from this manifest.
            if path.exists() and sha256_file(path) != expected.get("sha256"):
                raise EncoderCacheMismatch(f"encoder artifact checksum mismatch: {path}")
    return manifest


def write_cache_manifest(
    output_dir: Path,
    identity: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> Path:
    """Atomically write provenance for every completed encoder artifact."""

    artifacts = {
        name: _artifact_record(output_dir / name)
        for name in ENCODER_ARTIFACT_NAMES
        if (output_dir / name).exists()
    }
    payload = {
        "schema": MANIFEST_SCHEMA,
        "identity": dict(identity),
        "source": canonical_json(dict(source)),
        "artifacts": artifacts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    write_json_atomic(manifest_path, payload)
    return manifest_path


def load_open_clip_encoder(
    spec: EncoderSpec,
    *,
    device: str,
    precision: str,
    cache_dir: str | None = None,
):
    """Load an OpenCLIP encoder, pinning Hub weights/tokenizer when requested.

    Imports are intentionally local so manifest/path tooling remains usable in
    CPU-only environments without importing the model stack.
    """

    import importlib.metadata
    import open_clip

    pretrained_value = spec.pretrained
    transform_kwargs: dict[str, Any] = {}
    model_cfg = open_clip.get_model_config(spec.model) or {}
    pretrained_cfg = open_clip.get_pretrained_cfg(spec.model, spec.pretrained) or {}
    vision_cfg = model_cfg.get("vision_cfg", {})
    text_cfg = model_cfg.get("text_cfg", {})
    source: dict[str, Any] = {
        "open_clip_version": importlib.metadata.version("open_clip_torch"),
        "pretrained_tag": spec.pretrained,
        "embedding_dimension": model_cfg.get("embed_dim"),
        "image_size": vision_cfg.get("image_size"),
        "patch_size": vision_cfg.get("patch_size"),
        "context_length": text_cfg.get("context_length"),
        "preprocess": {
            "mean": pretrained_cfg.get("mean"),
            "std": pretrained_cfg.get("std"),
            "interpolation": pretrained_cfg.get("interpolation"),
            "resize_mode": pretrained_cfg.get("resize_mode"),
        },
    }
    if spec.revision:
        cfg = pretrained_cfg
        hub_entry = cfg.get("hf_hub", "")
        if not hub_entry:
            raise ValueError(
                f"encoder revision pin requires a Hugging Face pretrained source: {spec.model}"
            )
        if hub_entry.endswith("/"):
            repo_id, filename = hub_entry[:-1], None
        else:
            repo_id, filename = os.path.split(hub_entry)
        from open_clip.pretrained import download_pretrained_from_hf

        if filename:
            pretrained_value = download_pretrained_from_hf(
                repo_id,
                filename=filename,
                revision=spec.revision,
                cache_dir=cache_dir,
            )
        else:
            # Current timm OpenCLIP exports use the safe
            # ``open_clip_model.safetensors`` filename while OpenCLIP's generic
            # fallback first looks for a differently named safetensors file and
            # then a pickle .bin. Prefer the repository's safe artifact.
            try:
                pretrained_value = download_pretrained_from_hf(
                    repo_id,
                    filename="open_clip_model.safetensors",
                    revision=spec.revision,
                    cache_dir=cache_dir,
                )
            except FileNotFoundError:
                pretrained_value = download_pretrained_from_hf(
                    repo_id,
                    revision=spec.revision,
                    cache_dir=cache_dir,
                )
        transform_kwargs = {
            "image_mean": cfg.get("mean"),
            "image_std": cfg.get("std"),
            "image_interpolation": cfg.get("interpolation"),
            "image_resize_mode": cfg.get("resize_mode"),
        }
        transform_kwargs = {
            key: value for key, value in transform_kwargs.items() if value is not None
        }
        source.update(
            hf_repo=repo_id,
            hf_revision=spec.revision,
            checkpoint_filename=Path(pretrained_value).name,
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.model,
        pretrained=pretrained_value,
        device=device,
        precision=precision,
        cache_dir=cache_dir,
        **transform_kwargs,
    )
    tokenizer_kwargs = {"cache_dir": cache_dir}
    # Only Hugging Face tokenizers accept a Hub revision.  PE uses OpenCLIP's
    # local SimpleTokenizer, while SigLIP2 declares hf_tokenizer_name.
    if spec.revision and text_cfg.get("hf_tokenizer_name"):
        tokenizer_kwargs["revision"] = spec.revision
    tokenizer = open_clip.get_tokenizer(spec.model, **tokenizer_kwargs)
    model.eval()
    return model, preprocess, tokenizer, source
