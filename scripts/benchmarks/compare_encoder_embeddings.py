"""Compare a candidate image-text encoder with the frozen project encoder.

The four input arrays must contain the same image-caption pairs in the same row
order.  The encoders may use different embedding dimensions.  This script is
CPU-only: it validates the saved arrays, computes exact paired retrieval ranks
in bounded-memory blocks, and writes a provenance-rich JSON report.

The report is intended to live beside isolated candidate artifacts, never in
``data/``.  It records content hashes for every input so a result cannot be
mistaken for a comparison of different embedding runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from conceptbasis import BACKBONE, BACKBONE_PRETRAINED
from conceptbasis.encoders import sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


def load_array(path: Path, label: str) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {label} array {path}: {error}") from error
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{label} must have a floating dtype, got {array.dtype}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must be nonempty, got shape {array.shape}")
    return array


def load_benchmark_manifests(
    array_paths: dict[str, Path],
    arrays: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Verify benchmark manifests when present, requiring all four together."""
    manifest_paths = {
        label: Path(f"{path}.manifest.json") for label, path in array_paths.items()
    }
    present = {label for label, path in manifest_paths.items() if path.exists()}
    if not present:
        return {}
    if present != set(array_paths):
        missing = sorted(set(array_paths) - present)
        raise ValueError(
            "benchmark manifests must be supplied for all four arrays; missing "
            + ", ".join(missing)
        )

    manifests = {}
    for label, manifest_path in manifest_paths.items():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot load {label} benchmark manifest {manifest_path}: {error}"
            ) from error
        if manifest.get("schema") != "conceptbasis.embedding-benchmark/v1":
            raise ValueError(f"unsupported benchmark manifest schema: {manifest_path}")
        artifact = manifest.get("artifact") or {}
        array = arrays[label]
        expected = {
            "path": str(array_paths[label].resolve()),
            "sha256": sha256_file(array_paths[label]),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        mismatches = {
            key: (artifact.get(key), value)
            for key, value in expected.items()
            if artifact.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{label} benchmark manifest does not match its array: {mismatches}"
            )
        manifests[label] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "encoder": manifest.get("encoder"),
            "inference": manifest.get("inference"),
            "selection": manifest.get("selection"),
        }
    return manifests


def validate_benchmark_manifest_pairing(
    manifests: dict[str, dict],
    pairing_provenance: dict,
    ordered_pair_ids_hash: str,
) -> dict[str, dict]:
    if not manifests:
        return {}
    expected_encoders = {
        "baseline": manifests["baseline_image"]["encoder"],
        "candidate": manifests["candidate_image"]["encoder"],
    }
    for role in ("baseline", "candidate"):
        image_manifest = manifests[f"{role}_image"]
        caption_manifest = manifests[f"{role}_caption"]
        if image_manifest["encoder"] != caption_manifest["encoder"]:
            raise ValueError(f"{role} image/caption benchmark encoders differ")
        image_inference = image_manifest.get("inference") or {}
        caption_inference = caption_manifest.get("inference") or {}
        if image_inference.get("precision") != caption_inference.get("precision"):
            raise ValueError(f"{role} image/caption benchmark precisions differ")

    sidecars = pairing_provenance.get("sidecars") or {}
    for label, manifest in manifests.items():
        selection = manifest.get("selection") or {}
        if selection.get("ordered_ids_sha256") != ordered_pair_ids_hash:
            raise ValueError(f"{label} benchmark manifest has a different row identity")
        if sidecars and selection.get("ordered_ids_sidecar_sha256") != sidecars[label].get(
            "sha256"
        ):
            raise ValueError(f"{label} benchmark manifest sidecar checksum differs")
    return expected_encoders


def check_shapes(arrays: dict[str, np.ndarray]) -> int:
    row_counts = {label: int(array.shape[0]) for label, array in arrays.items()}
    if len(set(row_counts.values())) != 1:
        details = ", ".join(f"{label}={count}" for label, count in row_counts.items())
        raise ValueError(f"embedding row counts differ: {details}")
    for encoder in ("baseline", "candidate"):
        image_dim = arrays[f"{encoder}_image"].shape[1]
        caption_dim = arrays[f"{encoder}_caption"].shape[1]
        if image_dim != caption_dim:
            raise ValueError(
                f"{encoder} image/caption dimensions differ: {image_dim} != {caption_dim}"
            )
    return next(iter(row_counts.values()))


def ordered_pair_ids(
    source_ids: Sequence[str],
    row_count: int,
    array_order: str,
    selection_seed: int,
) -> list[str]:
    if len(source_ids) < row_count:
        raise ValueError(
            f"image ID manifest has {len(source_ids)} rows but arrays have {row_count}"
        )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("image ID manifest contains duplicate IDs")
    ids = list(source_ids)
    if array_order == "seeded-shuffle":
        random.Random(selection_seed).shuffle(ids)
    elif array_order != "canonical":
        raise ValueError(f"unsupported array order: {array_order}")
    return ids[:row_count]


def load_ordered_ids_sidecar(path: Path, label: str, row_count: int) -> tuple[list[str], dict]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label} ordered-ID sidecar {path}: {error}") from error
    if payload.get("schema") != "conceptbasis.ordered-embedding-ids/v1":
        raise ValueError(f"unsupported ordered-ID sidecar schema: {path}")
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise ValueError(f"{label} ordered-ID sidecar must contain a string list")
    if len(ids) != row_count:
        raise ValueError(
            f"{label} ordered-ID sidecar has {len(ids)} rows but array has {row_count}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} ordered-ID sidecar contains duplicate IDs")
    ordered_hash = sha256_json(ids)
    if payload.get("ordered_ids_sha256") != ordered_hash:
        raise ValueError(f"{label} ordered-ID sidecar hash does not match its IDs")
    return ids, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "ordered_ids_sha256": ordered_hash,
        "order": payload.get("order"),
        "selection_seed": payload.get("selection_seed"),
    }


def resolve_pair_ids(
    args: argparse.Namespace,
    array_paths: dict[str, Path],
    source_ids: Sequence[str],
    row_count: int,
) -> tuple[list[str], dict]:
    """Prefer exact saved row IDs and require every array to agree byte-for-byte."""

    sidecar_paths: dict[str, Path | None] = {}
    for label, array_path in array_paths.items():
        explicit = getattr(args, f"{label}_ids")
        automatic = Path(f"{array_path}.ids.json")
        sidecar_paths[label] = explicit.resolve() if explicit else (
            automatic if automatic.exists() else None
        )
    if any(path is not None for path in sidecar_paths.values()):
        missing = [label for label, path in sidecar_paths.items() if path is None]
        if missing:
            raise ValueError(
                "ordered-ID sidecars must be supplied for all four arrays; missing "
                + ", ".join(missing)
            )
        loaded = {
            label: load_ordered_ids_sidecar(path, label, row_count)
            for label, path in sidecar_paths.items()
            if path is not None
        }
        reference = loaded["baseline_image"][0]
        mismatched = [label for label, (ids, _) in loaded.items() if ids != reference]
        if mismatched:
            raise ValueError(
                "embedding arrays have different explicit row orders: "
                + ", ".join(mismatched)
            )
        return reference, {
            "mode": "explicit-sidecars",
            "sidecars": {label: metadata for label, (_, metadata) in loaded.items()},
        }

    return ordered_pair_ids(
        source_ids, row_count, args.array_order, args.selection_seed
    ), {
        "mode": "legacy-reconstruction",
        "array_order": args.array_order,
        "selection_seed": (
            args.selection_seed if args.array_order == "seeded-shuffle" else None
        ),
    }


def deterministic_subset(ids: Sequence[str], limit: int, seed: int) -> np.ndarray:
    """Select rows by stable SHA-256 priority, returning sorted row indexes."""
    if limit <= 0 or limit >= len(ids):
        return np.arange(len(ids), dtype=np.int64)
    priorities = []
    for index, image_id in enumerate(ids):
        material = f"{seed}\0{image_id}".encode("utf-8")
        priorities.append((hashlib.sha256(material).digest(), index))
    selected = sorted(index for _, index in sorted(priorities)[:limit])
    return np.asarray(selected, dtype=np.int64)


def embedding_checks(array: np.ndarray, norm_atol: float, chunk_size: int = 8192) -> dict:
    norm_parts = []
    finite = True
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size], dtype=np.float32)
        if not np.isfinite(chunk).all():
            finite = False
        norm_parts.append(np.linalg.norm(chunk, axis=1))
    norms = np.concatenate(norm_parts)
    deviations = np.abs(norms - 1.0)
    return {
        "rows": int(array.shape[0]),
        "dimensions": int(array.shape[1]),
        "dtype": str(array.dtype),
        "all_finite": finite,
        "zero_norm_rows": int(np.count_nonzero(norms == 0)),
        "norm": {
            "minimum": float(norms.min()),
            "mean": float(norms.mean()),
            "maximum": float(norms.max()),
            "max_abs_error_from_one": float(deviations.max()),
            "fraction_within_atol": float(np.mean(deviations <= norm_atol)),
            "atol": float(norm_atol),
        },
    }


def normalized_rows(array: np.ndarray, indexes: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    output = np.empty((len(indexes), array.shape[1]), dtype=np.float32)
    for start in range(0, len(indexes), chunk_size):
        selected = indexes[start : start + chunk_size]
        chunk = np.asarray(array[selected], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        if not np.isfinite(chunk).all() or not np.isfinite(norms).all():
            raise ValueError("retrieval embeddings contain non-finite values")
        if np.any(norms == 0):
            raise ValueError("retrieval embeddings contain zero-norm rows")
        output[start : start + len(chunk)] = chunk / norms
    return output


def retrieval_direction(
    queries: np.ndarray,
    gallery: np.ndarray,
    ks: Iterable[int],
    block_size: int,
) -> dict:
    """Return exact 1-based paired ranks with deterministic index tie-breaking."""
    if queries.shape != gallery.shape:
        raise ValueError(f"retrieval shapes differ: {queries.shape} != {gallery.shape}")
    count = len(queries)
    gallery_indexes = np.arange(count, dtype=np.int64)
    ranks = np.empty(count, dtype=np.int64)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        scores = queries[start:stop] @ gallery.T
        local = np.arange(stop - start)
        targets = np.arange(start, stop)
        target_scores = scores[local, targets]
        ahead = scores > target_scores[:, None]
        tied_before = (scores == target_scores[:, None]) & (
            gallery_indexes[None, :] < targets[:, None]
        )
        ranks[start:stop] = 1 + ahead.sum(axis=1) + tied_before.sum(axis=1)
    recalls = {str(k): float(np.mean(ranks <= k)) for k in ks}
    return {
        "recall_at": recalls,
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "maximum_rank": int(ranks.max()),
    }


def paired_retrieval_metrics(
    image: np.ndarray,
    caption: np.ndarray,
    indexes: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
    block_size: int = 256,
) -> dict:
    image_normalized = normalized_rows(image, indexes)
    caption_normalized = normalized_rows(caption, indexes)
    paired_cosine = np.sum(image_normalized * caption_normalized, axis=1)
    return {
        "count": int(len(indexes)),
        "paired_cosine": {
            "mean": float(paired_cosine.mean()),
            "standard_deviation": float(paired_cosine.std()),
            "p05": float(np.quantile(paired_cosine, 0.05)),
            "median": float(np.median(paired_cosine)),
            "p95": float(np.quantile(paired_cosine, 0.95)),
        },
        "image_to_caption": retrieval_direction(
            image_normalized, caption_normalized, ks, block_size
        ),
        "caption_to_image": retrieval_direction(
            caption_normalized, image_normalized, ks, block_size
        ),
    }


def quality_gates(
    checks: dict[str, dict],
    baseline_metrics: dict,
    candidate_metrics: dict,
    norm_atol: float,
    max_recall_drop_pp: float,
) -> dict:
    normalization_results = {}
    for label, result in checks.items():
        passed = (
            result["all_finite"]
            and result["zero_norm_rows"] == 0
            and result["norm"]["max_abs_error_from_one"] <= norm_atol
        )
        normalization_results[label] = {
            "passed": bool(passed),
            "max_abs_error_from_one": result["norm"]["max_abs_error_from_one"],
        }

    retrieval_results = {}
    for direction in ("image_to_caption", "caption_to_image"):
        baseline_recalls = baseline_metrics[direction]["recall_at"]
        candidate_recalls = candidate_metrics[direction]["recall_at"]
        for k, baseline_value in baseline_recalls.items():
            candidate_value = candidate_recalls[k]
            delta_pp = (candidate_value - baseline_value) * 100.0
            label = f"{direction}.recall_at_{k}"
            retrieval_results[label] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta_percentage_points": float(delta_pp),
                "passed": bool(delta_pp >= -max_recall_drop_pp),
            }

    normalization_passed = all(row["passed"] for row in normalization_results.values())
    retrieval_passed = all(row["passed"] for row in retrieval_results.values())
    minimum_delta_pp = min(
        row["delta_percentage_points"] for row in retrieval_results.values()
    )
    return {
        "passed": bool(normalization_passed and retrieval_passed),
        "normalization": {
            "passed": bool(normalization_passed),
            "max_abs_norm_error": float(norm_atol),
            "checks": normalization_results,
        },
        "paired_retrieval_noninferiority": {
            "passed": bool(retrieval_passed),
            "maximum_recall_drop_percentage_points": float(max_recall_drop_pp),
            "minimum_observed_delta_percentage_points": float(minimum_delta_pp),
            "checks": retrieval_results,
        },
    }


def artifact_manifest(path: Path, array: np.ndarray) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
    }


def repository_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "worktree_dirty": None if status is None else bool(status),
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-image", type=Path, required=True)
    parser.add_argument("--baseline-caption", type=Path, required=True)
    parser.add_argument("--candidate-image", type=Path, required=True)
    parser.add_argument("--candidate-caption", type=Path, required=True)
    parser.add_argument(
        "--baseline-image-ids",
        type=Path,
        help="ordered-ID sidecar (auto-detected as <array>.ids.json)",
    )
    parser.add_argument("--baseline-caption-ids", type=Path)
    parser.add_argument("--candidate-image-ids", type=Path)
    parser.add_argument("--candidate-caption-ids", type=Path)
    parser.add_argument("--image-ids", type=Path, default=ROOT / "data/image_ids.json")
    parser.add_argument("--captions", type=Path, default=ROOT / "data/captions.jsonl")
    parser.add_argument(
        "--array-order",
        choices=("canonical", "seeded-shuffle"),
        default="seeded-shuffle",
        help="row order used when the four arrays were encoded",
    )
    parser.add_argument("--selection-seed", type=int, default=20260718)
    parser.add_argument("--baseline-model", default=BACKBONE)
    parser.add_argument("--baseline-pretrained", default=BACKBONE_PRETRAINED)
    parser.add_argument("--baseline-revision")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-pretrained", required=True)
    parser.add_argument("--candidate-revision")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--retrieval-limit", type=int, default=4096)
    parser.add_argument("--retrieval-seed", type=int, default=20260718)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--norm-atol", type=float, default=5e-4)
    parser.add_argument("--max-recall-drop-pp", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="exit with status 2 after writing a report whose quality gates fail",
    )
    args = parser.parse_args(argv)
    if args.retrieval_limit < 0:
        parser.error("--retrieval-limit must be nonnegative (0 means all rows)")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if not args.recall_k or any(k < 1 for k in args.recall_k):
        parser.error("--recall-k values must be positive")
    if args.norm_atol < 0 or args.max_recall_drop_pp < 0:
        parser.error("quality thresholds must be nonnegative")
    return args


def run(args: argparse.Namespace) -> dict:
    paths = {
        "baseline_image": args.baseline_image.resolve(),
        "baseline_caption": args.baseline_caption.resolve(),
        "candidate_image": args.candidate_image.resolve(),
        "candidate_caption": args.candidate_caption.resolve(),
    }
    arrays = {label: load_array(path, label) for label, path in paths.items()}
    row_count = check_shapes(arrays)
    benchmark_manifests = load_benchmark_manifests(paths, arrays)

    source_ids = json.loads(args.image_ids.read_text())
    if not isinstance(source_ids, list) or not all(isinstance(value, str) for value in source_ids):
        raise ValueError("image ID manifest must be a JSON list of strings")
    pair_ids, pairing_provenance = resolve_pair_ids(
        args, paths, source_ids, row_count
    )
    ordered_pair_ids_hash = sha256_json(pair_ids)
    manifest_encoders = validate_benchmark_manifest_pairing(
        benchmark_manifests,
        pairing_provenance,
        ordered_pair_ids_hash,
    )
    if manifest_encoders:
        declarations = {
            "baseline": {
                "model": args.baseline_model,
                "pretrained": args.baseline_pretrained,
                "revision": args.baseline_revision,
            },
            "candidate": {
                "model": args.candidate_model,
                "pretrained": args.candidate_pretrained,
                "revision": args.candidate_revision,
            },
        }
        for role, declared in declarations.items():
            recorded = manifest_encoders[role] or {}
            mismatches = {
                key: (recorded.get(key), value)
                for key, value in declared.items()
                if recorded.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"{role} model declaration differs from benchmark manifest: "
                    f"{mismatches}"
                )
            recorded_precision = (
                benchmark_manifests[f"{role}_image"].get("inference") or {}
            ).get("precision")
            if recorded_precision != args.precision:
                raise ValueError(
                    f"{role} precision declaration differs from benchmark manifest: "
                    f"{recorded_precision!r} != {args.precision!r}"
                )
    retrieval_indexes = deterministic_subset(pair_ids, args.retrieval_limit, args.retrieval_seed)
    retrieval_ids = [pair_ids[index] for index in retrieval_indexes]

    checks = {
        label: embedding_checks(array, args.norm_atol) for label, array in arrays.items()
    }
    baseline_metrics = paired_retrieval_metrics(
        arrays["baseline_image"],
        arrays["baseline_caption"],
        retrieval_indexes,
        args.recall_k,
        args.block_size,
    )
    candidate_metrics = paired_retrieval_metrics(
        arrays["candidate_image"],
        arrays["candidate_caption"],
        retrieval_indexes,
        args.recall_k,
        args.block_size,
    )
    gates = quality_gates(
        checks,
        baseline_metrics,
        candidate_metrics,
        args.norm_atol,
        args.max_recall_drop_pp,
    )

    return {
        "schema": "conceptbasis.encoder-comparison",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "baseline": {
                "model": args.baseline_model,
                "pretrained": args.baseline_pretrained,
                "revision": args.baseline_revision,
                "precision": args.precision,
            },
            "candidate": {
                "model": args.candidate_model,
                "pretrained": args.candidate_pretrained,
                "revision": args.candidate_revision,
                "precision": args.precision,
            },
        },
        "inputs": {
            label: artifact_manifest(paths[label], arrays[label]) for label in paths
        },
        "benchmark_manifests": benchmark_manifests,
        "pairing": {
            "image_ids": {
                "path": str(args.image_ids.resolve()),
                "sha256": sha256_file(args.image_ids),
                "source_count": len(source_ids),
            },
            "captions": {
                "path": str(args.captions.resolve()),
                "sha256": sha256_file(args.captions),
            },
            "array_row_count": row_count,
            "array_order": (
                args.array_order
                if pairing_provenance["mode"] == "legacy-reconstruction"
                else "explicit-sidecars"
            ),
            "selection_seed": (
                args.selection_seed
                if pairing_provenance["mode"] == "legacy-reconstruction"
                and args.array_order == "seeded-shuffle"
                else None
            ),
            "row_identity": pairing_provenance,
            "ordered_pair_ids_sha256": ordered_pair_ids_hash,
        },
        "retrieval_evaluation": {
            "selection": "sha256-priority" if len(retrieval_indexes) < row_count else "all-rows",
            "limit": args.retrieval_limit,
            "seed": args.retrieval_seed,
            "count": len(retrieval_indexes),
            "pair_ids_sha256": sha256_json(retrieval_ids),
            "recall_k": sorted(set(args.recall_k)),
            "block_size": args.block_size,
        },
        "embedding_checks": checks,
        "retrieval_metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        },
        "quality_gates": gates,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            **repository_state(),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote={args.output} rows={report['pairing']['array_row_count']} "
        f"retrieval_rows={report['retrieval_evaluation']['count']} "
        f"quality_gates_passed={report['quality_gates']['passed']}"
    )
    if args.fail_on_gate and not report["quality_gates"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
