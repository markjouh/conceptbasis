"""Isolated throughput benchmark for a pinned OpenCLIP-compatible encoder.

This never writes production caches. Use an output path under /tmp and compare
the resulting arrays across batch/prefetch settings before adopting a change.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from conceptbasis import BACKBONE, BACKBONE_PRETRAINED
from conceptbasis.embedding import image_batches, load_image
from conceptbasis.encoders import (
    EncoderSpec,
    load_open_clip_encoder,
    save_npy_atomic,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[2]


def image_paths(image_dir: Path, n_items: int, seed: int) -> list[Path]:
    paths = [
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    # Sort the serialized IDs, not Path objects.  Path compares components, so
    # ``ice/...`` sorts before ``ice-cream_cone/...`` while their POSIX strings
    # sort in the opposite order.  Caption inputs and manifests use strings.
    paths.sort(key=lambda path: path.relative_to(image_dir).as_posix())
    random.Random(seed).shuffle(paths)
    return paths[:n_items] if n_items else paths


def autocast_context(device: str, precision: str):
    if device != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def image_input_dtype(args: argparse.Namespace, device: str):
    if device == "cuda" and args.input_fp16:
        return torch.float16
    return None


def output_sidecars(output: Path) -> tuple[Path, Path]:
    return (
        output.with_suffix(output.suffix + ".ids.json"),
        output.with_suffix(output.suffix + ".manifest.json"),
    )


def write_benchmark_sidecars(
    output_path: Path,
    output: np.ndarray,
    ordered_ids: list[str],
    *,
    kind: str,
    seed: int,
    requested_count: int,
    encoder: EncoderSpec,
    encoder_source: dict,
    inference: dict,
    selection_sources: dict,
    metrics: dict,
) -> tuple[Path, Path]:
    """Write explicit row order and provenance for a benchmark array."""

    ids_path, manifest_path = output_sidecars(output_path)
    ordered_ids_sha256 = sha256_json(ordered_ids)
    ids_payload = {
        "schema": "conceptbasis.ordered-embedding-ids/v1",
        "kind": kind,
        "order": "seeded-shuffle",
        "selection_seed": seed,
        "ordered_ids_sha256": ordered_ids_sha256,
        "ids": ordered_ids,
    }
    write_json_atomic(ids_path, ids_payload)
    manifest = {
        "schema": "conceptbasis.embedding-benchmark/v1",
        "encoder": encoder.as_dict(),
        "encoder_source": encoder_source,
        "inference": inference,
        "selection": {
            "kind": kind,
            "order": "seeded-shuffle",
            "seed": seed,
            "requested_count": requested_count,
            "actual_count": len(ordered_ids),
            "ordered_ids_sha256": ordered_ids_sha256,
            "ordered_ids_sidecar": str(ids_path.resolve()),
            "ordered_ids_sidecar_sha256": sha256_file(ids_path),
            "sources": selection_sources,
        },
        "artifact": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
        },
        "metrics": metrics,
    }
    write_json_atomic(manifest_path, manifest)
    return ids_path, manifest_path


@torch.inference_mode()
def encode_images(
    args: argparse.Namespace,
    model,
    preprocess,
    device: str,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    image_dir = (ROOT / args.image_dir).resolve()
    paths = image_paths(image_dir, args.n_items, args.seed)
    if not paths:
        raise ValueError("no benchmark images found")
    ordered_ids = [path.relative_to(image_dir).as_posix() for path in paths]

    # Warm the exact inference shape without including kernel startup in the rate.
    warm_paths = paths[: min(args.batch_size, len(paths))]
    warm = torch.stack([load_image(path, preprocess) for path in warm_paths]).to(
        device,
        dtype=image_input_dtype(args, device),
    )
    with autocast_context(device, args.precision):
        model.encode_image(warm)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    result = []
    started = time.monotonic()
    for host_batch in image_batches(
        paths,
        preprocess,
        args.batch_size,
        args.preprocess_workers,
        args.prefetch_batches,
    ):
        if args.pin_memory and device == "cuda":
            host_batch = host_batch.pin_memory()
        batch = host_batch.to(
            device,
            dtype=image_input_dtype(args, device),
            non_blocking=args.pin_memory and device == "cuda",
        )
        with autocast_context(device, args.precision):
            encoded = F.normalize(model.encode_image(batch), dim=-1)
        result.append(encoded.float().cpu().numpy().astype(np.float32))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output = np.concatenate(result)
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    items_per_second = len(output) / elapsed
    print(
        f"kind=image items={len(output)} batch_size={args.batch_size} "
        f"preprocess_workers={args.preprocess_workers} pin_memory={args.pin_memory} "
        f"input_fp16={args.input_fp16} "
        f"encode_seconds={elapsed:.3f} items_per_second={items_per_second:.3f} "
        f"peak_allocated_gib={peak_allocated_gib:.3f}"
    )
    return output, ordered_ids, {
        "encode_seconds": elapsed,
        "items_per_second": items_per_second,
        "peak_allocated_gib": peak_allocated_gib,
    }


@torch.inference_mode()
def encode_captions(
    args: argparse.Namespace,
    model,
    tokenizer,
    device: str,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    # Canonicalize before applying the seeded shuffle.  Image benchmarks start
    # from sorted paths, so this keeps image and caption arrays in byte-for-byte
    # identical row order even when image_ids.json was written in another order.
    image_ids = sorted(json.loads((ROOT / args.image_ids).read_text()))
    captions = {}
    for line in (ROOT / args.captions).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("caption"):
                captions[row["image_id"]] = row["caption"]
    pairs = [(image_id, captions.get(image_id, "an object")) for image_id in image_ids]
    random.Random(args.seed).shuffle(pairs)
    if args.n_items:
        pairs = pairs[: args.n_items]
    ordered_ids = [image_id for image_id, _ in pairs]
    texts = [text for _, text in pairs]
    if not texts:
        raise ValueError("no benchmark captions found")

    warm = tokenizer(texts[: min(args.batch_size, len(texts))]).to(device)
    with autocast_context(device, args.precision):
        model.encode_text(warm)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    result = []
    started = time.monotonic()
    for start in range(0, len(texts), args.batch_size):
        tokens = tokenizer(texts[start : start + args.batch_size]).to(device)
        with autocast_context(device, args.precision):
            encoded = F.normalize(model.encode_text(tokens), dim=-1)
        result.append(encoded.float().cpu().numpy().astype(np.float32))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output = np.concatenate(result)
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    items_per_second = len(output) / elapsed
    print(
        f"kind=caption items={len(output)} batch_size={args.batch_size} "
        f"encode_seconds={elapsed:.3f} items_per_second={items_per_second:.3f} "
        f"peak_allocated_gib={peak_allocated_gib:.3f}"
    )
    return output, ordered_ids, {
        "encode_seconds": elapsed,
        "items_per_second": items_per_second,
        "peak_allocated_gib": peak_allocated_gib,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--kind", choices=("image", "caption"), default="image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=BACKBONE)
    parser.add_argument("--pretrained", default=BACKBONE_PRETRAINED)
    parser.add_argument(
        "--revision",
        help="pin Hub weights and tokenizer to this exact revision",
    )
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--image-dir", default="data/raw/object_images")
    parser.add_argument("--image-ids", default="data/image_ids.json")
    parser.add_argument("--captions", default="data/captions.jsonl")
    parser.add_argument("--n-items", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--preprocess-workers", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-fp16", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.n_items < 0 or args.preprocess_workers < 0:
        parser.error("batch-size must be positive; counts must be nonnegative")
    if args.prefetch_batches < 1:
        parser.error("--prefetch-batches must be positive")
    if args.input_fp16 and args.precision != "fp16":
        parser.error("--input-fp16 is only valid with --precision fp16")
    output = args.output.resolve()
    production_data = (ROOT / "data").resolve()
    if output == production_data or production_data in output.parents:
        parser.error("benchmark outputs must not be written under production data/")
    args.output = output

    if not torch.cuda.is_available():
        raise RuntimeError("embedding throughput benchmark requires CUDA")
    device = "cuda"
    encoder = EncoderSpec(
        name=args.model,
        model=args.model,
        pretrained=args.pretrained,
        revision=args.revision,
    )
    load_started = time.monotonic()
    model, preprocess, tokenizer, encoder_source = load_open_clip_encoder(
        encoder,
        device=device,
        precision=args.precision,
    )
    model_load_seconds = time.monotonic() - load_started
    print(
        f"model={args.model} pretrained={args.pretrained} precision={args.precision} "
        f"revision={args.revision or 'floating'} "
        f"model_load_seconds={model_load_seconds:.3f}"
    )

    if args.kind == "image":
        output, ordered_ids, metrics = encode_images(args, model, preprocess, device)
        image_dir = (ROOT / args.image_dir).resolve()
        canonical_ids = sorted(
            path.relative_to(image_dir).as_posix()
            for path in image_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        selection_sources = {
            "image_dir": str(image_dir),
            "canonical_ids_sha256": sha256_json(canonical_ids),
            "canonical_count": len(canonical_ids),
        }
    else:
        output, ordered_ids, metrics = encode_captions(args, model, tokenizer, device)
        image_ids_path = (ROOT / args.image_ids).resolve()
        captions_path = (ROOT / args.captions).resolve()
        selection_sources = {
            "image_ids": str(image_ids_path),
            "image_ids_sha256": sha256_file(image_ids_path),
            "captions": str(captions_path),
            "captions_sha256": sha256_file(captions_path),
        }

    save_npy_atomic(args.output, output)
    ids_path, manifest_path = write_benchmark_sidecars(
        args.output,
        output,
        ordered_ids,
        kind=args.kind,
        seed=args.seed,
        requested_count=args.n_items,
        encoder=encoder,
        encoder_source=encoder_source,
        inference={
            "precision": args.precision,
            "autocast_dtype": None if args.precision == "fp32" else args.precision,
            "image_input_dtype": "fp16" if args.input_fp16 else "fp32",
            "batch_size": args.batch_size,
            "preprocess_workers": args.preprocess_workers,
            "prefetch_batches": args.prefetch_batches,
            "pin_memory": args.pin_memory,
            "model_load_seconds": model_load_seconds,
        },
        selection_sources=selection_sources,
        metrics=metrics,
    )
    print(f"wrote={args.output} shape={output.shape} dtype={output.dtype}")
    print(f"wrote_ids={ids_path} wrote_manifest={manifest_path}")


if __name__ == "__main__":
    main()
