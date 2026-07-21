"""Stage 4 — Materialize the frozen encoder inputs used by training.

Consumes the THINGS images, the class split, and a finalized dictionary. It
encodes the full and CC0 image sets, creates one prompt-derived text direction
per dictionary concept, and calibrates their train-split scores to soft labels.

Writes one checksummed directory containing ``image_embeddings*.npy``,
``image_ids.json``, ``concept_directions_initial.npy``, and ``labels.parquet``.
The expensive image arrays are reused when a compatible run is resumed; the
dictionary-specific directions and scores are rebuilt. Production runs use
CUDA and the pinned SigLIP2 Giant release.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

from conceptbasis.embedding import image_batches
from conceptbasis.encoders import (
    ENCODER_PRESETS,
    PROJECT_ENCODER,
    cache_identity,
    encoder_output_dir,
    load_open_clip_encoder,
    save_npy_atomic,
    select_encoder,
    validate_cache_manifest,
    write_json_atomic,
    write_cache_manifest,
)
from conceptbasis.splits import load_split_manifest, split_for_image
from conceptbasis.train import require_cuda

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR = os.path.join(ROOT, "data", "raw", "object_images")


def autocast_context(device: str):
    if device != "cuda":
        raise ValueError("embedding generation requires CUDA")
    return torch.autocast(device_type="cuda")


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


@torch.inference_mode()
def encode_images(
    model,
    preprocess,
    paths: list[str],
    device: str,
    batch_size: int,
    label: str,
    preprocess_workers: int,
    prefetch_batches: int,
):
    output = []
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    for batch_index, host_batch in enumerate(
        image_batches(paths, preprocess, batch_size, preprocess_workers, prefetch_batches)
    ):
        batch = (
            host_batch.to(device, dtype=torch.float16)
            if device == "cuda"
            else host_batch.to(device)
        )
        with autocast_context(device):
            encoded = torch.nn.functional.normalize(model.encode_image(batch), dim=-1)
        output.append(encoded.float().cpu().numpy().astype(np.float32))
        if batch_index == 0 or batch_index % 20 == 0:
            completed = min((batch_index + 1) * batch_size, len(paths))
            print(f"  embedded {label} {completed}/{len(paths)}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result = np.concatenate(output)
    print(
        f"  embedded {label}: {len(result) / elapsed:.3f} img/s "
        f"({len(result)} images in {elapsed:.3f}s)",
        flush=True,
    )
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--encoder",
        choices=tuple(ENCODER_PRESETS),
        default="siglip2-giant",
        help="named encoder release; non-legacy encoders use an isolated output directory",
    )
    parser.add_argument("--model", help="custom OpenCLIP model (requires --pretrained)")
    parser.add_argument("--pretrained", help="custom OpenCLIP pretrained tag")
    parser.add_argument(
        "--revision",
        help="Hugging Face commit/revision for exact weights and tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "artifact directory (defaults to data/ for project, isolated "
            "outputs/ for candidates)"
        ),
    )
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument(
        "--dictionary",
        default="data/dictionary_usage_profile_v8.json",
        help="dictionary used to construct concept directions and soft-label columns",
    )
    parser.add_argument("--img-dir", default="data/raw/object_images")
    parser.add_argument("--cc0-dir", default="data/raw/object_images_CC0")
    parser.add_argument("--cc0-image-ids", default="data/cc0_image_ids.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.preprocess_workers < 0 or args.prefetch_batches < 1:
        parser.error(
            "batch size and prefetch batches must be positive; workers must be nonnegative"
        )
    try:
        encoder = select_encoder(
            args.encoder,
            model=args.model,
            pretrained=args.pretrained,
            revision=args.revision,
        )
        output_dir = encoder_output_dir(Path(ROOT), encoder, args.output_dir)
    except ValueError as error:
        parser.error(str(error))

    dev = require_cuda()
    precision = "fp16"

    img_dir = os.path.join(ROOT, args.img_dir)
    ids = []
    for cur, _, fs in os.walk(img_dir):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ids.append(os.path.relpath(os.path.join(cur, f), img_dir))
    ids.sort()

    # The CC0 matrix uses the tracked attribute-file order expected by evaluation.
    cc0_dir = os.path.join(ROOT, args.cc0_dir)
    cc0_ids_path = os.path.join(ROOT, args.cc0_image_ids)
    with open(cc0_ids_path) as file:
        cc0_ids = json.load(file)
    cc0_files = sorted(
        name
        for name in os.listdir(cc0_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(cc0_ids) != len(set(cc0_ids)) or set(cc0_ids) != set(cc0_files):
        raise ValueError("CC0 image ID manifest does not match the CC0 image directory")

    dictionary_path = Path(args.dictionary)
    if not dictionary_path.is_absolute():
        dictionary_path = Path(ROOT) / dictionary_path
    dictionary_path = dictionary_path.resolve()
    if not dictionary_path.exists():
        parser.error(f"dictionary does not exist: {dictionary_path}")
    split_manifest_path = Path(args.split_manifest)
    if not split_manifest_path.is_absolute():
        split_manifest_path = Path(ROOT) / split_manifest_path
    identity = cache_identity(
        encoder,
        precision=precision,
        image_ids=ids,
        cc0_image_ids=cc0_ids,
        dictionary_path=dictionary_path,
        split_manifest_path=split_manifest_path,
    )
    production_run = output_dir == (Path(ROOT) / "data").resolve() and encoder == PROJECT_ENCODER
    model, pre, tok, encoder_source = load_open_clip_encoder(
        encoder,
        device=dev,
        precision=precision,
    )
    cache_manifest = validate_cache_manifest(
        output_dir,
        identity,
        source=encoder_source,
        allow_legacy=production_run,
        verify_hashes=not production_run,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(ids)} images device={dev} precision={precision} "
        f"encoder={encoder.model}/{encoder.pretrained} output={output_dir}"
    )
    if not production_run and cache_manifest is None:
        # Establish the run identity before writing expensive artifacts. Each
        # completed stage refreshes this manifest, so a later invocation can
        # resume after an interruption without adopting an unrelated cache.
        write_cache_manifest(
            output_dir,
            identity,
            source=encoder_source,
        )

    # ---- image embeddings ----
    emb_path = output_dir / "image_embeddings.npy"
    ids_path = output_dir / "image_ids.json"
    if os.path.exists(ids_path):
        with open(ids_path) as file:
            cached_ids = json.load(file)
        if cached_ids != ids:
            raise ValueError(f"{ids_path} does not match the full image directory")
    else:
        write_json_atomic(ids_path, ids)
    if os.path.exists(emb_path):
        img = np.load(emb_path)
        if img.shape[0] != len(ids):
            raise ValueError("cached image embeddings do not match the current image order")
        print("loaded cached embeddings")
    else:
        img = encode_images(
            model,
            pre,
            [os.path.join(img_dir, image_id) for image_id in ids],
            dev,
            args.batch_size,
            "full",
            args.preprocess_workers,
            args.prefetch_batches,
        )
        save_npy_atomic(emb_path, img)
    if not production_run:
        write_cache_manifest(output_dir, identity, source=encoder_source)

    cc0_path = output_dir / "image_embeddings_cc0.npy"
    if os.path.exists(cc0_path):
        cc0 = np.load(cc0_path)
        if cc0.shape[0] != len(cc0_ids):
            raise ValueError("cached CC0 embeddings do not match data/cc0_image_ids.json")
        print("loaded cached CC0 embeddings")
    else:
        cc0 = encode_images(
            model,
            pre,
            [os.path.join(cc0_dir, image_id) for image_id in cc0_ids],
            dev,
            args.batch_size,
            "CC0",
            args.preprocess_workers,
            args.prefetch_batches,
        )
        save_npy_atomic(cc0_path, cc0)
    if not production_run:
        write_cache_manifest(output_dir, identity, source=encoder_source)

    # ---- concept directions ----
    with dictionary_path.open() as file:
        d = json.load(file)
    with autocast_context(dev):
        e_base = torch.nn.functional.normalize(
            model.encode_text(tok(["an object"]).to(dev)), dim=-1
        )[0].float().cpu().numpy()
    dirs = []
    for c in d:
        t = tok([f"an object that is {m}" for m in c["members"]]).to(dev)
        with autocast_context(dev):
            e = (
                torch.nn.functional.normalize(model.encode_text(t), dim=-1)
                .mean(0)
                .float()
                .cpu()
                .numpy()
            )
        v = e - e_base
        dirs.append(v / (np.linalg.norm(v) + 1e-8))
    D = np.stack(dirs).astype(np.float32)
    save_npy_atomic(output_dir / "concept_directions_initial.npy", D)

    manifest = load_split_manifest(ROOT, args.split_manifest)
    concept_of = [x.split(os.sep)[0] for x in ids]
    split = np.array([split_for_image(manifest, image_id) for image_id in ids])
    train = split == "train"

    # ---- scores -> train-fitted GMM-calibrated soft labels ----
    S = img @ D.T                                          # [N, 256]
    soft = np.zeros_like(S, dtype=np.float32)
    for k in range(S.shape[1]):
        all_scores = S[:, k].reshape(-1, 1)
        gm = GaussianMixture(2, random_state=0, n_init=2).fit(all_scores[train])
        soft[:, k] = gm.predict_proba(all_scores)[:, int(gm.means_.argmax())]
        if k % 32 == 0:
            print(f"  calibrated {k}/256", flush=True)

    out = pd.DataFrame({"image_id": ids, "concept": concept_of, "split": split})
    score_frame = pd.DataFrame(
        soft,
        columns=[f"s_{concept['name']}" for concept in d],
    )
    out = pd.concat([out, score_frame], axis=1)
    write_parquet_atomic(output_dir / "labels.parquet", out)
    if not production_run:
        manifest_path = write_cache_manifest(
            output_dir,
            identity,
            source=encoder_source,
        )
        print(f"wrote {manifest_path}")
    print("split sizes:", out.split.value_counts().to_dict())
    print("mean soft-positive rate (s>=0.5):", float((soft >= 0.5).mean()))
    print(f"wrote {output_dir / 'labels.parquet'}")


if __name__ == "__main__":
    main()
