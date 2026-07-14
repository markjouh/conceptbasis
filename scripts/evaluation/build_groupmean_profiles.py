"""Build standardized CC0 profiles for group-mean or reverse-ridge directions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conceptbasis.losses import reverse_ridge_objective
from conceptbasis.models import Adapter
from conceptbasis.splits import load_split_manifest, split_for_image
from conceptbasis.train import load_dictionary_labels


ROOT = Path(__file__).resolve().parents[2]


def l2_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def standardized_profile(
    train_z: np.ndarray,
    cc0_z: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    train_projection = train_z @ directions.T
    mean = train_projection.mean(axis=0)
    std = train_projection.std(axis=0) + 1e-6
    return ((cc0_z @ directions.T - mean) / std).astype(np.float32)


def group_mean_profile(
    z: np.ndarray,
    cc0_z: np.ndarray,
    scores: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    directions = []
    for concept in range(scores.shape[1]):
        score = scores[train, concept]
        positive = (score[:, None] * z[train]).sum(0) / max(float(score.sum()), 1e-3)
        negative_score = 1.0 - score
        negative = (negative_score[:, None] * z[train]).sum(0) / max(
            float(negative_score.sum()), 1e-3
        )
        directions.append(positive - negative)
    return standardized_profile(z[train], cc0_z, l2_rows(np.stack(directions)))


def checkpoint_direction_kind(checkpoint: dict, override: str) -> str:
    if override != "auto":
        return override
    config = checkpoint["config"]
    if config.get("objective") == "reverse-ridge" or "final_reverse_ridge" in checkpoint:
        return "reverse-ridge"
    return "group-mean"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--cc0-embeddings", required=True)
    parser.add_argument("--labels", required=True, help="soft group-mean label parquet")
    parser.add_argument("--reverse-labels", default="data/dictionary_labels_train_gemma26.jsonl")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument("--image-ids", default="data/image_ids.json")
    parser.add_argument("--attributes", default="data/attributes_dev.jsonl")
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--cc0-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--include-frozen", action="store_true")
    parser.add_argument(
        "--direction-kind",
        choices=("auto", "group-mean", "reverse-ridge"),
        default="auto",
        help="auto uses reverse ridge only for reverse-ridge checkpoints",
    )
    parser.add_argument("--ridge-alpha", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cc0_split == "test" and not args.allow_test:
        raise ValueError("reading test requires --allow-test")

    embeddings = np.load(ROOT / args.embeddings).astype(np.float32)
    cc0_all = np.load(ROOT / args.cc0_embeddings).astype(np.float32)
    labels = pd.read_parquet(ROOT / args.labels)
    score_columns = [column for column in labels if column.startswith("s_")]
    names = np.asarray([column[2:] for column in score_columns])
    scores = labels[score_columns].to_numpy(dtype=np.float32)
    train = (labels.split == "train").to_numpy()
    image_ids = json.loads((ROOT / args.image_ids).read_text())
    if list(labels.image_id) != image_ids or len(embeddings) != len(image_ids):
        raise ValueError("embeddings, soft labels, and image_ids are not aligned")

    rows = [
        json.loads(line)
        for line in (ROOT / args.attributes).read_text().splitlines()
        if line.strip()
    ]
    manifest = load_split_manifest(str(ROOT), str(ROOT / args.split_manifest))
    if any(split_for_image(manifest, row["image_id"]) != args.cc0_split for row in rows):
        raise ValueError("attribute file contains rows outside --cc0-split")
    cc0_order = json.loads((ROOT / "data/cc0_image_ids.json").read_text())
    if len(cc0_all) != len(cc0_order):
        raise ValueError("CC0 embeddings do not match their image ID manifest")
    cc0_index = {image_id: index for index, image_id in enumerate(cc0_order)}
    eval_ids = [row["image_id"] for row in rows]
    cc0 = cc0_all[[cc0_index[image_id] for image_id in eval_ids]]

    arrays: dict[str, np.ndarray] = {
        "names": names,
        "image_ids": np.asarray(eval_ids),
    }
    if args.include_frozen:
        arrays["frozen"] = group_mean_profile(embeddings, cc0, scores, train)
        print("built frozen", arrays["frozen"].shape, flush=True)

    reverse_data = None
    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError("--checkpoint must be NAME=PATH")
        label, checkpoint_path = spec.split("=", 1)
        checkpoint = torch.load(ROOT / checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        adapter = Adapter(
            embeddings.shape[1],
            config["embed_dim"],
            config.get("hidden_dim", 1024),
        )
        adapter.load_state_dict(checkpoint["img_adapter"])
        adapter.eval()
        with torch.no_grad():
            z = adapter(torch.from_numpy(embeddings)).numpy()
            z_cc0 = adapter(torch.from_numpy(cc0)).numpy()

        kind = checkpoint_direction_kind(checkpoint, args.direction_kind)
        if kind == "group-mean":
            arrays[label] = group_mean_profile(z, z_cc0, scores, train)
        else:
            if reverse_data is None:
                positions, target_np, observed_np, reverse_names, _ = load_dictionary_labels(
                    image_ids=image_ids,
                    dictionary_path=ROOT / args.dictionary,
                    labels_path=ROOT / args.reverse_labels,
                    manifest_path=ROOT / args.split_manifest,
                )
                if not np.array_equal(positions, np.flatnonzero(train)):
                    raise ValueError("reverse labels differ from the soft-label train split")
                if list(reverse_names) != list(names):
                    raise ValueError("reverse and group-mean concept order differs")
                reverse_data = (
                    positions,
                    torch.from_numpy(target_np),
                    torch.from_numpy(observed_np),
                )
            positions, targets, observed = reverse_data
            alpha = args.ridge_alpha
            if alpha is None:
                alpha = float(config.get("ridge_alpha", 1e-3))
            result = reverse_ridge_objective(
                torch.from_numpy(z[positions]),
                targets,
                observed,
                alpha=alpha,
            )
            directions = result["directions"].numpy()
            arrays[label] = standardized_profile(z[positions], z_cc0, directions)
            print(
                f"{label}: reverse alpha={alpha:g} "
                f"orth_rms={float(result['orth_rms']):.5f}",
                flush=True,
            )
        print(f"built {label} ({kind}) {arrays[label].shape}", flush=True)

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
