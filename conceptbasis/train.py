"""Train contrastive, group-mean, or reverse-ridge ConceptBasis adapters.

The three objectives preserve the project's incremental development path:

``contrastive``
    Symmetric image-text contrastive loss only.
``group-mean``
    Contrastive loss plus minibatch soft positive-minus-negative direction
    orthogonality (the original ConceptBasis objective).
``reverse-ridge``
    Contrastive minibatch updates followed by one differentiable full-training-
    set concept-to-embedding ridge solve and direction-orthogonality update per
    epoch (the active objective).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from conceptbasis import BACKBONE as MODEL, BACKBONE_PRETRAINED as PRETRAINED
from conceptbasis.losses import (
    GroupMeanOrthogonalityLoss,
    ReverseRidgeOrthogonalityLoss,
    symmetric_clip_loss,
)
from conceptbasis.models import Adapter
from conceptbasis.splits import load_split_manifest, split_for_image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return requested


@torch.no_grad()
def caption_embeddings(
    ids: list[str],
    caption_path: Path,
    cache_path: Path,
    device: str,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model.eval().to(device)
    captions = {}
    for line in caption_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("caption"):
                captions[row["image_id"]] = row["caption"]
    texts = [captions.get(image_id, "an object") for image_id in ids]
    result = []
    for start in range(0, len(texts), 512):
        tokens = tokenizer(texts[start : start + 512]).to(device)
        encoded = F.normalize(model.encode_text(tokens), dim=-1)
        result.append(encoded.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(result)
    np.save(cache_path, embeddings)
    return embeddings


def load_dictionary_labels(
    *,
    image_ids: list[str],
    dictionary_path: Path,
    labels_path: Path,
    manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load complete sparse train labels for the reverse-ridge objective."""
    dictionary = json.loads(dictionary_path.read_text())
    names = [row["name"] for row in dictionary]
    if len(names) != len(set(names)):
        raise ValueError("dictionary concept names must be unique")
    name_to_index = {name: index for index, name in enumerate(names)}
    id_to_position = {image_id: index for index, image_id in enumerate(image_ids)}
    manifest = load_split_manifest(str(ROOT), str(manifest_path))

    rows_by_id: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(labels_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = row.get("image_id")
        if image_id in rows_by_id:
            raise ValueError(f"duplicate image_id at {labels_path}:{line_number}: {image_id}")
        if image_id not in id_to_position:
            raise ValueError(f"label image is absent from embedding manifest: {image_id}")
        if split_for_image(manifest, image_id) != "train":
            raise ValueError(f"non-train image in regression labels: {image_id}")
        present = row.get("present")
        uncertain = row.get("uncertain")
        if not isinstance(present, list) or not isinstance(uncertain, list):
            raise ValueError(f"invalid sparse label lists for {image_id}")
        unknown = (set(present) | set(uncertain)) - set(names)
        if unknown:
            raise ValueError(f"noncanonical labels for {image_id}: {sorted(unknown)}")
        if set(present) & set(uncertain):
            raise ValueError(f"present/uncertain overlap for {image_id}")
        rows_by_id[image_id] = row

    expected = {
        image_id for image_id in image_ids if split_for_image(manifest, image_id) == "train"
    }
    actual = set(rows_by_id)
    if actual != expected:
        raise ValueError(
            f"dictionary-label coverage mismatch: missing={len(expected - actual)} "
            f"extra={len(actual - expected)}"
        )

    ordered_ids = sorted(actual, key=id_to_position.__getitem__)
    positions = np.asarray([id_to_position[image_id] for image_id in ordered_ids], dtype=np.int64)
    targets = np.zeros((len(ordered_ids), len(names)), dtype=np.float32)
    observed = np.ones_like(targets)
    for row_index, image_id in enumerate(ordered_ids):
        row = rows_by_id[image_id]
        for name in row["present"]:
            targets[row_index, name_to_index[name]] = 1.0
        for name in row["uncertain"]:
            observed[row_index, name_to_index[name]] = 0.0

    positives = (targets * observed).sum(axis=0)
    diagnostics = {
        "rows": len(ordered_ids),
        "concepts": len(names),
        "positive_rate": float(targets.mean()),
        "uncertain_rate": float(1.0 - observed.mean()),
        "min_positives": int(positives.min()),
        "min_definite": int(observed.sum(axis=0).min()),
    }
    return positions, targets, observed, names, diagnostics


@torch.no_grad()
def adapted(adapter: Adapter, features: torch.Tensor, device: str, batch: int = 4096):
    was_training = adapter.training
    adapter.eval()
    result = torch.cat(
        [adapter(features[i : i + batch].to(device)).cpu() for i in range(0, len(features), batch)]
    )
    adapter.train(was_training)
    return result


def group_mean_directions(embeddings: torch.Tensor, scores: torch.Tensor, eps: float = 1e-6):
    positive = scores.sum(0)
    negative_scores = 1.0 - scores
    negative = negative_scores.sum(0)
    positive_mean = (scores.T @ embeddings) / positive.clamp(min=eps).unsqueeze(1)
    negative_mean = (negative_scores.T @ embeddings) / negative.clamp(min=eps).unsqueeze(1)
    return F.normalize(positive_mean - negative_mean, dim=1, eps=eps)


@torch.no_grad()
def evaluate(
    img_adapter: Adapter,
    txt_adapter: Adapter,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    scores: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    image_z = adapted(img_adapter, image_features, device)
    text_z = adapted(txt_adapter, text_features, device)
    image_n, text_n = F.normalize(image_z, dim=-1), F.normalize(text_z, dim=-1)
    similarity = image_n @ text_n.T
    order = similarity.argsort(dim=1, descending=True)
    rank = (order == torch.arange(len(image_n)).view(-1, 1)).float().argmax(1)
    retrieval = {k: float((rank < k).float().mean()) for k in (1, 5, 10)}

    directions = group_mean_directions(image_z, scores)
    gram = directions @ directions.T
    off = ~torch.eye(len(directions), dtype=torch.bool)
    aucs = []
    score_np = scores.numpy()
    projection = image_z @ directions.T
    for concept in range(scores.shape[1]):
        hard = score_np[:, concept] >= 0.5
        if 0 < hard.sum() < len(hard):
            aucs.append(roc_auc_score(hard, projection[:, concept].numpy()))
    return {
        "R@k": retrieval,
        "auroc": float(np.mean(aucs)),
        "group_mean_orth_rms": float(gram[off].square().mean().sqrt()),
    }


def configure_group_mean_loss(
    loss: GroupMeanOrthogonalityLoss,
    train_scores: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> None:
    if args.corr_exempt <= 0:
        return
    correlation = np.corrcoef(train_scores.T)
    if args.corr_weighting == "hard":
        mask = torch.from_numpy(np.abs(correlation) < args.corr_exempt).to(device)
        loss.set_pair_mask(mask)
        off = ~np.eye(loss.n, dtype=bool)
        exempt = int((~mask.cpu().numpy() & off).sum() / 2)
        print(f"group-mean orthogonality: exempting {exempt} correlated pairs", flush=True)
        return
    if not 0 <= args.corr_weight_floor < 1:
        raise ValueError("--corr_weight_floor must be in [0, 1)")
    if args.corr_weight_power <= 0:
        raise ValueError("--corr_weight_power must be positive")
    weights = args.corr_weight_floor + (1 - args.corr_weight_floor) * np.exp(
        -(np.abs(correlation) / args.corr_exempt) ** args.corr_weight_power
    )
    weights = weights.astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    loss.set_pair_weights(torch.from_numpy(weights).to(device))
    print(
        f"group-mean orthogonality: tau={args.corr_exempt:g} "
        f"floor={args.corr_weight_floor:g} power={args.corr_weight_power:g}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        choices=("contrastive", "group-mean", "reverse-ridge"),
        default="reverse-ridge",
    )
    parser.add_argument("--embed_dim", type=int, default=320)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ridge_chunk", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_orth", type=float, default=None)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--max_uncertain_fraction", type=float, default=0.25)
    parser.add_argument("--min_positives", type=int, default=1)
    parser.add_argument("--ema", type=float, default=0.9)
    parser.add_argument("--corr_exempt", type=float, default=0.15)
    parser.add_argument("--corr_weighting", choices=("hard", "smooth"), default="smooth")
    parser.add_argument("--corr_weight_floor", type=float, default=0.01)
    parser.add_argument("--corr_weight_power", type=float, default=4.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--image_ids", default="data/image_ids.json")
    parser.add_argument("--image_embeddings", default="data/image_embeddings.npy")
    parser.add_argument("--caption_embeddings", default="data/caption_embeddings.npy")
    parser.add_argument("--captions", default="data/captions.jsonl")
    parser.add_argument("--labels", default="data/labels.parquet")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument(
        "--reverse_labels",
        default="data/dictionary_labels_train_gemma26.jsonl",
    )
    parser.add_argument("--split_manifest", default="data/splits.json")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    defaults = {"contrastive": 0.0, "group-mean": 8.0, "reverse-ridge": 512.0}
    if args.lambda_orth is None:
        args.lambda_orth = defaults[args.objective]
    if args.lambda_orth < 0:
        raise ValueError("--lambda_orth must be nonnegative")
    if args.objective == "contrastive" and args.lambda_orth != 0:
        raise ValueError("contrastive objective requires --lambda_orth 0")
    if args.eval_every < 1 or args.batch < 2 or args.ridge_chunk < 1:
        raise ValueError("eval_every/ridge_chunk must be positive and batch at least 2")
    if args.run_name is None:
        args.run_name = {
            "contrastive": f"contrastive_d{args.embed_dim}",
            "group-mean": f"group_mean_d{args.embed_dim}",
            "reverse-ridge": f"reverse_ridge_d{args.embed_dim}",
        }[args.objective]
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    ids_path = ROOT / args.image_ids
    image_path = ROOT / args.image_embeddings
    caption_path = ROOT / args.caption_embeddings
    captions_path = ROOT / args.captions
    soft_labels_path = ROOT / args.labels
    dictionary_path = ROOT / args.dictionary
    reverse_labels_path = ROOT / args.reverse_labels
    manifest_path = ROOT / args.split_manifest

    image_ids = json.loads(ids_path.read_text())
    image_all = np.load(image_path)
    text_all = caption_embeddings(image_ids, captions_path, caption_path, device)
    if len(image_all) != len(image_ids) or len(text_all) != len(image_ids):
        raise ValueError("embedding caches do not match image_ids")

    frame = pd.read_parquet(soft_labels_path)
    if list(frame.image_id) != image_ids:
        raise ValueError("soft-label rows do not match image_ids")
    manifest = load_split_manifest(str(ROOT), str(manifest_path))
    expected_splits = [split_for_image(manifest, image_id) for image_id in image_ids]
    if list(frame.split) != expected_splits:
        raise ValueError("labels do not follow the class-level split manifest")
    score_columns = [column for column in frame.columns if column.startswith("s_")]
    score_all = frame[score_columns].to_numpy(dtype=np.float32)
    masks = {split: (frame.split == split).to_numpy() for split in ("train", "dev")}
    image = {split: torch.from_numpy(image_all[mask]) for split, mask in masks.items()}
    text = {split: torch.from_numpy(text_all[mask]) for split, mask in masks.items()}
    scores = {split: torch.from_numpy(score_all[mask]) for split, mask in masks.items()}
    n_concepts = len(score_columns)

    reverse_targets = reverse_observed = None
    score_names = [column.removeprefix("s_") for column in score_columns]
    concept_names = score_names
    label_info = None
    if args.objective == "reverse-ridge":
        positions, target_np, observed_np, concept_names, label_info = load_dictionary_labels(
            image_ids=image_ids,
            dictionary_path=dictionary_path,
            labels_path=reverse_labels_path,
            manifest_path=manifest_path,
        )
        expected_positions = np.flatnonzero(masks["train"])
        if not np.array_equal(positions, expected_positions):
            raise ValueError("reverse labels and soft-label train split differ in order")
        if concept_names != score_names:
            raise ValueError("reverse and group-mean concept order differs")
        reverse_targets = torch.from_numpy(target_np)
        reverse_observed = torch.from_numpy(observed_np)

    print(
        f"device={device} objective={args.objective} "
        f"train={len(image['train'])} dev={len(image['dev'])} "
        f"concepts={n_concepts} d={args.embed_dim} label_info={label_info}",
        flush=True,
    )

    img_adapter = Adapter(image_all.shape[1], args.embed_dim, args.hidden_dim).to(device)
    txt_adapter = Adapter(text_all.shape[1], args.embed_dim, args.hidden_dim).to(device)
    group_loss = None
    reverse_loss = None
    if args.objective == "group-mean":
        group_loss = GroupMeanOrthogonalityLoss(
            n_concepts, args.embed_dim, args.ema
        ).to(device)
        configure_group_mean_loss(group_loss, score_all[masks["train"]], args, device)
    elif args.objective == "reverse-ridge":
        reverse_loss = ReverseRidgeOrthogonalityLoss(
            alpha=args.ridge_alpha,
            max_uncertain_fraction=args.max_uncertain_fraction,
            min_positives=args.min_positives,
        )

    logit_scale = torch.nn.Parameter(
        torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        list(img_adapter.parameters()) + list(txt_adapter.parameters()) + [logit_scale],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    run_dir = ROOT / "outputs" / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": vars(args),
        "optimization": (
            "minibatch contrastive updates"
            if args.objective != "reverse-ridge"
            else (
                "minibatch contrastive updates followed by one exact full-train "
                "reverse-ridge update per epoch"
            )
        ),
        "inputs": {
            "split_manifest_sha256": sha256(manifest_path),
            "soft_labels_sha256": sha256(soft_labels_path),
            **(
                {
                    "dictionary_sha256": sha256(dictionary_path),
                    "reverse_labels_sha256": sha256(reverse_labels_path),
                }
                if args.objective == "reverse-ridge"
                else {}
            ),
        },
        "concept_names": concept_names,
    }
    (run_dir / "config.json").write_text(json.dumps(provenance, indent=2) + "\n")

    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.monotonic()
    stop = False
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(image["train"]))
        clip_sum = group_orth_sum = 0.0
        batches = 0
        for offset in range(0, len(permutation), args.batch):
            index = permutation[offset : offset + args.batch]
            if len(index) < 2:
                continue
            image_z = img_adapter(image["train"][index].to(device))
            text_z = txt_adapter(text["train"][index].to(device))
            clip = symmetric_clip_loss(
                F.normalize(image_z, dim=-1),
                F.normalize(text_z, dim=-1),
                logit_scale.clamp(max=np.log(100)).exp(),
            )
            loss = clip
            if group_loss is not None:
                group = group_loss(image_z, scores["train"][index].to(device))
                loss = loss + args.lambda_orth * group["orth"]
                group_orth_sum += float(group["orth"].detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clip_sum += float(clip.detach().cpu())
            batches += 1
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        reverse = None
        if reverse_loss is not None:
            assert reverse_targets is not None and reverse_observed is not None
            full_z = torch.cat(
                [
                    img_adapter(image["train"][start : start + args.ridge_chunk].to(device))
                    for start in range(0, len(image["train"]), args.ridge_chunk)
                ]
            )
            reverse = reverse_loss(
                full_z,
                reverse_targets.to(device),
                reverse_observed.to(device),
            )
            optimizer.zero_grad()
            (args.lambda_orth * reverse["orth"]).backward()
            optimizer.step()
        scheduler.step()

        train_metrics = {
            "clip": clip_sum / max(1, batches),
            "orth": (
                float(reverse["orth"].detach().cpu())
                if reverse is not None
                else group_orth_sum / max(1, batches)
            ),
        }
        if reverse is not None:
            train_metrics.update(
                {
                    "orth_rms": float(reverse["orth_rms"].detach().cpu()),
                    "explained_fraction": float(
                        reverse["explained_fraction"].detach().cpu()
                    ),
                }
            )
        record: dict[str, Any] = {
            "epoch": epoch,
            "step": global_step,
            "train": train_metrics,
        }
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1 or stop:
            record["dev"] = evaluate(
                img_adapter,
                txt_adapter,
                image["dev"],
                text["dev"],
                scores["dev"],
                device,
            )
            print(
                f"ep{epoch:03d} step={global_step} "
                f"clip={train_metrics['clip']:.3f} orth={train_metrics['orth']:.5f} "
                f"dev_R@5={record['dev']['R@k'][5]:.3f}",
                flush=True,
            )
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stop:
            break

    train_z = adapted(img_adapter, image["train"], device)
    if reverse_loss is not None:
        assert reverse_targets is not None and reverse_observed is not None
        final = reverse_loss(train_z, reverse_targets, reverse_observed)
        directions = final["directions"].numpy()
        final_direction = {
            key: float(final[key])
            for key in (
                "reconstruction",
                "fit_mse",
                "ridge_penalty",
                "explained_fraction",
                "orth",
                "orth_rms",
            )
        }
    else:
        directions_tensor = group_mean_directions(train_z, scores["train"])
        gram = directions_tensor @ directions_tensor.T
        off = ~torch.eye(len(directions_tensor), dtype=torch.bool)
        directions = directions_tensor.numpy()
        final_direction = {
            "orth_rms": float(gram[off].square().mean().sqrt()),
        }
    np.save(run_dir / "concept_directions.npy", directions)

    checkpoint = {
        "img_adapter": img_adapter.state_dict(),
        "txt_adapter": txt_adapter.state_dict(),
        "logit_scale": logit_scale.detach().cpu(),
        "config": vars(args),
        "provenance": provenance,
        "concept_names": concept_names,
        "final_direction": final_direction,
    }
    if group_loss is not None:
        checkpoint["orthogonality_loss"] = group_loss.state_dict()
    torch.save(checkpoint, run_dir / "ckpt.pt")
    print(
        f"saved {run_dir} elapsed_seconds={time.monotonic() - started:.1f} "
        f"final_orth_rms={final_direction['orth_rms']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
