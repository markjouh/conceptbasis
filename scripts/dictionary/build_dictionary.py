"""Build the 256-concept dictionary from the discovery split.

Verified recipe:
  - attributes: model-generated tags on train-class CC0 images
  - dictionary encoder: ViT-B-32/laion2b_s34b_b79k
  - phrase profiles: standardized image/text scores with top SVD component removed
  - clustering: complete linkage at profile correlation >= 0.50
  - selection: support-ranked, rejecting profile twins above correlation 0.75

Development and test classes are excluded by default so the same frozen
dictionary can be evaluated on unseen object classes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os

import numpy as np
import torch

from conceptbasis.splits import image_class, load_split_manifest, split_for_image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
IMG_CACHE = os.path.join(ROOT, "data", "dictionary_image_embeddings_vitb32.npy")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def clip_embed(
    img_dir: str,
    phrases: list[str],
    allowed_classes: set[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    import open_clip
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL)
    model.eval().to(device)

    paths = sorted(
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if os.path.exists(IMG_CACHE):
        all_image_embeddings = np.load(IMG_CACHE)
        if len(all_image_embeddings) != len(paths):
            raise ValueError("dictionary image cache does not match CC0 files")
    else:
        batches = []
        for start in range(0, len(paths), 256):
            images = torch.stack(
                [preprocess(Image.open(path).convert("RGB")) for path in paths[start:start + 256]]
            ).to(device)
            batches.append(
                torch.nn.functional.normalize(model.encode_image(images), dim=-1)
                .cpu()
                .numpy()
            )
        all_image_embeddings = np.concatenate(batches)
        np.save(IMG_CACHE, all_image_embeddings)

    if allowed_classes is None:
        image_embeddings = all_image_embeddings
    else:
        keep = np.array(
            [image_class(os.path.basename(path)) in allowed_classes for path in paths]
        )
        image_embeddings = all_image_embeddings[keep]

    text_batches = []
    for start in range(0, len(phrases), 256):
        tokens = tokenizer(
            [f"an object that is {phrase}" for phrase in phrases[start:start + 256]]
        ).to(device)
        text_batches.append(
            torch.nn.functional.normalize(model.encode_text(tokens), dim=-1)
            .cpu()
            .numpy()
        )
    return image_embeddings, np.concatenate(text_batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attrs", default="data/attributes_train.jsonl")
    parser.add_argument("--img-dir", default="data/raw/object_images_CC0")
    parser.add_argument("--out", default="data/dictionary.json")
    parser.add_argument("--provenance-out", default="data/dictionary_provenance.json")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--min-mentions", type=int, default=3)
    parser.add_argument("--merge-corr", type=float, default=0.5)
    parser.add_argument("--max-twin-corr", type=float, default=0.75)
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument(
        "--split", choices=("train", "dev", "test", "all"), default="train"
    )
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split in {"test", "all"} and not args.allow_test:
        raise ValueError("dictionary construction may read test only with --allow-test")

    rows = [
        json.loads(line)
        for line in open(os.path.join(ROOT, args.attrs))
        if line.strip()
    ]
    manifest = load_split_manifest(ROOT, args.split_manifest)
    if args.split != "all":
        rows = [
            row
            for row in rows
            if split_for_image(manifest, row["image_id"]) == args.split
        ]
        allowed_classes = {
            concept
            for concept, split in manifest["classes"].items()
            if split == args.split
        }
    else:
        allowed_classes = None
    image_sets = [set(row["attributes"]) for row in rows if row.get("attributes")]
    counts = Counter(attribute for attributes in image_sets for attribute in attributes)
    phrases = sorted(
        phrase for phrase, count in counts.items() if count >= args.min_mentions
    )
    print(
        f"{len(image_sets)} images | {len(counts)} unique phrases | "
        f"{len(phrases)} with >= {args.min_mentions} mentions"
    )

    # Lexically fold explicit negations onto the negative pole. CLIP text
    # embeddings are not reliable negation detectors.
    prefixes = ("in", "un", "non", "anti", "dis")
    base_of = {}
    phrase_set = set(phrases)
    for phrase in phrases:
        word = phrase.replace("-", "")
        for prefix in prefixes:
            if word.startswith(prefix) and word[len(prefix):] in phrase_set:
                base_of[phrase] = word[len(prefix):]
    phrases = [phrase for phrase in phrases if phrase not in base_of]
    print(f"negation fold-in: {len(base_of)} phrases -> negative poles")

    image_embeddings, text_embeddings = clip_embed(
        os.path.join(ROOT, args.img_dir), phrases, allowed_classes
    )
    scores = image_embeddings @ text_embeddings.T
    scores = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    left, singular_values, right = np.linalg.svd(scores, full_matrices=False)
    scores = scores - (left[:, :1] * singular_values[:1]) @ right[:1]

    correlations = np.corrcoef(scores.T)
    distances = 1.0 - correlations
    np.fill_diagonal(distances, 0.0)
    distances = (distances + distances.T) / 2

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    tree = linkage(squareform(distances, checks=False), method="complete")
    labels = fcluster(
        tree, t=1.0 - args.merge_corr, criterion="distance"
    )
    print(
        f"complete-linkage @ corr>{args.merge_corr} -> "
        f"{len(set(labels))} clusters (target {args.k})"
    )

    concepts = []
    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        members = [phrases[index] for index in indices]
        name = max(members, key=lambda member: counts[member])
        negatives = sorted(
            phrase for phrase, base in base_of.items() if base in members
        )
        member_set = set(members)
        prevalence = sum(bool(attributes & member_set) for attributes in image_sets) / len(image_sets)
        concept = {
            "name": name,
            "members": members,
            "prevalence": prevalence,
            "mentions": sum(counts[member] for member in members),
            "profile": scores[:, indices].mean(1),
        }
        if negatives:
            concept["negative_pole"] = negatives
        concepts.append(concept)

    concepts.sort(key=lambda concept: -concept["mentions"])
    profiles = np.stack([concept["profile"] for concept in concepts])
    profiles = (profiles - profiles.mean(1, keepdims=True)) / (
        profiles.std(1, keepdims=True) + 1e-8
    )
    selected = []
    selected_profiles = []
    skipped = []
    for index, concept in enumerate(concepts):
        if len(selected) >= args.k:
            break
        if selected_profiles:
            correlation = np.stack(selected_profiles) @ profiles[index] / profiles.shape[1]
            if correlation.max() > args.max_twin_corr:
                skipped.append(
                    (
                        concept["name"],
                        selected[int(correlation.argmax())]["name"],
                        float(correlation.max()),
                    )
                )
                continue
        selected.append(concept)
        selected_profiles.append(profiles[index])

    print(f"greedy selection: kept {len(selected)}, skipped {len(skipped)} near-twins")
    for concept in selected:
        concept.pop("profile")
    selected.sort(key=lambda concept: -concept["prevalence"])

    output = os.path.join(ROOT, args.out)
    with open(output, "w") as file:
        json.dump(selected, file, indent=2)
    provenance = {
        "model": MODEL,
        "pretrained": PRETRAINED,
        "attribute_split": args.split,
        "n_attribute_images": len(image_sets),
        "k": args.k,
        "min_mentions": args.min_mentions,
        "merge_corr": args.merge_corr,
        "max_twin_corr": args.max_twin_corr,
        "inputs": {
            args.attrs: sha256(os.path.join(ROOT, args.attrs)),
            args.split_manifest: sha256(os.path.join(ROOT, args.split_manifest)),
        },
        "dictionary_sha256": sha256(output),
    }
    provenance_output = os.path.join(ROOT, args.provenance_out)
    with open(provenance_output, "w") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")
    print(f"wrote {args.out} ({len(selected)} concepts)")


if __name__ == "__main__":
    main()
