"""Reproduce the 256-concept dictionary used by the original playground.

Verified recipe:
  - attributes: data/attributes.jsonl (1,854 CC0 images)
  - dictionary encoder: ViT-B-32/laion2b_s34b_b79k
  - phrase profiles: standardized image/text scores with top SVD component removed
  - clustering: complete linkage at profile correlation >= 0.50
  - selection: support-ranked, rejecting profile twins above correlation 0.75

With the archived ViT-B/32 image cache, the default command regenerates the
checked-in data/dictionary.json byte-for-byte (SHA-256
b5388af5425af0596768b0b72a531a5f71566fe0729325f78913e69f3bf55a6e).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
IMG_CACHE = os.path.join(ROOT, "data", "dictionary_image_embeddings_vitb32.npy")


@torch.no_grad()
def clip_embed(img_dir: str, phrases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import open_clip
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL)
    model.eval().to(device)

    if os.path.exists(IMG_CACHE):
        image_embeddings = np.load(IMG_CACHE)
    else:
        paths = sorted(
            os.path.join(img_dir, name)
            for name in os.listdir(img_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
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
        image_embeddings = np.concatenate(batches)
        np.save(IMG_CACHE, image_embeddings)

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
    parser.add_argument("--attrs", default="data/attributes.jsonl")
    parser.add_argument("--img-dir", default="data/raw/object_images_CC0")
    parser.add_argument("--out", default="data/dictionary.json")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--min-mentions", type=int, default=3)
    parser.add_argument("--merge-corr", type=float, default=0.5)
    parser.add_argument("--max-twin-corr", type=float, default=0.75)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in open(os.path.join(ROOT, args.attrs))
        if line.strip()
    ]
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
        os.path.join(ROOT, args.img_dir), phrases
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
    print(f"wrote {args.out} ({len(selected)} concepts)")


if __name__ == "__main__":
    main()
