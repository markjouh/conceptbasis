"""Stage 0 — Create the deterministic THINGS train/dev/test split.

Writes ``data/splits.json`` at object-class granularity so every full-set image
and its matching CC0 representative remain in the same partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np

from conceptbasis.splits import image_class


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--image-ids", default="data/image_ids.json")
    parser.add_argument("--cc0-dir", default="data/raw/object_images_CC0")
    parser.add_argument("--out", default="data/splits.json")
    args = parser.parse_args()
    if args.train_fraction <= 0 or args.dev_fraction <= 0:
        raise ValueError("train and development fractions must be positive")
    if args.train_fraction + args.dev_fraction >= 1:
        raise ValueError("train + development fractions must leave a test split")

    image_ids_path = os.path.join(ROOT, args.image_ids)
    image_ids = json.load(open(image_ids_path))
    full_classes = sorted({image_class(image_id) for image_id in image_ids})
    cc0_dir = os.path.join(ROOT, args.cc0_dir)
    cc0_files = sorted(
        name
        for name in os.listdir(cc0_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    cc0_classes = sorted({image_class(name) for name in cc0_files})
    if full_classes != cc0_classes:
        raise ValueError("full THINGS and CC0 class sets do not match")

    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(full_classes))
    n_classes = len(shuffled)
    n_train = round(n_classes * args.train_fraction)
    n_dev = round(n_classes * args.dev_fraction)
    assignments = {}
    for index, concept in enumerate(shuffled):
        if index < n_train:
            assignments[concept] = "train"
        elif index < n_train + n_dev:
            assignments[concept] = "dev"
        else:
            assignments[concept] = "test"
    assignments = dict(sorted(assignments.items()))

    image_counts = Counter(assignments[image_class(image_id)] for image_id in image_ids)
    class_counts = Counter(assignments.values())
    manifest = {
        "version": 1,
        "unit": "things_object_class",
        "seed": args.seed,
        "fractions": {
            "train": args.train_fraction,
            "dev": args.dev_fraction,
            "test": 1 - args.train_fraction - args.dev_fraction,
        },
        "public_preview_split": "dev",
        "class_counts": dict(sorted(class_counts.items())),
        "image_counts": dict(sorted(image_counts.items())),
        "inputs": {
            args.image_ids: sha256(image_ids_path),
            args.cc0_dir: hashlib.sha256("\n".join(cc0_files).encode()).hexdigest(),
        },
        "classes": assignments,
    }
    output = os.path.join(ROOT, args.out)
    with open(output, "w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print("class counts", manifest["class_counts"])
    print("image counts", manifest["image_counts"])
    print("public preview", manifest["public_preview_split"])
    print("wrote", output)


if __name__ == "__main__":
    main()
