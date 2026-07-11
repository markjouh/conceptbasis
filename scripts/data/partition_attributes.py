"""Partition an existing attribute JSONL according to the class manifest."""

from __future__ import annotations

import argparse
import json
import os

from conceptbasis.splits import load_split_manifest, split_for_image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--train-out", default="data/attributes_train.jsonl")
    parser.add_argument("--dev-out", default="data/attributes_dev.jsonl")
    parser.add_argument("--test-out", default="data/heldout/attributes_test.jsonl")
    parser.add_argument("--image-ids-out", default="data/cc0_image_ids.json")
    args = parser.parse_args()

    manifest = load_split_manifest(ROOT, args.split_manifest)
    outputs = {
        "train": os.path.join(ROOT, args.train_out),
        "dev": os.path.join(ROOT, args.dev_out),
        "test": os.path.join(ROOT, args.test_out),
    }
    handles = {}
    counts = {split: 0 for split in outputs}
    image_ids = []
    try:
        for split, path in outputs.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handles[split] = open(path, "w")
        with open(os.path.join(ROOT, args.attributes)) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_ids.append(row["image_id"])
                split = split_for_image(manifest, row["image_id"])
                handles[split].write(json.dumps(row) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    with open(os.path.join(ROOT, args.image_ids_out), "w") as file:
        json.dump(image_ids, file, indent=2)
        file.write("\n")
    print("wrote", counts)


if __name__ == "__main__":
    main()
