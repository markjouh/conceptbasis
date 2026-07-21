"""Stage 1 utility — Partition an existing open-tag JSONL by object-class split.

This is for imported or legacy tag files. New production mining jobs select a
split directly and normally do not need this conversion step.
"""

from __future__ import annotations

import argparse
import json
import os

from conceptbasis.splits import load_split_manifest, split_for_image


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--open-tags", "--attributes", dest="open_tags", required=True
    )
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
        with open(os.path.join(ROOT, args.open_tags)) as source:
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
