"""Stage 3 utility — Merge complete fixed-dictionary label shards.

Validates split membership, dictionary vocabulary, image coverage, uniqueness,
and present/uncertain consistency before writing one ordered JSONL artifact and
its provenance sidecar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from conceptbasis.encoders import write_json_atomic
from conceptbasis.splits import load_split_manifest, split_for_image


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            raise ValueError(f"non-success row at {path}:{line_number}")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--image-ids", required=True)
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("merging the sealed test split requires --allow-test")

    source_paths = [resolve(value) for value in args.source]
    dictionary_path = resolve(args.dictionary)
    image_ids_path = resolve(args.image_ids)
    split_manifest_path = resolve(args.split_manifest)
    output = resolve(args.out)
    meta_path = output.with_suffix(output.suffix + ".meta.json")

    dictionary = json.loads(dictionary_path.read_text())
    names = [row["name"] for row in dictionary]
    if len(names) != len(set(names)):
        raise ValueError("dictionary concept names must be unique")
    name_set = set(names)
    image_ids = json.loads(image_ids_path.read_text())
    position = {image_id: index for index, image_id in enumerate(image_ids)}
    if len(position) != len(image_ids):
        raise ValueError("image ID manifest contains duplicates")
    manifest = load_split_manifest(str(ROOT), str(split_manifest_path))
    expected = {
        image_id for image_id in image_ids if split_for_image(manifest, image_id) == args.split
    }

    rows_by_id: dict[str, dict[str, Any]] = {}
    sources = []
    for path in source_paths:
        rows = read_rows(path)
        sources.append({"path": display(path), "sha256": sha256(path), "rows": len(rows)})
        for row in rows:
            image_id = row.get("image_id")
            if image_id in rows_by_id:
                raise ValueError(f"duplicate image ID across shards: {image_id}")
            if image_id not in expected:
                raise ValueError(f"image outside requested {args.split} split: {image_id}")
            present = row.get("present")
            uncertain = row.get("uncertain")
            if not isinstance(present, list) or not isinstance(uncertain, list):
                raise ValueError(f"invalid sparse labels for {image_id}")
            unknown = (set(present) | set(uncertain)) - name_set
            if unknown:
                raise ValueError(f"noncanonical labels for {image_id}: {sorted(unknown)}")
            if set(present) & set(uncertain):
                raise ValueError(f"present/uncertain overlap for {image_id}")
            rows_by_id[image_id] = row

    actual = set(rows_by_id)
    if actual != expected:
        raise ValueError(
            f"coverage mismatch: missing={len(expected - actual)} "
            f"extra={len(actual - expected)}"
        )
    ordered = sorted(rows_by_id.values(), key=lambda row: position[row["image_id"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in ordered)
    )
    temporary.replace(output)
    metadata = {
        "schema": "conceptbasis.merged-dictionary-labels/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "rows": len(ordered),
        "concepts": len(names),
        "sources": sources,
        "dictionary": {"path": display(dictionary_path), "sha256": sha256(dictionary_path)},
        "image_ids": {"path": display(image_ids_path), "sha256": sha256(image_ids_path)},
        "split_manifest": {
            "path": display(split_manifest_path),
            "sha256": sha256(split_manifest_path),
        },
        "artifact": {"path": display(output), "sha256": sha256(output)},
    }
    write_json_atomic(meta_path, metadata)
    print(f"wrote {output} rows={len(ordered)} sha256={metadata['artifact']['sha256']}")


if __name__ == "__main__":
    main()
