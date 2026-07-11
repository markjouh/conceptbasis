"""Shared class-level split utilities.

Every THINGS image and its corresponding one-image CC0 representative inherit
the split of their object class. This prevents any object class from appearing
in more than one of train, development, and test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


SPLITS = ("train", "dev", "test")


def image_class(image_id: str) -> str:
    """Return the THINGS class for a full-set relative path or CC0 filename."""
    normalized = image_id.replace("\\", "/")
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    return Path(normalized).stem


def load_split_manifest(root: str, path: str = "data/splits.json") -> dict:
    manifest_path = path if os.path.isabs(path) else os.path.join(root, path)
    with open(manifest_path) as file:
        manifest = json.load(file)
    if manifest.get("unit") != "things_object_class":
        raise ValueError("split manifest must use THINGS object classes")
    values = set(manifest["classes"].values())
    if not values <= set(SPLITS):
        raise ValueError(f"unknown split names: {sorted(values - set(SPLITS))}")
    return manifest


def split_for_image(manifest: dict, image_id: str) -> str:
    concept = image_class(image_id)
    try:
        return manifest["classes"][concept]
    except KeyError as error:
        raise KeyError(f"class {concept!r} is absent from split manifest") from error
