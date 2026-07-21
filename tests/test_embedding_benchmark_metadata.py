import json
import random

import numpy as np
import pytest
import torch

from conceptbasis.encoders import SIGLIP2_GIANT, sha256_file, sha256_json
from scripts.benchmarks.benchmark_embeddings import (
    autocast_context,
    image_paths,
    main,
    save_npy_atomic,
    write_benchmark_sidecars,
)


def test_autocast_dtype_follows_selected_precision(monkeypatch):
    calls = []

    class Context:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    def fake_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        return Context()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with autocast_context("cuda", "fp16"):
        pass
    with autocast_context("cuda", "bf16"):
        pass
    with autocast_context("cuda", "fp32"):
        pass

    assert calls == [("cuda", torch.float16), ("cuda", torch.bfloat16)]


def test_input_fp16_rejects_incompatible_model_precision(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--output",
                str(tmp_path / "unused.npy"),
                "--precision",
                "bf16",
                "--input-fp16",
            ]
        )
    assert error.value.code == 2


def test_benchmark_refuses_production_cache_output():
    with pytest.raises(SystemExit) as error:
        main(["--output", "data/image_embeddings.npy"])
    assert error.value.code == 2


def test_benchmark_sidecars_record_exact_order_release_and_artifact(tmp_path):
    output_path = tmp_path / "siglip-image.npy"
    output = np.eye(3, dtype=np.float32)
    ordered_ids = ["b/2.jpg", "a/1.jpg", "c/3.jpg"]
    save_npy_atomic(output_path, output)

    ids_path, manifest_path = write_benchmark_sidecars(
        output_path,
        output,
        ordered_ids,
        kind="image",
        seed=42,
        requested_count=3,
        encoder=SIGLIP2_GIANT,
        encoder_source={
            "hf_revision": SIGLIP2_GIANT.revision,
            "image_size": 384,
            "preprocess": {"resize_mode": "squash"},
        },
        inference={"precision": "fp16", "image_input_dtype": "fp16"},
        selection_sources={"canonical_count": 3},
        metrics={"items_per_second": 12.5},
    )

    ids = json.loads(ids_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert ids["ids"] == ordered_ids
    assert ids["ordered_ids_sha256"] == sha256_json(ordered_ids)
    assert manifest["encoder"]["revision"] == SIGLIP2_GIANT.revision
    assert manifest["encoder_source"]["image_size"] == 384
    assert manifest["inference"]["precision"] == "fp16"
    assert manifest["selection"]["order"] == "seeded-shuffle"
    assert manifest["selection"]["seed"] == 42
    assert manifest["artifact"]["sha256"] == sha256_file(output_path)
    assert manifest["artifact"]["shape"] == [3, 3]


def test_image_selection_sorts_serialized_ids_before_seeded_shuffle(tmp_path):
    ids = [
        "ice/ice_01.jpg",
        "ice-cream_cone/ice-cream_cone_01.jpg",
        "zebra/zebra_01.jpg",
    ]
    for image_id in ids:
        path = tmp_path / image_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    expected = sorted(ids)
    random.Random(42).shuffle(expected)
    selected = [
        path.relative_to(tmp_path).as_posix()
        for path in image_paths(tmp_path, n_items=0, seed=42)
    ]

    assert selected == expected
