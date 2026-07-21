import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.compare_encoder_embeddings import (
    check_shapes,
    deterministic_subset,
    embedding_checks,
    load_benchmark_manifests,
    main,
    paired_retrieval_metrics,
    quality_gates,
    resolve_pair_ids,
    sha256_json,
    validate_benchmark_manifest_pairing,
)


def test_paired_retrieval_is_exact_and_block_size_invariant():
    embeddings = np.eye(6, dtype=np.float32)
    indexes = np.arange(6)

    one_row_blocks = paired_retrieval_metrics(
        embeddings, embeddings, indexes, ks=(1, 3), block_size=1
    )
    one_block = paired_retrieval_metrics(
        embeddings, embeddings, indexes, ks=(1, 3), block_size=6
    )

    assert one_row_blocks == one_block
    assert one_block["image_to_caption"]["recall_at"] == {"1": 1.0, "3": 1.0}
    assert one_block["caption_to_image"]["mean_rank"] == 1.0
    assert one_block["paired_cosine"]["mean"] == pytest.approx(1.0)


def test_retrieval_ties_have_deterministic_index_order():
    embeddings = np.ones((4, 2), dtype=np.float32)
    result = paired_retrieval_metrics(
        embeddings, embeddings, np.arange(4), ks=(1, 2, 4), block_size=3
    )

    assert result["image_to_caption"]["recall_at"] == {
        "1": 0.25,
        "2": 0.5,
        "4": 1.0,
    }
    assert result["image_to_caption"]["mean_rank"] == 2.5
    assert result["caption_to_image"] == result["image_to_caption"]


def test_shape_check_allows_encoders_with_different_dimensions():
    arrays = {
        "baseline_image": np.zeros((3, 2), dtype=np.float32),
        "baseline_caption": np.zeros((3, 2), dtype=np.float32),
        "candidate_image": np.zeros((3, 5), dtype=np.float32),
        "candidate_caption": np.zeros((3, 5), dtype=np.float32),
    }

    assert check_shapes(arrays) == 3
    arrays["candidate_caption"] = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="candidate image/caption dimensions differ"):
        check_shapes(arrays)


def test_subset_is_stable_and_independent_of_input_row_order():
    ids = [f"class/image_{index}.jpg" for index in range(20)]
    selected = deterministic_subset(ids, limit=7, seed=42)
    selected_ids = {ids[index] for index in selected}

    reversed_ids = list(reversed(ids))
    reversed_selected = deterministic_subset(reversed_ids, limit=7, seed=42)

    assert len(selected) == 7
    assert np.all(selected[:-1] < selected[1:])
    assert selected_ids == {reversed_ids[index] for index in reversed_selected}


def test_explicit_array_id_sidecars_must_have_identical_row_order(tmp_path):
    labels = (
        "baseline_image",
        "baseline_caption",
        "candidate_image",
        "candidate_caption",
    )
    paths = {label: tmp_path / f"{label}.npy" for label in labels}
    ids = ["b/2.jpg", "a/1.jpg", "c/3.jpg"]

    def write_sidecar(label, values):
        payload = {
            "schema": "conceptbasis.ordered-embedding-ids/v1",
            "kind": "caption" if label.endswith("caption") else "image",
            "order": "seeded-shuffle",
            "selection_seed": 42,
            "ordered_ids_sha256": sha256_json(values),
            "ids": values,
        }
        (tmp_path / f"{label}.npy.ids.json").write_text(json.dumps(payload))

    for label in labels:
        write_sidecar(label, ids)
    args = Namespace(**{f"{label}_ids": None for label in labels})
    resolved, provenance = resolve_pair_ids(args, paths, list(reversed(ids)), len(ids))
    assert resolved == ids
    assert provenance["mode"] == "explicit-sidecars"

    write_sidecar("candidate_caption", ["a/1.jpg", "b/2.jpg", "c/3.jpg"])
    with pytest.raises(ValueError, match="different explicit row orders"):
        resolve_pair_ids(args, paths, list(reversed(ids)), len(ids))


def test_benchmark_manifests_bind_arrays_encoders_and_row_sidecars(tmp_path):
    labels = (
        "baseline_image",
        "baseline_caption",
        "candidate_image",
        "candidate_caption",
    )
    arrays = {}
    paths = {}
    ids = ["a/1.jpg", "b/2.jpg"]
    ids_hash = sha256_json(ids)
    sidecars = {}
    for label in labels:
        array = np.eye(2, dtype=np.float32)
        path = tmp_path / f"{label}.npy"
        np.save(path, array)
        arrays[label] = np.load(path)
        paths[label] = path
        sidecar_path = Path(f"{path}.ids.json")
        sidecar_path.write_text(json.dumps({"ids": ids}))
        sidecar_hash = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        sidecars[label] = {"sha256": sidecar_hash}
        role = label.split("_", 1)[0]
        manifest = {
            "schema": "conceptbasis.embedding-benchmark/v1",
            "encoder": {
                "name": role,
                "model": role,
                "pretrained": "release",
                "revision": "commit",
            },
            "inference": {"precision": "fp16"},
            "selection": {
                "ordered_ids_sha256": ids_hash,
                "ordered_ids_sidecar_sha256": sidecar_hash,
            },
            "artifact": {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": [2, 2],
                "dtype": "float32",
            },
        }
        Path(f"{path}.manifest.json").write_text(json.dumps(manifest))

    manifests = load_benchmark_manifests(paths, arrays)
    encoders = validate_benchmark_manifest_pairing(
        manifests,
        {"sidecars": sidecars},
        ids_hash,
    )
    assert encoders["candidate"]["revision"] == "commit"

    np.save(paths["candidate_image"], np.ones((2, 2), dtype=np.float32))
    arrays["candidate_image"] = np.load(paths["candidate_image"])
    with pytest.raises(ValueError, match="does not match its array"):
        load_benchmark_manifests(paths, arrays)


def test_quality_gates_detect_norm_and_retrieval_regressions():
    baseline = np.eye(4, dtype=np.float32)
    candidate_image = 2 * np.eye(4, dtype=np.float32)
    candidate_caption = np.eye(4, dtype=np.float32)[[1, 0, 2, 3]]
    indexes = np.arange(4)
    checks = {
        "baseline_image": embedding_checks(baseline, 5e-4),
        "baseline_caption": embedding_checks(baseline, 5e-4),
        "candidate_image": embedding_checks(candidate_image, 5e-4),
        "candidate_caption": embedding_checks(candidate_caption, 5e-4),
    }
    baseline_metrics = paired_retrieval_metrics(baseline, baseline, indexes, ks=(1,))
    candidate_metrics = paired_retrieval_metrics(
        candidate_image, candidate_caption, indexes, ks=(1,)
    )

    gates = quality_gates(checks, baseline_metrics, candidate_metrics, 5e-4, 1.0)

    assert not gates["passed"]
    assert not gates["normalization"]["checks"]["candidate_image"]["passed"]
    regression = gates["paired_retrieval_noninferiority"]["checks"]
    assert regression["image_to_caption.recall_at_1"]["delta_percentage_points"] == -50.0
    assert not regression["image_to_caption.recall_at_1"]["passed"]


def test_run_writes_provenance_for_exact_input_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.benchmarks.compare_encoder_embeddings.repository_state",
        lambda: {
            "git_commit": "test-commit",
            "worktree_dirty": True,
            "evaluator_path": "/test/evaluator.py",
            "evaluator_sha256": "evaluator-hash",
        },
    )
    baseline = np.eye(4, dtype=np.float32)
    candidate = np.pad(baseline, ((0, 0), (0, 2)))
    paths = {}
    for label, array in {
        "baseline-image": baseline,
        "baseline-caption": baseline,
        "candidate-image": candidate,
        "candidate-caption": candidate,
    }.items():
        path = tmp_path / f"{label}.npy"
        np.save(path, array)
        paths[label] = path
    image_ids = tmp_path / "image_ids.json"
    image_ids.write_text(json.dumps([f"object/{index}.jpg" for index in range(4)]))
    captions = tmp_path / "captions.jsonl"
    captions.write_text(
        "".join(
            json.dumps({"image_id": f"object/{index}.jpg", "caption": f"object {index}"})
            + "\n"
            for index in range(4)
        )
    )
    output = tmp_path / "comparison.json"
    main(
        [
            "--baseline-image",
            str(paths["baseline-image"]),
            "--baseline-caption",
            str(paths["baseline-caption"]),
            "--candidate-image",
            str(paths["candidate-image"]),
            "--candidate-caption",
            str(paths["candidate-caption"]),
            "--image-ids",
            str(image_ids),
            "--captions",
            str(captions),
            "--array-order",
            "canonical",
            "--candidate-model",
            "ViT-gopt-16-SigLIP2-384",
            "--candidate-pretrained",
            "webli",
            "--candidate-revision",
            "ad3410b",
            "--retrieval-limit",
            "0",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())

    expected_hash = hashlib.sha256(paths["candidate-image"].read_bytes()).hexdigest()
    assert report["schema"] == "conceptbasis.encoder-comparison"
    assert report["models"]["candidate"]["revision"] == "ad3410b"
    assert report["inputs"]["candidate_image"]["sha256"] == expected_hash
    assert report["inputs"]["baseline_image"]["shape"] == [4, 4]
    assert report["inputs"]["candidate_image"]["shape"] == [4, 6]
    assert report["pairing"]["array_order"] == "canonical"
    assert report["pairing"]["captions"]["sha256"] == hashlib.sha256(
        captions.read_bytes()
    ).hexdigest()
    assert report["retrieval_evaluation"]["count"] == 4
    assert report["quality_gates"]["passed"]
    assert report["software"]["git_commit"] == "test-commit"
