import numpy as np
import json

from scripts.evaluation.evaluate_compositional_retrieval import (
    build_rollouts,
    load_query_labels,
    model_eval,
    parse_rollout_schedule,
)


def test_load_query_labels_reads_exhaustive_present_field(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        {"image_id": "a.jpg", "present": ["rigid", "metallic"]},
        {"image_id": "b.jpg", "present": ["soft"]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = load_query_labels(str(path), "present")

    assert [row["image_id"] for row in loaded] == ["a.jpg", "b.jpg"]
    assert [row["query_concepts"] for row in loaded] == [
        ["rigid", "metallic"],
        ["soft"],
    ]


def test_rollout_schedule_uses_every_class_eligible_at_each_k():
    sizes = [1, 2, 4]
    schedule = parse_rollout_schedule(sizes, 2, "2:3,4:5")
    concepts = [
        np.array([0], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int32),
    ]
    rollout_data = build_rollouts(concepts, 8, sizes, schedule, seed=0)
    profiles = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = model_eval(
        profiles,
        rollout_data,
        sizes,
        schedule,
        batch=16,
    )["true_attributes"]

    assert schedule == {1: 2, 2: 3, 4: 5}
    assert [result[str(size)]["n_images"] for size in sizes] == [3, 2, 1]
    assert [result[str(size)]["n_rollout_queries"] for size in sizes] == [6, 6, 5]
