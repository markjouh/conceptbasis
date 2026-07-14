import numpy as np

from scripts.evaluation.eval_playground_subset_composability import (
    build_rollouts,
    model_eval,
    parse_rollout_schedule,
)


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
