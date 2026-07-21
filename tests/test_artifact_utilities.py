import json

import pytest

from scripts.data.merge_fixed_label_shards import read_rows
from scripts.evaluation.summarize_seeded_composability import (
    mean_std,
    parse_history_specs,
)


def test_merge_reader_rejects_unsuccessful_rows(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_text(json.dumps({"status": "error"}) + "\n")

    with pytest.raises(ValueError, match="non-success row"):
        read_rows(path)


def test_seed_summary_uses_sample_standard_deviation():
    result = mean_std([1.0, 3.0])

    assert result["mean"] == 2.0
    assert result["sample_std"] == pytest.approx(2**0.5)


def test_history_specs_require_seed_placeholder_and_unique_names():
    assert parse_history_specs(["model=run-s{seed}/history.json"]) == {
        "model": "run-s{seed}/history.json"
    }
    with pytest.raises(ValueError, match="PATH_WITH"):
        parse_history_specs(["model=history.json"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_history_specs(["model=a{seed}", "model=b{seed}"])
