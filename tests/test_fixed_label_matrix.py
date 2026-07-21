import pytest

from scripts.visualization.make_fixed_label_matrix import validate_fixed_rows


def test_fixed_rows_are_validated_and_ordered():
    dictionary = [{"name": "red"}, {"name": "round"}]
    rows = [
        {"image_id": "b.jpg", "status": "ok", "present": ["round"], "uncertain": []},
        {"image_id": "a.jpg", "status": "ok", "present": ["red"], "uncertain": []},
    ]

    names, ordered = validate_fixed_rows(dictionary, rows, ["a.jpg", "b.jpg"])

    assert names == ["red", "round"]
    assert [row["image_id"] for row in ordered] == ["a.jpg", "b.jpg"]


@pytest.mark.parametrize(
    "row",
    [
        {"image_id": "a.jpg", "status": "failed", "present": []},
        {"image_id": "a.jpg", "status": "ok", "present": ["unknown"]},
        {"image_id": "a.jpg", "status": "ok", "present": [], "uncertain": ["red"]},
    ],
)
def test_fixed_rows_reject_non_boolean_or_unknown_labels(row):
    with pytest.raises(ValueError):
        validate_fixed_rows([{"name": "red"}], [row], ["a.jpg"])
