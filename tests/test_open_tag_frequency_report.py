from scripts.visualization.make_open_tag_frequency_report import attribute_statistics


def test_attribute_statistics_reports_image_class_and_balanced_support():
    rows = [
        {"image_id": "apple/a.jpg", "attributes": ["red", "round"]},
        {"image_id": "apple/b.jpg", "attributes": ["red"]},
        {"image_id": "ball/a.jpg", "attributes": ["round"]},
    ]

    statistics = {row["attribute"]: row for row in attribute_statistics(rows)}

    assert statistics["red"] == {
        "attribute": "red",
        "images": 2,
        "image_prevalence": 2 / 3,
        "classes": 1,
        "class_coverage": 1 / 2,
        "balanced_prevalence": 1 / 2,
        "rank": 2,
    }
    assert statistics["round"] == {
        "attribute": "round",
        "images": 2,
        "image_prevalence": 2 / 3,
        "classes": 2,
        "class_coverage": 1.0,
        "balanced_prevalence": 3 / 4,
        "rank": 1,
    }
