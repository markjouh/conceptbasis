from scripts.visualization.make_dictionary_gallery import diverse_examples


def test_dictionary_preview_prefers_distinct_object_classes():
    candidates = [
        ({"image_id": "apple/a.jpg"}, ["red"]),
        ({"image_id": "apple/b.jpg"}, ["red"]),
        ({"image_id": "ball/a.jpg"}, ["red"]),
        ({"image_id": "car/a.jpg"}, ["red"]),
    ]

    selected = diverse_examples(candidates, 3)

    assert [row["image_id"] for row, _matched in selected] == [
        "apple/a.jpg",
        "ball/a.jpg",
        "car/a.jpg",
    ]
