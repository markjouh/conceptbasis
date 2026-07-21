import numpy as np

from scripts.dictionary.build_dictionary import (
    attribute_support,
    candidate_phrases,
    class_mean_profiles,
    complete_linkage_labels,
    cosine_similarity_matrix,
    lexical_variant_similarity,
    standardize_and_debias,
)


def test_attribute_support_gives_each_class_equal_weight():
    rows = [
        {"image_id": "apple/a.jpg", "attributes": ["red", "round"]},
        {"image_id": "apple/b.jpg", "attributes": ["red"]},
        {"image_id": "ball/a.jpg", "attributes": ["round"]},
    ]

    support = attribute_support(rows)

    assert support["image_mentions"] == {"red": 2, "round": 2}
    assert support["class_support"] == {"red": 1, "round": 2}
    assert support["balanced_prevalence"]["red"] == 0.5
    assert support["balanced_prevalence"]["round"] == 0.75


def test_candidate_phrases_applies_support_and_excludes_negative_forms():
    rows = [
        {"image_id": "a/1.jpg", "attributes": ["rigid", "nonregular"]},
        {"image_id": "b/1.jpg", "attributes": ["rigid", "regular"]},
        {"image_id": "c/1.jpg", "attributes": ["rigid", "regular"]},
    ]

    phrases, negated_of, _support = candidate_phrases(
        rows, min_mentions=1, min_class_support=1
    )

    assert "rigid" in phrases
    assert "regular" in phrases
    assert "nonregular" not in phrases
    assert negated_of == {"nonregular": "regular"}


def test_class_mean_profiles_average_views_before_geometry():
    scores = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=np.float32)

    profiles, classes = class_mean_profiles(
        scores,
        ["apple/a.jpg", "apple/b.jpg", "ball/a.jpg"],
    )

    assert classes == ["apple", "ball"]
    np.testing.assert_allclose(profiles, [[2.0, 3.0], [10.0, 20.0]])


def test_standardize_and_debias_preserves_shape_and_finite_values():
    scores = np.array(
        [[1.0, 3.0, 2.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]],
        dtype=np.float32,
    )

    result = standardize_and_debias(scores, 1)

    assert result.shape == scores.shape
    assert np.isfinite(result).all()


def test_cosine_similarity_matrix_normalizes_rows():
    result = cosine_similarity_matrix(
        np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    )

    np.testing.assert_allclose(np.diag(result), 1.0)
    np.testing.assert_allclose(result[0, 1], 1 / np.sqrt(2), rtol=1e-6)


def test_consensus_merge_requires_profile_and_text_agreement():
    profile = np.array(
        [[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]],
        dtype=np.float32,
    )
    text = np.array(
        [[1.0, 0.98, 0.4], [0.98, 1.0, 0.4], [0.4, 0.4, 1.0]],
        dtype=np.float32,
    )

    labels = complete_linkage_labels(
        profile,
        text,
        method="profile-and-text",
        profile_threshold=0.5,
        text_threshold=0.95,
    )

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_lexical_variants_are_narrow_and_auditable():
    phrases = ["gray", "grey", "multi-colored", "multicolored", "rigid", "lightweight"]
    similarity = lexical_variant_similarity(phrases)

    assert similarity[0, 1] == 1
    assert similarity[2, 3] == 1
    assert similarity[4, 5] == 0


def test_average_adjudicated_linkage_allows_dense_non_clique_cluster():
    profile = np.eye(3, dtype=np.float32)
    text = np.eye(3, dtype=np.float32)
    adjudicated = np.array(
        [[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32
    )

    complete = complete_linkage_labels(
        profile,
        text,
        method="adjudicated",
        profile_threshold=0.5,
        text_threshold=0.5,
        adjudicated_similarity=adjudicated,
    )
    average = complete_linkage_labels(
        profile,
        text,
        method="adjudicated",
        profile_threshold=0.5,
        text_threshold=0.5,
        adjudicated_similarity=adjudicated,
        adjudicated_linkage="average",
        adjudicated_min_similarity=0.5,
    )

    assert len(set(complete)) == 2
    assert len(set(average)) == 1
