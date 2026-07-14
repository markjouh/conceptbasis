import json

import pytest

from scripts.data.label_dictionary_concepts import check_or_write_meta, parse_response


CONCEPTS = {"metallic", "shiny", "red"}


def test_parse_response_accepts_fenced_exact_names_and_deduplicates():
    parsed = parse_response(
        '```json\n{"yes":["metallic","metallic"],'
        '"uncertain":["shiny","metallic","not-a-concept"]}\n```',
        CONCEPTS,
    )

    assert parsed == (["metallic"], ["shiny"], ["not-a-concept"])


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"yes":"metallic","uncertain":[]}',
        '{"yes":[],"uncertain":["shiny"]}',
    ],
)
def test_parse_response_rejects_unusable_answers(content):
    assert parse_response(content, CONCEPTS) is None


def test_metadata_refuses_incompatible_resume(tmp_path):
    meta_path = tmp_path / "labels.jsonl.meta.json"
    expected = {
        "schema_version": 1,
        "prompt_version": "v1",
        "prompt_sha256": "prompt-a",
        "dictionary_sha256": "dictionary-a",
        "split_manifest_sha256": "split-a",
        "selection_sha256": "selection-a",
        "source_image_count": 2,
        "model": "gemma",
        "temperature": 0,
        "reasoning_effort": "none",
        "max_output_tokens": 1600,
        "max_image_side": 512,
        "split": "train",
        "image_dir": "images",
    }
    check_or_write_meta(meta_path, expected)

    changed = dict(expected, selection_sha256="selection-b")
    with pytest.raises(ValueError, match="incompatible resume"):
        check_or_write_meta(meta_path, changed)


def test_metadata_refuses_orphaned_existing_output(tmp_path):
    output_path = tmp_path / "labels.jsonl"
    output_path.write_text(json.dumps({"image_id": "one.jpg"}) + "\n")

    with pytest.raises(ValueError, match="without metadata sidecar"):
        check_or_write_meta(
            output_path.with_suffix(".jsonl.meta.json"),
            {},
            output_path=output_path,
        )


def test_metadata_accepts_legacy_missing_default_fields(tmp_path):
    meta_path = tmp_path / "labels.jsonl.meta.json"
    legacy = {
        "schema_version": 1,
        "prompt_version": "v1",
        "prompt_sha256": "prompt-a",
        "dictionary_sha256": "dictionary-a",
        "split_manifest_sha256": "split-a",
        "selection_sha256": "selection-a",
        "source_image_count": 2,
        "model": "gemma",
        "temperature": 0,
        "reasoning_effort": "none",
        "max_output_tokens": 1600,
        "max_image_side": 512,
        "split": "train",
        "image_dir": "images",
    }
    meta_path.write_text(json.dumps(legacy))

    expected = dict(
        legacy,
        top_p=None,
        repeat_penalty=None,
        retry_repeat_penalty_step=0.04,
        structured_output=False,
    )
    check_or_write_meta(meta_path, expected)
