import json

import pytest

from scripts.data.label_fixed_dictionary import (
    build_named_binary_prompt,
    build_prompt,
    check_or_write_meta,
    named_binary_regex,
    parse_named_binary_response,
    parse_response,
)


CONCEPTS = {"metallic", "shiny", "red"}


def test_build_prompt_only_exposes_canonical_dictionary_leaders():
    prompt = build_prompt(
        ["red", "domestic"],
        "chair",
    )

    assert "- red\n" in prompt
    assert "- domestic\n" in prompt
    assert "member phrases used to construct the dictionary" in prompt
    assert "dark red" not in prompt
    assert "pink" not in prompt
    assert prompt.endswith("The labeled main object is: chair")


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


def test_named_binary_prompt_and_parser_preserve_exact_dictionary_order():
    concepts = ["red", "metallic", "shiny"]
    prompt = build_named_binary_prompt(concepts, "chair")

    assert "- red\n- metallic\n- shiny\n" in prompt
    assert "exact concept name: YES" in prompt
    assert "ignore the background, other objects" in prompt
    assert "source, contents, or an associated whole" in prompt
    assert prompt.endswith("The labeled main object is: chair")
    assert parse_named_binary_response(
        "red: YES\nmetallic: NO\nshiny: YES",
        concepts,
    ) == (["red", "shiny"], [], [])
    assert named_binary_regex(concepts) == (
        "red: (YES|NO)\nmetallic: (YES|NO)\nshiny: (YES|NO)"
    )


@pytest.mark.parametrize(
    "content",
    [
        "1. red: YES\nmetallic: NO\nshiny: YES",
        "metallic: NO\nred: YES\nshiny: YES",
        "red: YES\nmetallic: NO",
        "red YES\nmetallic: NO\nshiny: YES",
    ],
)
def test_named_binary_parser_rejects_noncanonical_checklists(content):
    assert parse_named_binary_response(content, ["red", "metallic", "shiny"]) is None


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


def test_metadata_refuses_changed_run_id(tmp_path):
    meta_path = tmp_path / "labels.jsonl.meta.json"
    expected = {
        "schema_version": 1,
        "run_id": "gemma-quality-v280",
    }
    check_or_write_meta(meta_path, expected)

    with pytest.raises(ValueError, match="incompatible resume"):
        check_or_write_meta(meta_path, dict(expected, run_id="qwen36-maxpx589824"))


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
