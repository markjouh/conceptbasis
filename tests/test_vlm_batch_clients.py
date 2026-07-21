import base64
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from conceptbasis.vlm import (
    image_reference,
    load_completed_image_ids,
    recursive_image_paths,
)
from scripts.data import caption_images, label_fixed_dictionary, mine_open_tags


VALID_CAPTION = (
    "The smooth red metal object has a rounded body with two narrow handles "
    "and a slightly worn reflective surface."
)
VALID_ATTRIBUTES = [
    "red",
    "metallic",
    "smooth",
    "rounded",
    "rigid",
    "portable",
    "handheld",
    "manmade",
    "reflective",
    "solid",
]
ROOT = Path(__file__).resolve().parents[1]


def test_shared_image_transport_preserves_files_and_bounds_base64(tmp_path):
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1200, 600), (10, 20, 30)).save(image_path)

    assert image_reference(
        image_path, transport="file", max_side=768
    ) == image_path.resolve().as_uri()
    encoded = image_reference(image_path, transport="base64", max_side=768)
    with Image.open(io.BytesIO(base64.b64decode(encoded.partition(",")[2]))) as image:
        assert image.size == (768, 384)


def test_shared_image_discovery_and_resume_loading_are_strict(tmp_path):
    image_dir = tmp_path / "images"
    (image_dir / "b").mkdir(parents=True)
    (image_dir / "a").mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "b" / "two.png")
    Image.new("RGB", (2, 2)).save(image_dir / "a" / "one.jpg")
    (image_dir / "a" / "ignore.txt").write_text("not an image")
    discovered = [
        path.relative_to(image_dir).as_posix()
        for path in recursive_image_paths(image_dir)
    ]
    assert discovered == ["a/one.jpg", "b/two.png"]

    output = tmp_path / "rows.jsonl"
    output.write_text(
        json.dumps({"image_id": "a/one.jpg", "status": "ok"}) + "\n"
        + json.dumps({"image_id": "b/two.png", "status": "error"}) + "\n"
    )
    assert load_completed_image_ids(output, required_status="ok") == {"a/one.jpg"}


def test_shared_resume_loader_rejects_duplicate_successes(tmp_path):
    output = tmp_path / "rows.jsonl"
    row = json.dumps({"image_id": "same.jpg", "caption": "caption"}) + "\n"
    output.write_text(row + row)
    with pytest.raises(ValueError, match="duplicate completed"):
        load_completed_image_ids(output, required_field="caption")


class StubResponse:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ]
        }


class RecordingSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.max_tokens = []
        self.prompts = []

    def post(self, _url, *, headers, json, timeout):
        del headers, timeout
        self.max_tokens.append(json["max_tokens"])
        self.prompts.append(json["messages"][1]["content"][0]["text"])
        return next(self.responses)


@pytest.mark.parametrize(
    "module", [caption_images, mine_open_tags, label_fixed_dictionary]
)
def test_file_transport_requires_loopback_endpoint(module):
    for endpoint in (
        "http://localhost:8000/v1/chat/completions",
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://[::1]:8000/v1/chat/completions",
    ):
        module.validate_image_transport(endpoint, "file")

    with pytest.raises(ValueError, match="requires a localhost"):
        module.validate_image_transport(
            "https://api.example.com/v1/chat/completions", "file"
        )
    module.validate_image_transport(
        "https://api.example.com/v1/chat/completions", "base64"
    )


@pytest.mark.parametrize("module", [caption_images, mine_open_tags])
def test_cli_rejects_remote_file_transport(module, monkeypatch, capsys):
    monkeypatch.setattr(module, "API_URL", "https://api.example.com/v1/chat/completions")
    monkeypatch.setattr(
        sys,
        "argv",
        [module.__name__, "--image-transport", "file"],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert "requires a localhost" in capsys.readouterr().err


def test_fixed_label_cli_rejects_remote_file_transport(monkeypatch):
    monkeypatch.setattr(
        label_fixed_dictionary.argparse.ArgumentParser,
        "parse_args",
        lambda _self: SimpleNamespace(
            split="train",
            allow_test=False,
            run_id="test",
            workers=1,
            max_side=768,
            max_output_tokens=16,
            retries=1,
            top_p=None,
            repeat_penalty=None,
            retry_repeat_penalty_step=0.0,
            api_url="https://api.example.com/v1/chat/completions",
            image_transport="file",
        ),
    )

    with pytest.raises(ValueError, match="requires a localhost"):
        label_fixed_dictionary.main()


def test_run_metadata_contains_inference_provenance():
    caption_meta = caption_images.build_run_metadata(
        "caption-run",
        image_transport="file",
        max_output_tokens=64,
        retry_max_output_tokens=90,
        image_dir="data/raw/object_images",
        selected_ids=["class/image-1.jpg", "class/image-2.jpg"],
        selection_seed=7,
    )
    attribute_meta = mine_open_tags.build_run_metadata(
        "attribute-run",
        image_transport="file",
        max_output_tokens=96,
        retry_max_output_tokens=300,
        image_dir="data/raw/object_images_CC0",
        split="train",
        split_manifest="data/splits.json",
        selected_ids=["class/image-1.jpg", "class/image-2.jpg"],
        selection_seed=7,
    )

    required = {
        "run_id",
        "model",
        "api_url",
        "model_revision",
        "server_profile",
        "prompt_sha256",
        "image_transport",
        "max_image_side",
        "max_output_tokens",
        "retry_max_output_tokens",
        "client_transform",
        "source_image_count",
        "selection_seed",
        "selection_sha256",
    }
    assert required <= caption_meta.keys()
    assert required <= attribute_meta.keys()
    assert len(caption_meta["prompt_sha256"]) == 64
    assert len(attribute_meta["prompt_sha256"]) == 64
    assert caption_meta["max_image_side"] is None
    assert attribute_meta["max_image_side"] is None
    assert caption_meta["client_transform"] == "none"
    assert attribute_meta["client_transform"] == "none"
    assert caption_meta["source_image_count"] == 2
    assert attribute_meta["source_image_count"] == 2


def test_caption_prompt_variant_changes_provenance_hash():
    common = dict(
        run_id="caption-run",
        image_transport="file",
        max_output_tokens=64,
        retry_max_output_tokens=90,
        image_dir="data/raw/object_images",
        selected_ids=["class/image.jpg"],
        selection_seed=0,
    )
    current = caption_images.build_run_metadata(
        **common,
        prompt_template=caption_images.LEGACY_PROMPT,
        min_words=15,
    )
    grounded = caption_images.build_run_metadata(
        **common,
        prompt_template=caption_images.CLIP_GROUNDED_PROMPT,
        min_words=8,
    )
    assert current["prompt_sha256"] != grounded["prompt_sha256"]
    assert current["caption_word_bounds"] == [15, 30]
    assert grounded["caption_word_bounds"] == [8, 30]


@pytest.mark.parametrize("module", [caption_images, mine_open_tags])
def test_run_metadata_guards_nonempty_resume(module, tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text('{"image_id":"old.jpg"}\n')
    expected = {"run_id": "run-a", "model": "model-a"}

    with pytest.raises(ValueError, match="without metadata sidecar"):
        module.check_or_write_run_metadata(output, expected)

    output.write_text("")
    meta_path = module.check_or_write_run_metadata(output, expected)
    assert json.loads(meta_path.read_text()) == expected

    output.write_text('{"image_id":"one.jpg"}\n')
    module.check_or_write_run_metadata(output, expected)
    with pytest.raises(ValueError, match="metadata differs"):
        module.check_or_write_run_metadata(output, dict(expected, model="model-b"))


def wrapper_args(script: str, extra_env: dict[str, str] | None = None) -> list[str]:
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("VLM_") or key == "CONCEPTBASIS_PYTHON":
            environment.pop(key)
    environment["CONCEPTBASIS_PYTHON"] = "/bin/echo"
    environment.update(extra_env or {})
    result = subprocess.run(
        [str(ROOT / "scripts" / "vllm" / script)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return shlex.split(result.stdout)


@pytest.mark.parametrize(
    ("script", "expected_run_id"),
    [
        (
            "caption_images.sh",
            "gemma4-26b-a4b-nvfp4-full-clip-grounded-caption-v2",
        ),
        (
            "mine_open_tags.sh",
            "gemma4-26b-a4b-nvfp4-full-train-open-tags-nonredundant-v8",
        ),
        (
            "label_fixed_dictionary.sh",
            (
                "gemma4-26b-a4b-nvfp4-quality-v280-file-usage-profile-v8-"
                "object-grounded-v11"
            ),
        ),
    ],
)
def test_wrappers_pass_explicit_profile_run_ids(script, expected_run_id):
    arguments = wrapper_args(script)
    assert arguments[arguments.index("--run-id") + 1] == expected_run_id


@pytest.mark.parametrize(
    ("script", "expected_workers"),
    [
        ("caption_images.sh", "80"),
        ("mine_open_tags.sh", "80"),
        ("label_fixed_dictionary.sh", "80"),
    ],
)
def test_wrappers_pass_tuned_client_concurrency(script, expected_workers):
    arguments = wrapper_args(script)
    assert arguments[arguments.index("--workers") + 1] == expected_workers


@pytest.mark.parametrize(
    "script", ["caption_images.sh", "mine_open_tags.sh", "label_fixed_dictionary.sh"]
)
def test_wrappers_select_the_same_gemma_endpoint_and_model(script):
    arguments = wrapper_args(script)
    assert arguments[arguments.index("--api-url") + 1] == (
        "http://127.0.0.1:8000/v1/chat/completions"
    )
    assert arguments[arguments.index("--model") + 1] == (
        "nvidia/Gemma-4-26B-A4B-NVFP4"
    )


@pytest.mark.parametrize(
    ("script", "specific_variable"),
    [
        ("caption_images.sh", "VLM_CAPTION_MAX_OUTPUT_TOKENS"),
        ("mine_open_tags.sh", "VLM_ATTRIBUTE_MAX_OUTPUT_TOKENS"),
        ("label_fixed_dictionary.sh", "VLM_LABEL_MAX_OUTPUT_TOKENS"),
    ],
)
def test_task_specific_output_token_setting_precedes_legacy_fallback(
    script, specific_variable
):
    legacy_arguments = wrapper_args(script, {"VLM_MAX_OUTPUT_TOKENS": "71"})
    assert legacy_arguments[legacy_arguments.index("--max-output-tokens") + 1] == "71"

    specific_arguments = wrapper_args(
        script,
        {"VLM_MAX_OUTPUT_TOKENS": "71", specific_variable: "72"},
    )
    assert specific_arguments[specific_arguments.index("--max-output-tokens") + 1] == "72"


def test_fixed_label_wrapper_uses_named_binary_checklist():
    arguments = wrapper_args("label_fixed_dictionary.sh")

    assert arguments[arguments.index("--review-mode") + 1] == "named_binary"
    assert "--no-structured-output" in arguments
    assert arguments[arguments.index("--max-output-tokens") + 1] == "1350"


@pytest.mark.parametrize(
    ("script", "specific_variable"),
    [
        ("caption_images.sh", "VLM_CAPTION_RETRY_MAX_OUTPUT_TOKENS"),
        ("mine_open_tags.sh", "VLM_ATTRIBUTE_RETRY_MAX_OUTPUT_TOKENS"),
    ],
)
def test_task_specific_retry_token_setting_precedes_legacy_fallback(
    script, specific_variable
):
    arguments = wrapper_args(
        script,
        {"VLM_RETRY_MAX_OUTPUT_TOKENS": "301", specific_variable: "302"},
    )
    assert arguments[arguments.index("--retry-max-output-tokens") + 1] == "302"


def test_caption_validator_enforces_prompt_shape():
    assert caption_images.valid_caption(VALID_CAPTION)
    assert caption_images.valid_caption(
        "A black 3.5-inch diskette has a rectangular plastic shell and a "
        "silver sliding metal shutter."
    )
    assert caption_images.valid_caption(
        "A silver U.S. Coast Guard airboat has a black propeller cage and "
        "rests on a sandy shoreline."
    )
    assert not caption_images.valid_caption("A short red metal object.")
    assert not caption_images.valid_caption(" ".join(["word"] * 31) + ".")
    assert not caption_images.valid_caption(
        "The smooth red metal object has a rounded body. "
        "It also has two narrow handles and a worn reflective surface."
    )
    assert not caption_images.valid_caption(
        "The smooth red metal object has a rounded body.It also has two narrow "
        "handles and a worn reflective surface."
    )
    assert not caption_images.valid_caption(
        "The smooth red metal object has a rounded body with two narrow handles "
        "and a slightly worn reflective surface"
    )


def test_caption_retries_truncation_with_larger_token_budget(monkeypatch):
    client = RecordingSession(
        [
            StubResponse(VALID_CAPTION, finish_reason="length"),
            StubResponse(VALID_CAPTION),
        ]
    )
    monkeypatch.setattr(caption_images, "session", lambda: client)
    monkeypatch.setattr(caption_images, "encode_image", lambda *_args: "image")

    result = caption_images.caption_one(
        "key",
        "image.jpg",
        "object",
        retries=2,
        max_output_tokens=16,
        retry_max_output_tokens=64,
    )

    assert result == VALID_CAPTION
    assert client.max_tokens == [16, 64]


def test_attribute_validator_enforces_unique_count_and_word_limits():
    assert mine_open_tags.validated_attributes(json.dumps(VALID_ATTRIBUTES)) == VALID_ATTRIBUTES
    assert mine_open_tags.validated_attributes(json.dumps(["red"])) is None
    assert mine_open_tags.validated_attributes(json.dumps([])) is None
    thirty = [
        f"tag {chr(97 + index // 26)}{chr(97 + index % 26)}"
        for index in range(30)
    ]
    assert mine_open_tags.validated_attributes(json.dumps(thirty)) == thirty
    assert mine_open_tags.validated_attributes(json.dumps(thirty + ["tag zz"])) == (
        thirty + ["tag zz"]
    )
    fifty_one = [
        f"tag {chr(97 + index // 26)}{chr(97 + index % 26)}"
        for index in range(51)
    ]
    assert mine_open_tags.validated_attributes(json.dumps(fifty_one)) is None


def test_attribute_validator_removes_explicit_negations():
    values = [
        "red", "smooth", "natural", "organic", "soft", "rounded",
        "manmade-absent", "not metallic", "without handles", "sugar-free",
    ]
    assert mine_open_tags.validated_attributes(json.dumps(values)) == [
        "red", "smooth", "natural", "organic", "soft", "rounded",
    ]


def test_open_tag_prompt_requires_unambiguous_non_vacuous_properties():
    assert "UNAMBIGUOUS" in mine_open_tags.PROMPT
    assert "NON-VACUOUS" in mine_open_tags.PROMPT
    assert "solid" in mine_open_tags.PROMPT
    assert "three-dimensional" in mine_open_tags.PROMPT
    assert "surface" not in mine_open_tags.PROMPT
    assert "fine" not in mine_open_tags.PROMPT
    assert "detailed" not in mine_open_tags.PROMPT
    assert "functional" not in mine_open_tags.PROMPT
    assert "..." in mine_open_tags.PROMPT
    assert "good examples, ...:" in mine_open_tags.PROMPT
    assert "bad examples, ...:" in mine_open_tags.PROMPT
    assert "mutually non-redundant" in mine_open_tags.PROMPT
    assert "do not also list 'blue denim'" in mine_open_tags.PROMPT
    assert "never add" in mine_open_tags.PROMPT
    assert "5-50" in mine_open_tags.PROMPT
    assert "*-absent" in mine_open_tags.PROMPT
    assert "If a property does not apply, omit it" in mine_open_tags.PROMPT
    assert "6-30" not in mine_open_tags.PROMPT


@pytest.mark.parametrize(
    ("image_id", "expected"),
    [
        ("stopwatch.jpg", "stopwatch"),
        ("stopwatch/stopwatch_07s.jpg", "stopwatch"),
        ("bow2.jpg", "bow"),
        ("bow2/bow2_03s.jpg", "bow"),
        ("air_conditioner/air_conditioner_01b.jpg", "air conditioner"),
    ],
)
def test_attribute_subject_comes_from_object_class(image_id, expected):
    assert mine_open_tags.label_from_id(image_id) == expected
    assert mine_open_tags.validated_attributes(
        json.dumps(["very dark blue"])
    ) is None


def test_open_tag_wrapper_budget_supports_long_tag_lists():
    arguments = wrapper_args("mine_open_tags.sh")
    assert arguments[arguments.index("--max-output-tokens") + 1] == "192"


def test_attribute_client_retries_invalid_count_with_larger_budget(monkeypatch):
    client = RecordingSession(
        [
            StubResponse(json.dumps([])),
            StubResponse(json.dumps(VALID_ATTRIBUTES)),
        ]
    )
    monkeypatch.setattr(mine_open_tags, "session", lambda: client)
    monkeypatch.setattr(mine_open_tags, "encode_image", lambda *_args: "image")

    result = mine_open_tags.attrs_one(
        "key",
        "image.jpg",
        "object",
        retries=2,
        max_output_tokens=32,
        retry_max_output_tokens=96,
    )

    assert result == VALID_ATTRIBUTES
    assert client.max_tokens == [32, 96]
    assert mine_open_tags.RETRY_PROMPT not in client.prompts[0]
    assert mine_open_tags.RETRY_PROMPT in client.prompts[1]


def test_caption_resume_rate_counts_only_current_work(tmp_path, monkeypatch, capsys):
    image_dir = tmp_path / "data" / "raw" / "object_images" / "apple"
    image_dir.mkdir(parents=True)
    (image_dir / "apple1.jpg").write_bytes(b"")
    output = tmp_path / "data" / "captions.jsonl"
    output.write_text(json.dumps({"image_id": "old.jpg", "caption": VALID_CAPTION}) + "\n")

    monkeypatch.setattr(caption_images, "ROOT", str(tmp_path))
    monkeypatch.setattr(caption_images, "caption_one", lambda *_args, **_kwargs: VALID_CAPTION)
    monkeypatch.setattr(
        caption_images,
        "time",
        SimpleNamespace(monotonic=iter((10.0, 12.0)).__next__),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["caption_images.py", "--out", "data/captions.jsonl", "--workers", "1"],
    )

    caption_images.main()

    assert "effective_images_per_second=0.500" in capsys.readouterr().out


def test_attribute_resume_rate_counts_only_current_work(tmp_path, monkeypatch, capsys):
    image_dir = tmp_path / "data" / "raw" / "object_images_CC0"
    image_dir.mkdir(parents=True)
    (image_dir / "apple.jpg").write_bytes(b"")
    (tmp_path / "data" / "splits.json").write_text(
        json.dumps({"unit": "things_object_class", "classes": {"apple": "train"}})
    )
    output = tmp_path / "data" / "attributes.jsonl"
    output.write_text(
        json.dumps({"image_id": "old.jpg", "attributes": VALID_ATTRIBUTES}) + "\n"
    )

    monkeypatch.setattr(mine_open_tags, "ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        mine_open_tags, "attrs_one", lambda *_args, **_kwargs: VALID_ATTRIBUTES
    )
    monkeypatch.setattr(
        mine_open_tags,
        "time",
        SimpleNamespace(monotonic=iter((20.0, 22.0)).__next__),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_open_tags.py",
            "--out",
            "data/attributes.jsonl",
            "--workers",
            "1",
        ],
    )

    mine_open_tags.main()

    output_text = capsys.readouterr().out
    assert "completed=1 failed=0" in output_text
    assert "effective_images_per_second=0.500" in output_text


def test_attribute_failures_are_recorded_outside_main_output(
    tmp_path, monkeypatch, capsys
):
    image_dir = tmp_path / "data" / "raw" / "object_images" / "apple"
    image_dir.mkdir(parents=True)
    (image_dir / "apple_01s.jpg").write_bytes(b"")
    (tmp_path / "data" / "splits.json").write_text(
        json.dumps({"unit": "things_object_class", "classes": {"apple": "train"}})
    )

    monkeypatch.setattr(mine_open_tags, "ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(mine_open_tags, "attrs_one", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_open_tags.py",
            "--img-dir",
            "data/raw/object_images",
            "--out",
            "data/attributes.jsonl",
            "--workers",
            "1",
        ],
    )

    mine_open_tags.main()

    assert (tmp_path / "data" / "attributes.jsonl").read_text() == ""
    error_row = json.loads(
        (tmp_path / "data" / "attributes.jsonl.errors.jsonl").read_text()
    )
    assert error_row == {
        "image_id": "apple/apple_01s.jpg",
        "error": "no_valid_attributes_after_retries",
    }
    assert "completed=0 failed=1" in capsys.readouterr().out
