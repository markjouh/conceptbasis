import json

import numpy as np
import pytest

from conceptbasis.encoders import (
    MANIFEST_NAME,
    PROJECT_ENCODER,
    SIGLIP2_GIANT,
    EncoderSpec,
    EncoderCacheMismatch,
    cache_identity,
    encoder_output_dir,
    load_open_clip_encoder,
    release_visual_tower,
    save_npy_atomic,
    select_encoder,
    validate_cache_manifest,
    write_json_atomic,
    write_cache_manifest,
)


def test_release_visual_tower_is_safe_for_text_only_models():
    class Model:
        visual = object()

    model = Model()
    release_visual_tower(model, "cpu")
    assert model.visual is None


def test_atomic_artifact_writes_leave_only_complete_targets(tmp_path):
    array_path = tmp_path / "embeddings.npy"
    json_path = tmp_path / "ids.json"
    expected = np.eye(3, dtype=np.float32)

    save_npy_atomic(array_path, expected)
    write_json_atomic(json_path, ["a.jpg", "b.jpg", "c.jpg"])

    np.testing.assert_array_equal(np.load(array_path), expected)
    assert json.loads(json_path.read_text()) == ["a.jpg", "b.jpg", "c.jpg"]
    assert not list(tmp_path.glob("*.tmp"))


def test_siglip2_giant_preset_is_pinned_and_isolated(tmp_path):
    spec = select_encoder("siglip2-giant")

    assert spec == SIGLIP2_GIANT
    assert spec.model == "ViT-gopt-16-SigLIP2-384"
    assert spec.pretrained == "webli"
    assert len(spec.revision) == 40
    assert encoder_output_dir(tmp_path, spec) == (
        tmp_path
        / "outputs"
        / "encoder_candidates"
        / "siglip2-gopt-p16-384@ad3410b"
    ).resolve()


def test_custom_encoder_requires_model_and_pretrained_together():
    with pytest.raises(ValueError, match="supplied together"):
        select_encoder("project", model="custom")

    custom = select_encoder(
        "project",
        model="custom-model",
        pretrained="custom-release",
        revision="abc123",
    )
    assert custom.model == "custom-model"
    assert custom.pretrained == "custom-release"
    assert custom.revision == "abc123"


def test_candidate_cannot_target_production_data_tree(tmp_path):
    for requested in ("data", "data/candidates/siglip"):
        with pytest.raises(ValueError, match="cannot write to data"):
            encoder_output_dir(tmp_path, SIGLIP2_GIANT, requested)

    assert encoder_output_dir(tmp_path, PROJECT_ENCODER) == (tmp_path / "data").resolve()


def test_manifest_guards_identity_and_artifact_checksums(tmp_path):
    dictionary = tmp_path / "dictionary.json"
    splits = tmp_path / "splits.json"
    dictionary.write_text('[{"name":"red"}]\n')
    splits.write_text('{"classes":{}}\n')
    output = encoder_output_dir(tmp_path, SIGLIP2_GIANT)
    output.mkdir(parents=True)
    identity = cache_identity(
        SIGLIP2_GIANT,
        precision="fp16",
        image_ids=["a/1.jpg", "b/2.jpg"],
        cc0_image_ids=["cc0.jpg"],
        dictionary_path=dictionary,
        split_manifest_path=splits,
    )
    np.save(output / "image_embeddings.npy", np.eye(2, dtype=np.float32))
    (output / "image_ids.json").write_text(json.dumps(["a/1.jpg", "b/2.jpg"]) + "\n")

    with pytest.raises(EncoderCacheMismatch, match="unmanifested"):
        validate_cache_manifest(output, identity)

    source = {
        "hf_revision": SIGLIP2_GIANT.revision,
        "preprocess": {"mean": (0.5, 0.5, 0.5)},
    }
    manifest_path = write_cache_manifest(output, identity, source=source)
    manifest = validate_cache_manifest(output, identity, source=source)
    assert manifest_path == output / MANIFEST_NAME
    assert manifest["artifacts"]["image_embeddings.npy"]["shape"] == [2, 2]
    assert manifest["artifacts"]["image_embeddings.npy"]["dtype"] == "float32"
    assert manifest["source"]["preprocess"]["mean"] == [0.5, 0.5, 0.5]

    changed = dict(identity)
    changed["precision"] = "fp32"
    with pytest.raises(EncoderCacheMismatch, match="identity"):
        validate_cache_manifest(output, changed, verify_hashes=False)

    with pytest.raises(EncoderCacheMismatch, match="runtime/preprocess"):
        validate_cache_manifest(
            output,
            identity,
            source={"hf_revision": "different"},
            verify_hashes=False,
        )

    np.save(output / "image_embeddings.npy", np.ones((2, 2), dtype=np.float32))
    with pytest.raises(EncoderCacheMismatch, match="checksum"):
        validate_cache_manifest(output, identity)


def test_pinned_loader_uses_safe_checkpoint_and_same_tokenizer_revision(
    tmp_path, monkeypatch
):
    import open_clip
    import open_clip.pretrained

    calls = {}

    class Model:
        def eval(self):
            calls["eval"] = True

    checkpoint = tmp_path / "open_clip_model.safetensors"
    checkpoint.write_bytes(b"safe")
    monkeypatch.setattr(
        open_clip,
        "get_model_config",
        lambda _: {
            "embed_dim": 1536,
            "vision_cfg": {"image_size": 384},
            "text_cfg": {
                "context_length": 64,
                "hf_tokenizer_name": "timm/ViT-gopt-16-SigLIP2-384",
            },
        },
    )
    monkeypatch.setattr(
        open_clip,
        "get_pretrained_cfg",
        lambda *_: {
            "hf_hub": "timm/ViT-gopt-16-SigLIP2-384/",
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "interpolation": "bicubic",
            "resize_mode": "squash",
        },
    )

    def fake_download(repo_id, filename=None, revision=None, cache_dir=None):
        calls["download"] = (repo_id, filename, revision, cache_dir)
        return str(checkpoint)

    def fake_create(model, **kwargs):
        calls["create"] = (model, kwargs)
        return Model(), None, "preprocess"

    def fake_tokenizer(model, **kwargs):
        calls["tokenizer"] = (model, kwargs)
        return "tokenizer"

    monkeypatch.setattr(open_clip.pretrained, "download_pretrained_from_hf", fake_download)
    monkeypatch.setattr(open_clip, "create_model_and_transforms", fake_create)
    monkeypatch.setattr(open_clip, "get_tokenizer", fake_tokenizer)

    model, preprocess, tokenizer, source = load_open_clip_encoder(
        SIGLIP2_GIANT,
        device="cpu",
        precision="fp32",
        cache_dir=str(tmp_path / "cache"),
    )

    assert isinstance(model, Model)
    assert preprocess == "preprocess"
    assert tokenizer == "tokenizer"
    assert calls["download"][1] == "open_clip_model.safetensors"
    assert calls["download"][2] == SIGLIP2_GIANT.revision
    assert calls["create"][1]["pretrained"] == str(checkpoint)
    assert calls["create"][1]["image_resize_mode"] == "squash"
    assert calls["tokenizer"][1]["revision"] == SIGLIP2_GIANT.revision
    assert source["embedding_dimension"] == 1536
    assert source["preprocess"]["interpolation"] == "bicubic"


def test_pinned_loader_does_not_pass_revision_to_simple_tokenizer(
    tmp_path, monkeypatch
):
    import open_clip
    import open_clip.pretrained

    calls = {}

    class Model:
        def eval(self):
            pass

    checkpoint = tmp_path / "open_clip_model.safetensors"
    checkpoint.write_bytes(b"safe")
    monkeypatch.setattr(
        open_clip,
        "get_model_config",
        lambda _: {
            "embed_dim": 1280,
            "vision_cfg": {"image_size": 448},
            "text_cfg": {"context_length": 72},
        },
    )
    monkeypatch.setattr(
        open_clip,
        "get_pretrained_cfg",
        lambda *_: {"hf_hub": "timm/PE-Core-bigG-14-448/"},
    )
    monkeypatch.setattr(
        open_clip.pretrained,
        "download_pretrained_from_hf",
        lambda *_args, **_kwargs: str(checkpoint),
    )
    monkeypatch.setattr(
        open_clip,
        "create_model_and_transforms",
        lambda *_args, **_kwargs: (Model(), None, "preprocess"),
    )

    def fake_tokenizer(_model, **kwargs):
        calls.update(kwargs)
        return "tokenizer"

    monkeypatch.setattr(open_clip, "get_tokenizer", fake_tokenizer)
    load_open_clip_encoder(
        EncoderSpec(
            name="pe",
            model="PE-Core-bigG-14-448",
            pretrained="meta",
            revision="17aa0c25addfa14198fa2ff73d845a22d433432e",
        ),
        device="cpu",
        precision="fp32",
    )

    assert "revision" not in calls
