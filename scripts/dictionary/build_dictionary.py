"""Stage 2 — Build a compact concept dictionary from train-split open tags.

The production recipe uses all tagged train exemplars, a pinned frozen encoder,
and class-mean phrase profiles.  Averaging image/text scores within each THINGS
class before clustering lets every class contribute equally while still using
all of its available views.  Phrase merging and near-twin rejection are
independently configurable so visual-profile, text-semantic, consensus, and
unmerged frequency baselines can be compared without changing the input data.

The usage-profile recipe reads dictionary-independent ``image_embeddings.npy``
from a pinned encoder cache and encodes candidate phrases on CUDA. It writes the
dictionary JSON plus a provenance sidecar; it never reads development or test
classes unless explicitly authorized.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from conceptbasis.encoders import (
    ENCODER_PRESETS,
    MANIFEST_SCHEMA,
    encoder_output_dir,
    load_open_clip_encoder,
    release_visual_tower,
    select_encoder,
    sha256_file,
    write_json_atomic,
)
from conceptbasis.splits import image_class, load_split_manifest, split_for_image
from conceptbasis.train import require_cuda


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TEXT_TEMPLATE = "an object that is {phrase}"


def autocast_context(device: str, precision: str):
    if device != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def recursive_image_ids(image_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(image_dir).as_posix()
        for path in image_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def attribute_support(rows: list[dict]) -> dict:
    """Return raw and equal-class-weighted support for every raw phrase."""
    class_totals = Counter(image_class(row["image_id"]) for row in rows)
    image_mentions = Counter()
    phrase_class_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        concept = image_class(row["image_id"])
        for phrase in set(row["attributes"]):
            image_mentions[phrase] += 1
            phrase_class_counts[phrase][concept] += 1
    n_classes = len(class_totals)
    class_support = {
        phrase: len(per_class) for phrase, per_class in phrase_class_counts.items()
    }
    balanced_prevalence = {
        phrase: sum(
            count / class_totals[concept]
            for concept, count in per_class.items()
        )
        / n_classes
        for phrase, per_class in phrase_class_counts.items()
    }
    return {
        "class_totals": class_totals,
        "image_mentions": image_mentions,
        "phrase_class_counts": phrase_class_counts,
        "class_support": class_support,
        "balanced_prevalence": balanced_prevalence,
    }


def candidate_phrases(
    rows: list[dict], *, min_mentions: int, min_class_support: int
) -> tuple[list[str], dict[str, str], dict]:
    """Return affirmative phrases, excluded lexical negations, and support.

    A basis coefficient denotes evidence for a named, affirmatively present
    concept.  Explicit lexical negations are therefore excluded rather than
    attached as a synthetic negative pole.
    """
    support = attribute_support(rows)
    counts = support["image_mentions"]
    phrases = sorted(
        phrase
        for phrase, count in counts.items()
        if count >= min_mentions
        and support["class_support"][phrase] >= min_class_support
    )
    prefixes = ("in", "un", "non", "anti", "dis")
    negated_of = {}
    phrase_set = set(phrases)
    for phrase in phrases:
        word = phrase.replace("-", "")
        for prefix in prefixes:
            if word.startswith(prefix) and word[len(prefix):] in phrase_set:
                negated_of[phrase] = word[len(prefix):]
                break
    return [phrase for phrase in phrases if phrase not in negated_of], negated_of, support


def validate_and_load_image_cache(
    embedding_dir: Path,
    encoder,
    image_dir: Path,
    selected_ids: list[str],
    *,
    verify_hashes: bool,
) -> tuple[np.ndarray, dict]:
    manifest_path = embedding_dir / "encoder_manifest.json"
    ids_path = embedding_dir / "image_ids.json"
    embeddings_path = embedding_dir / "image_embeddings.npy"
    if not manifest_path.exists() or not ids_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"complete encoder cache is required under {embedding_dir}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported encoder cache manifest: {manifest_path}")
    if manifest.get("identity", {}).get("encoder") != encoder.as_dict():
        raise ValueError("encoder cache release does not match requested encoder")
    source = manifest.get("source", {})
    if source.get("hf_revision") != encoder.revision:
        raise ValueError("encoder cache source revision does not match requested release")
    artifacts = manifest.get("artifacts", {})
    for name, path in (
        ("image_ids.json", ids_path),
        ("image_embeddings.npy", embeddings_path),
    ):
        record = artifacts.get(name)
        if record is None:
            raise ValueError(f"encoder manifest does not declare {name}")
        if verify_hashes and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"encoder cache checksum mismatch: {path}")

    cached_ids = json.loads(ids_path.read_text())
    directory_ids = recursive_image_ids(image_dir)
    if cached_ids != directory_ids:
        raise ValueError("encoder image-ID cache does not match --img-dir")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("attribute rows contain duplicate image IDs")
    index = {image_id: position for position, image_id in enumerate(cached_ids)}
    missing = [image_id for image_id in selected_ids if image_id not in index]
    if missing:
        raise ValueError(f"attribute images are missing from encoder cache: {missing[:5]}")
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != tuple(artifacts["image_embeddings.npy"].get("shape", ())):
        raise ValueError("encoder embedding shape does not match its manifest")
    selected = np.asarray(
        embeddings[[index[image_id] for image_id in selected_ids]],
        dtype=np.float32,
    )
    return selected, {
        "directory": str(embedding_dir.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "image_embeddings_sha256": artifacts["image_embeddings.npy"]["sha256"],
        "image_ids_sha256": artifacts["image_ids.json"]["sha256"],
        "source": source,
    }


@torch.inference_mode()
def encode_phrases(
    phrases: list[str],
    encoder,
    *,
    device: str,
    precision: str,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    started = time.monotonic()
    model, _preprocess, tokenizer, source = load_open_clip_encoder(
        encoder,
        device=device,
        precision=precision,
    )
    release_visual_tower(model, device)
    batches = []
    for start in range(0, len(phrases), batch_size):
        texts = [TEXT_TEMPLATE.format(phrase=phrase) for phrase in phrases[start:start + batch_size]]
        tokens = tokenizer(texts).to(device)
        with autocast_context(device, precision):
            encoded = torch.nn.functional.normalize(
                model.encode_text(tokens), dim=-1
            )
        batches.append(encoded.float().cpu().numpy().astype(np.float32))
    result = np.concatenate(batches)
    print(
        f"encoded {len(result)} phrases in {time.monotonic() - started:.3f}s",
        flush=True,
    )
    return result, source


def class_mean_profiles(scores: np.ndarray, image_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    classes = np.array([image_class(image_id) for image_id in image_ids])
    class_names = sorted(set(classes))
    profiles = np.empty((len(class_names), scores.shape[1]), dtype=np.float32)
    for index, concept in enumerate(class_names):
        profiles[index] = scores[classes == concept].mean(axis=0)
    return profiles, class_names


def standardize_and_debias(scores: np.ndarray, n_components: int) -> np.ndarray:
    scores = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-8)
    if n_components:
        left, singular_values, right = np.linalg.svd(scores, full_matrices=False)
        scores = scores - (
            left[:, :n_components] * singular_values[:n_components]
        ) @ right[:n_components]
    return scores.astype(np.float32, copy=False)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def lexical_variant_similarity(phrases: list[str]) -> np.ndarray:
    """Identify spelling, spacing, hyphenation, and simple plural variants.

    This deliberately does not attempt synonym discovery.  It is a
    high-precision alternative for experiments where false aliasing is more
    damaging than leaving ``wood`` and ``wooden`` as separate candidates.
    """
    compact = [re.sub(r"[\s_-]+", "", phrase.lower()).replace("grey", "gray")
               for phrase in phrases]
    vocabulary = set(compact)
    keys = []
    for value in compact:
        if value.endswith("ies") and value[:-3] + "y" in vocabulary:
            value = value[:-3] + "y"
        elif value.endswith("s") and not value.endswith("ss") and value[:-1] in vocabulary:
            value = value[:-1]
        keys.append(value)
    return np.equal.outer(keys, keys).astype(np.float32)


def complete_linkage_labels(
    profile_similarity: np.ndarray,
    text_similarity: np.ndarray,
    *,
    method: str,
    profile_threshold: float,
    text_threshold: float,
    lexical_similarity: np.ndarray | None = None,
    adjudicated_similarity: np.ndarray | None = None,
    adjudicated_linkage: str = "complete",
    adjudicated_min_similarity: float = 1.0,
) -> np.ndarray:
    """Cluster phrases while keeping the meaning of each threshold explicit.

    ``profile-and-text`` uses a normalized maximum distance.  Consequently all
    phrase pairs admitted to a complete-linkage cluster must pass both gates;
    a merely related pair cannot merge solely because its usage is similar.

    For adjudicated average linkage, accepted edges retain their proposal
    cosine and rejected or unproposed pairs have similarity zero.  This permits
    coherent non-cliques while making weakly connected cluster unions costly.
    """
    count = profile_similarity.shape[0]
    if profile_similarity.shape != (count, count):
        raise ValueError("profile similarity must be square")
    if text_similarity.shape != (count, count):
        raise ValueError("text similarity must match profile similarity")
    if method == "none":
        return np.arange(1, count + 1, dtype=np.int32)
    if method == "adjudicated":
        if (
            adjudicated_similarity is None
            or adjudicated_similarity.shape != (count, count)
        ):
            raise ValueError("adjudicated merge requires a matching decision matrix")
        distances = 1.0 - adjudicated_similarity
        cutoff = 1.0 - adjudicated_min_similarity
        linkage_method = adjudicated_linkage
    elif method == "lexical":
        if lexical_similarity is None or lexical_similarity.shape != (count, count):
            raise ValueError("lexical merge requires a matching similarity matrix")
        distances = 1.0 - lexical_similarity
        cutoff = 0.0
        linkage_method = "complete"
    elif method == "profile":
        distances = 1.0 - profile_similarity
        cutoff = 1.0 - profile_threshold
        linkage_method = "complete"
    elif method == "text":
        distances = 1.0 - text_similarity
        cutoff = 1.0 - text_threshold
        linkage_method = "complete"
    elif method == "profile-and-text":
        profile_scale = max(1.0 - profile_threshold, 1e-8)
        text_scale = max(1.0 - text_threshold, 1e-8)
        distances = np.maximum(
            (1.0 - profile_similarity) / profile_scale,
            (1.0 - text_similarity) / text_scale,
        )
        cutoff = 1.0
        linkage_method = "complete"
    else:
        raise ValueError(f"unknown merge method: {method}")
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    distances = (distances + distances.T) / 2

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    tree = linkage(squareform(distances, checks=False), method=linkage_method)
    return fcluster(tree, t=cutoff, criterion="distance")


def load_adjudicated_similarity(
    path: Path, phrases: list[str], *, weighted: bool = False
) -> tuple[np.ndarray, dict]:
    """Load a complete, phrase-bound set of LLM edge decisions."""
    artifact = json.loads(path.read_text())
    if artifact.get("schema") != "conceptbasis.merge-adjudication/v1":
        raise ValueError(f"unsupported merge adjudication schema: {path}")
    if artifact.get("phrases") != phrases:
        raise ValueError("merge adjudication phrase list does not match candidates")
    decisions = artifact.get("decisions", [])
    proposed = artifact.get("n_proposed_edges")
    if proposed != len(decisions):
        raise ValueError("merge adjudication is incomplete")
    index = {phrase: position for position, phrase in enumerate(phrases)}
    similarity = np.eye(len(phrases), dtype=np.float32)
    seen = set()
    for row in decisions:
        pair = (row["left"], row["right"])
        if pair in seen or pair[0] not in index or pair[1] not in index:
            raise ValueError(f"invalid or duplicate adjudication edge: {pair}")
        seen.add(pair)
        if row.get("equivalent") is True:
            left, right = index[pair[0]], index[pair[1]]
            value = float(row["text_cosine"]) if weighted else 1.0
            similarity[left, right] = similarity[right, left] = value
    return similarity, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--open-tags",
        "--attrs",
        dest="open_tags",
        default="data/attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl",
    )
    parser.add_argument("--img-dir", default="data/raw/object_images")
    parser.add_argument(
        "--out", default="data/dictionary_usage_profile_v8.json"
    )
    parser.add_argument(
        "--provenance-out",
        default="data/dictionary_usage_profile_v8.provenance.json",
    )
    parser.add_argument(
        "--cluster-candidates-out",
        help="optional pre-selection synonym-cluster artifact for downstream refinement",
    )
    parser.add_argument("--encoder", choices=tuple(ENCODER_PRESETS), default="siglip2-giant")
    parser.add_argument("--embedding-dir")
    parser.add_argument("--model")
    parser.add_argument("--pretrained")
    parser.add_argument("--revision")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--min-mentions", type=int, default=3)
    parser.add_argument("--min-class-support", type=int, default=3)
    parser.add_argument(
        "--merge-method",
        choices=("profile", "text", "profile-and-text", "lexical", "adjudicated", "none"),
        default="profile",
    )
    parser.add_argument("--merge-adjudication")
    parser.add_argument(
        "--adjudicated-linkage",
        choices=("complete", "average"),
        default="complete",
    )
    parser.add_argument(
        "--adjudicated-min-similarity",
        type=float,
        default=1.0,
        help="minimum average approved-edge cosine for average adjudicated linkage",
    )
    parser.add_argument("--merge-corr", type=float, default=0.5)
    parser.add_argument("--text-merge-cosine", type=float, default=0.95)
    parser.add_argument(
        "--selection-method",
        choices=("profile", "text", "profile-and-text", "frequency"),
        default="profile",
    )
    parser.add_argument("--max-twin-corr", type=float, default=0.75)
    parser.add_argument("--max-text-twin-cosine", type=float, default=0.98)
    parser.add_argument("--remove-components", type=int, default=1)
    parser.add_argument("--split-manifest", default="data/splits.json")
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="train")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--verify-cache-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.k < 1 or args.min_mentions < 1:
        parser.error("batch size, k, and minimum support must be positive")
    if args.min_class_support < 1 or args.remove_components < 0:
        parser.error("class support must be positive and components nonnegative")
    for name in (
        "merge_corr",
        "text_merge_cosine",
        "max_twin_corr",
        "max_text_twin_cosine",
    ):
        if not -1.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between -1 and 1")
    if not 0.0 <= args.adjudicated_min_similarity <= 1.0:
        parser.error("--adjudicated-min-similarity must be between 0 and 1")
    if args.split in {"test", "all"} and not args.allow_test:
        raise ValueError("dictionary construction may read test only with --allow-test")

    encoder = select_encoder(
        args.encoder,
        model=args.model,
        pretrained=args.pretrained,
        revision=args.revision,
    )
    embedding_dir = encoder_output_dir(ROOT, encoder, args.embedding_dir)
    device = require_cuda()
    precision = args.precision
    attrs_path = ROOT / args.open_tags
    image_dir = ROOT / args.img_dir
    split_manifest_path = ROOT / args.split_manifest
    manifest = load_split_manifest(str(ROOT), args.split_manifest)
    rows = [json.loads(line) for line in attrs_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("attributes")]
    if args.split != "all":
        rows = [
            row
            for row in rows
            if split_for_image(manifest, row["image_id"]) == args.split
        ]
    if not rows:
        raise ValueError("no valid attribute rows remain after split filtering")
    selected_ids = [row["image_id"] for row in rows]

    phrases, negated_of, support = candidate_phrases(
        rows,
        min_mentions=args.min_mentions,
        min_class_support=args.min_class_support,
    )
    counts = support["image_mentions"]
    print(
        f"{len(rows)} images / {len(support['class_totals'])} classes | "
        f"{len(counts)} raw phrases | {len(phrases) + len(negated_of)} candidates",
        flush=True,
    )
    print(
        f"positive-only filter: excluded {len(negated_of)} lexical negations",
        flush=True,
    )

    image_embeddings, cache_provenance = validate_and_load_image_cache(
        embedding_dir,
        encoder,
        image_dir,
        selected_ids,
        verify_hashes=args.verify_cache_hashes,
    )
    text_embeddings, encoder_source = encode_phrases(
        phrases,
        encoder,
        device=device,
        precision=precision,
        batch_size=args.batch_size,
    )
    if text_embeddings.shape[1] != image_embeddings.shape[1]:
        raise ValueError("image and phrase embedding dimensions do not match")

    started = time.monotonic()
    image_scores = image_embeddings @ text_embeddings.T
    scores, profile_classes = class_mean_profiles(image_scores, selected_ids)
    scores = standardize_and_debias(scores, args.remove_components)
    print(
        f"built {scores.shape[0]} class-balanced phrase profiles in "
        f"{time.monotonic() - started:.3f}s",
        flush=True,
    )

    profile_similarity = np.corrcoef(scores.T)
    if not np.isfinite(profile_similarity).all():
        raise ValueError("non-finite phrase correlations")
    text_similarity = cosine_similarity_matrix(text_embeddings)
    adjudicated_similarity = None
    adjudication_artifact = None
    if args.merge_method == "adjudicated":
        if not args.merge_adjudication:
            parser.error("--merge-method adjudicated requires --merge-adjudication")
        adjudicated_similarity, adjudication_artifact = load_adjudicated_similarity(
            ROOT / args.merge_adjudication,
            phrases,
            weighted=args.adjudicated_linkage == "average",
        )
    labels = complete_linkage_labels(
        profile_similarity,
        text_similarity,
        method=args.merge_method,
        profile_threshold=args.merge_corr,
        text_threshold=args.text_merge_cosine,
        lexical_similarity=lexical_variant_similarity(phrases),
        adjudicated_similarity=adjudicated_similarity,
        adjudicated_linkage=args.adjudicated_linkage,
        adjudicated_min_similarity=args.adjudicated_min_similarity,
    )
    print(
        f"{args.merge_method} {args.adjudicated_linkage if args.merge_method == 'adjudicated' else 'complete'}-linkage "
        f"(profile>{args.merge_corr}, text>{args.text_merge_cosine}) -> "
        f"{len(set(labels))} clusters (target {args.k})",
        flush=True,
    )

    cluster_for_phrase = {
        phrase: int(labels[index]) for index, phrase in enumerate(phrases)
    }
    cluster_image_mentions = Counter()
    cluster_class_counts: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        concept = image_class(row["image_id"])
        active = {
            cluster_for_phrase[phrase]
            for phrase in row["attributes"]
            if phrase in cluster_for_phrase
        }
        for label in active:
            cluster_image_mentions[label] += 1
            cluster_class_counts[label][concept] += 1

    n_images = len(rows)
    n_classes = len(support["class_totals"])
    concepts = []
    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        members = [phrases[index] for index in indices]
        name = sorted(
            members,
            key=lambda member: (
                -support["balanced_prevalence"][member],
                -support["class_support"][member],
                -counts[member],
                member,
            ),
        )[0]
        per_class = cluster_class_counts[label]
        balanced_prevalence = sum(
            count / support["class_totals"][concept]
            for concept, count in per_class.items()
        ) / n_classes
        concept = {
            "name": name,
            "members": members,
            "prevalence": balanced_prevalence,
            "image_prevalence": cluster_image_mentions[label] / n_images,
            "mentions": cluster_image_mentions[label],
            "class_support": len(per_class),
            "profile": scores[:, indices].mean(axis=1),
            "text_embedding": text_embeddings[indices].mean(axis=0),
        }
        concepts.append(concept)

    concepts.sort(
        key=lambda concept: (
            -concept["prevalence"],
            -concept["class_support"],
            -concept["mentions"],
            concept["name"],
        )
    )
    profiles = np.stack([concept["profile"] for concept in concepts])
    profiles = (profiles - profiles.mean(axis=1, keepdims=True)) / (
        profiles.std(axis=1, keepdims=True) + 1e-8
    )
    concept_text_embeddings = np.stack(
        [concept["text_embedding"] for concept in concepts]
    )
    concept_text_embeddings /= np.maximum(
        np.linalg.norm(concept_text_embeddings, axis=1, keepdims=True), 1e-8
    )
    cluster_candidates_output = None
    if args.cluster_candidates_out:
        cluster_candidates_output = ROOT / args.cluster_candidates_out
        cluster_candidates = [
            {
                key: value
                for key, value in concept.items()
                if key not in {"profile", "text_embedding"}
            }
            for concept in concepts
        ]
        write_json_atomic(cluster_candidates_output, cluster_candidates)
        print(
            f"wrote {args.cluster_candidates_out} "
            f"({len(cluster_candidates)} pre-selection clusters)"
        )
    selected = []
    selected_profiles = []
    selected_text_embeddings = []
    skipped = []
    for index, concept in enumerate(concepts):
        if len(selected) >= args.k:
            break
        if selected_profiles and args.selection_method != "frequency":
            profile_matches = (
                np.stack(selected_profiles) @ profiles[index] / profiles.shape[1]
            )
            text_matches = np.stack(selected_text_embeddings) @ concept_text_embeddings[index]
            if args.selection_method == "profile":
                redundant = profile_matches > args.max_twin_corr
            elif args.selection_method == "text":
                redundant = text_matches > args.max_text_twin_cosine
            elif args.selection_method == "profile-and-text":
                redundant = (
                    (profile_matches > args.max_twin_corr)
                    & (text_matches > args.max_text_twin_cosine)
                )
            else:
                raise ValueError(f"unknown selection method: {args.selection_method}")
            if redundant.any():
                twin_index = int(np.flatnonzero(redundant)[0])
                skipped.append(
                    {
                        "candidate": concept["name"],
                        "selected_twin": selected[twin_index]["name"],
                        "profile_correlation": float(profile_matches[twin_index]),
                        "text_cosine": float(text_matches[twin_index]),
                    }
                )
                continue
        selected.append(concept)
        selected_profiles.append(profiles[index])
        selected_text_embeddings.append(concept_text_embeddings[index])
    if len(selected) != args.k:
        raise ValueError(
            f"only {len(selected)} concepts survived near-twin rejection; target={args.k}"
        )
    print(f"greedy selection: kept {len(selected)}, skipped {len(skipped)} near-twins")
    for concept in selected:
        concept.pop("profile")
        concept.pop("text_embedding")
    selected.sort(key=lambda concept: (-concept["prevalence"], concept["name"]))

    output = ROOT / args.out
    provenance_output = ROOT / args.provenance_out
    write_json_atomic(output, selected)
    provenance = {
        "schema": "conceptbasis.dictionary-build/v5",
        "encoder": encoder.as_dict(),
        "encoder_source": encoder_source,
        "embedding_cache": cache_provenance,
        "attribute_split": args.split,
        "profile_unit": "things_object_class_mean",
        "n_attribute_images": n_images,
        "n_profile_classes": len(profile_classes),
        "n_raw_phrases": len(counts),
        "n_candidate_phrases_before_negation_filter": len(phrases) + len(negated_of),
        "n_phrase_profiles": len(phrases),
        "n_clusters": len(concepts),
        "k": args.k,
        "min_mentions": args.min_mentions,
        "min_class_support": args.min_class_support,
        "merge_method": args.merge_method,
        "merge_corr": args.merge_corr,
        "text_merge_cosine": args.text_merge_cosine,
        "selection_method": args.selection_method,
        "max_twin_corr": args.max_twin_corr,
        "max_text_twin_cosine": args.max_text_twin_cosine,
        "remove_components": args.remove_components,
        "text_template": TEXT_TEMPLATE,
        "precision": precision,
        "coefficient_domain": "nonnegative",
        "concept_policy": "affirmatively-present-traits-only",
        "negation_policy": "exclude",
        "excluded_lexical_negations": negated_of,
        "near_twins_skipped": skipped,
        "inputs": {
            args.open_tags: sha256_file(attrs_path),
            args.split_manifest: sha256_file(split_manifest_path),
        },
        "dictionary_sha256": sha256_file(output),
    }
    if adjudication_artifact is not None:
        provenance["merge_adjudication"] = {
            "path": args.merge_adjudication,
            "sha256": sha256_file(ROOT / args.merge_adjudication),
            "model": adjudication_artifact.get("model"),
            "prompt_sha256": adjudication_artifact.get("prompt_sha256"),
            "n_proposed_edges": adjudication_artifact.get("n_proposed_edges"),
            "n_accepted_edges": adjudication_artifact.get("n_accepted_edges"),
            "linkage": args.adjudicated_linkage,
            "min_similarity": args.adjudicated_min_similarity,
            "edge_weights": (
                "approved_pair_text_cosine"
                if args.adjudicated_linkage == "average"
                else "approved_pair_binary"
            ),
        }
    if cluster_candidates_output is not None:
        provenance["cluster_candidates"] = {
            "path": args.cluster_candidates_out,
            "sha256": sha256_file(cluster_candidates_output),
            "count": len(concepts),
        }
    write_json_atomic(provenance_output, provenance)
    print(f"wrote {args.out} ({len(selected)} concepts)")
    print(f"wrote {args.provenance_out}")


if __name__ == "__main__":
    main()
