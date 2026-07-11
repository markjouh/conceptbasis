"""Playground-faithful composability benchmark with attribute subset rollouts.

This evaluates the operation performed by the static playgrounds. Each gallery
image is represented by standardized projections onto the model's concept
directions. A query containing concepts S scores gallery item j as

    score(j | S) = sum(profile[j, k] for k in S).

For every CC0 image, VLM-listed attribute phrases are mapped through the exact
dictionary used by the playground. Fixed-seed random permutations provide
nested attribute subsets, so retrieval can be measured as the attribute budget
grows. The same subsets are used for frozen and trained profile spaces.

The current working tree's dictionary has moved beyond the checked-in
playgrounds. For historical artifacts, pass --dictionary-git with a git object
such as HEAD:data/dictionary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_dictionary(path: str | None, git_spec: str | None) -> list[dict]:
    if git_spec:
        raw = subprocess.check_output(["git", "show", git_spec], cwd=ROOT)
        return json.loads(raw)
    if not path:
        raise ValueError("provide --dictionary or --dictionary-git")
    return load_json(os.path.join(ROOT, path))


def load_attributes(path: str) -> list[dict]:
    rows = []
    seen = set()
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("attributes"):
                continue
            if row["image_id"] in seen:
                raise ValueError(f"duplicate image_id at line {line_no}: {row['image_id']}")
            seen.add(row["image_id"])
            rows.append(row)
    return rows


def load_playground(path: str) -> dict:
    text = open(path).read()
    match = re.search(r"const DATA=(.*?);\s*const N=", text, re.S)
    if not match:
        raise ValueError(f"could not locate embedded DATA in {path}")
    return json.loads(match.group(1))


def image_label(image_id: str) -> str:
    stem = os.path.splitext(image_id)[0]
    stem = re.sub(r"\d+$", "", stem)
    return stem.replace("_", " ")


def parse_sizes(value: str) -> list[int]:
    sizes = sorted({int(x) for x in value.split(",") if x.strip()})
    if not sizes or sizes[0] < 1:
        raise ValueError("subset sizes must be positive integers")
    return sizes


def mapped_concepts(rows: list[dict], dictionary: list[dict], flags: dict,
                    include_flagged: bool) -> list[np.ndarray]:
    member_to_concept = {}
    for k, concept in enumerate(dictionary):
        for member in concept["members"]:
            if member in member_to_concept:
                raise ValueError(f"dictionary member appears in multiple concepts: {member!r}")
            member_to_concept[member] = k

    out = []
    for row in rows:
        indices = {
            member_to_concept[a]
            for a in row["attributes"]
            if a in member_to_concept
        }
        if not include_flagged:
            indices = {k for k in indices if dictionary[k]["name"] not in flags}
        out.append(np.array(sorted(indices), dtype=np.int32))
    return out


def build_rollouts(concepts: list[np.ndarray], n_axes: int, sizes: list[int],
                   rollouts: int, seed: int):
    """Return nested true/random concept sequences and their source-image rows."""
    max_size = max(sizes)
    rng = np.random.default_rng(seed)
    true_seq, random_seq, targets, rollout_ids, lengths = [], [], [], [], []
    universe = np.arange(n_axes, dtype=np.int32)

    for image_i, available in enumerate(concepts):
        # Keep one fixed image cohort across the entire attribute-budget curve.
        if len(available) < max_size:
            continue
        excluded = np.zeros(n_axes, dtype=bool)
        excluded[available] = True
        complement = universe[~excluded]
        for rollout_i in range(rollouts):
            true = rng.permutation(available)
            usable = min(len(true), max_size)
            padded = np.zeros(max_size, dtype=np.int32)
            padded[:usable] = true[:usable]

            random = rng.choice(complement, size=max_size, replace=False)
            true_seq.append(padded)
            random_seq.append(random)
            targets.append(image_i)
            rollout_ids.append(rollout_i)
            lengths.append(usable)

    return {
        "true": np.stack(true_seq),
        "random": np.stack(random_seq),
        "target": np.array(targets, dtype=np.int32),
        "rollout": np.array(rollout_ids, dtype=np.int16),
        "length": np.array(lengths, dtype=np.int16),
    }


def rank_prefixes(profiles: np.ndarray, sequences: np.ndarray, targets: np.ndarray,
                  lengths: np.ndarray, sizes: list[int], batch: int) -> np.ndarray:
    """Average-tie rank of each target for every valid prefix size."""
    ranks = np.full((len(sequences), len(sizes)), np.nan, dtype=np.float32)
    for start in range(0, len(sequences), batch):
        stop = min(start + batch, len(sequences))
        seq = sequences[start:stop]
        # [gallery, batch, max_k], then cumulative score for all nested prefixes.
        cumulative = np.cumsum(profiles[:, seq], axis=2)
        batch_targets = targets[start:stop]
        batch_lengths = lengths[start:stop]
        for si, size in enumerate(sizes):
            valid = batch_lengths >= size
            if not valid.any():
                continue
            scores = cumulative[:, valid, size - 1].T
            target_scores = scores[np.arange(len(scores)), batch_targets[valid]]
            greater = (scores > target_scores[:, None] + 1e-7).sum(axis=1)
            equal = np.isclose(scores, target_scores[:, None], atol=1e-7, rtol=0).sum(axis=1)
            ranks[start:stop, si][valid] = 1 + greater + 0.5 * (equal - 1)
    return ranks


def curve(ranks: np.ndarray, targets: np.ndarray, sizes: list[int], gallery_size: int):
    result = {}
    for si, size in enumerate(sizes):
        valid = ~np.isnan(ranks[:, si])
        r = ranks[valid, si]
        t = targets[valid]
        if not len(r):
            continue
        result[str(size)] = {
            "n_images": int(len(np.unique(t))),
            "n_rollout_queries": int(len(r)),
            "R@1": float((r <= 1).mean()),
            "R@5": float((r <= 5).mean()),
            "R@10": float((r <= 10).mean()),
            "R@20": float((r <= 20).mean()),
            "R@25": float((r <= 25).mean()),
            "R@50": float((r <= 50).mean()),
            "mean_reciprocal_rank": float((1.0 / r).mean()),
            "median_rank": float(np.median(r)),
            "mean_normalized_rank": float(((r - 1) / (gallery_size - 1)).mean()),
        }
    return result


def monotonicity(ranks: np.ndarray, sizes: list[int]) -> dict:
    out = {}
    for i in range(1, len(sizes)):
        before, after = ranks[:, i - 1], ranks[:, i]
        valid = ~np.isnan(before) & ~np.isnan(after)
        delta = after[valid] - before[valid]
        if not len(delta):
            continue
        out[f"{sizes[i - 1]}->{sizes[i]}"] = {
            "n": int(len(delta)),
            "improved": float((delta < 0).mean()),
            "unchanged": float((delta == 0).mean()),
            "worsened": float((delta > 0).mean()),
            "mean_rank_change": float(delta.mean()),
        }
    return out


def model_eval(profiles: np.ndarray, rollout_data: dict, sizes: list[int], batch: int):
    true_ranks = rank_prefixes(
        profiles, rollout_data["true"], rollout_data["target"],
        rollout_data["length"], sizes, batch,
    )
    random_ranks = rank_prefixes(
        profiles, rollout_data["random"], rollout_data["target"],
        rollout_data["length"], sizes, batch,
    )
    return {
        "true_attributes": curve(true_ranks, rollout_data["target"], sizes, len(profiles)),
        "random_attributes": curve(random_ranks, rollout_data["target"], sizes, len(profiles)),
        "monotonicity": monotonicity(true_ranks, sizes),
        "_true_ranks": true_ranks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained-html", default="docs/playground.html")
    ap.add_argument("--frozen-html", default="docs/playground-baseline.html")
    ap.add_argument("--attributes", default="data/attributes.jsonl")
    ap.add_argument("--dictionary", default=None)
    ap.add_argument("--dictionary-git", default=None,
                    help="git object containing the matching dictionary")
    ap.add_argument("--profiles-npz", default=None,
                    help="optional NPZ with matched group-mean profile matrices")
    ap.add_argument("--reference-model", default=None,
                    help="profile key used as the reference for paired rank comparisons")
    ap.add_argument("--subset-sizes", default="1,2,3,4,6,8")
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--out", default="outputs/evals/playground_subset_composability.json")
    args = ap.parse_args()

    sizes = parse_sizes(args.subset_sizes)
    trained_path = os.path.join(ROOT, args.trained_html)
    frozen_path = os.path.join(ROOT, args.frozen_html)
    attrs_path = os.path.join(ROOT, args.attributes)
    trained = load_playground(trained_path)
    frozen = load_playground(frozen_path)
    dictionary = load_dictionary(args.dictionary, args.dictionary_git)
    rows = load_attributes(attrs_path)

    names = [c["name"] for c in dictionary]
    if trained["names"] != names or frozen["names"] != names:
        raise ValueError("playground concept names/order do not match the selected dictionary")
    if len(trained["gallery"]) != len(rows) or len(frozen["gallery"]) != len(rows):
        raise ValueError("playground gallery length does not match attribute rows")
    expected_labels = [image_label(r["image_id"]) for r in rows]
    for label, data in (("trained", trained), ("frozen", frozen)):
        gallery_labels = [g["concept"] for g in data["gallery"]]
        mismatches = [i for i, (a, b) in enumerate(zip(expected_labels, gallery_labels)) if a != b]
        if mismatches:
            raise ValueError(f"{label} gallery order differs from attributes at rows {mismatches[:5]}")

    flags = trained.get("flags", {})
    if frozen.get("flags", {}) != flags:
        raise ValueError("trained and frozen playground flag sets differ")
    concepts = mapped_concepts(rows, dictionary, flags, args.include_flagged)
    counts = np.array([len(x) for x in concepts])
    rollout_data = build_rollouts(concepts, len(names), sizes, args.rollouts, args.seed)

    profile_path = None
    if args.profiles_npz:
        profile_path = os.path.join(ROOT, args.profiles_npz)
        cached = np.load(profile_path)
        if list(cached["names"]) != names:
            raise ValueError("profile NPZ concept names/order do not match the dictionary")
        if list(cached["image_ids"]) != [r["image_id"] for r in rows]:
            raise ValueError("profile NPZ image IDs/order do not match attributes.jsonl")
        profiles = {
            key: cached[key].astype(np.float32)
            for key in cached.files
            if key not in {"names", "image_ids"}
        }
    else:
        profiles = {
            "frozen": np.array([g["p"] for g in frozen["gallery"]], dtype=np.float32),
            "trained": np.array([g["p"] for g in trained["gallery"]], dtype=np.float32),
        }
    evaluations = {}
    rank_arrays = {}
    for label, matrix in profiles.items():
        print(f"evaluating {label}: {matrix.shape}", flush=True)
        ev = model_eval(matrix, rollout_data, sizes, args.batch)
        rank_arrays[label] = ev.pop("_true_ranks")
        evaluations[label] = ev

    reference = args.reference_model
    if reference is None and {"frozen", "trained"}.issubset(rank_arrays):
        reference = "trained"
    if reference is not None and reference not in rank_arrays:
        raise ValueError(f"reference model {reference!r} not present in profile NPZ")
    deltas = {}
    if reference is not None:
        for label, ranks in rank_arrays.items():
            if label == reference:
                continue
            by_size = {}
            for si, size in enumerate(sizes):
                other, ref = ranks[:, si], rank_arrays[reference][:, si]
                valid = ~np.isnan(other) & ~np.isnan(ref)
                if not valid.any():
                    continue
                d = ref[valid] - other[valid]
                by_size[str(size)] = {
                    "reference_better_rank_fraction": float((d < 0).mean()),
                    "same_rank_fraction": float((d == 0).mean()),
                    "reference_worse_rank_fraction": float((d > 0).mean()),
                    "mean_reference_minus_model_rank": float(d.mean()),
                }
            deltas[label] = by_size

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "source-image retrieval from sums of standardized concept-direction profiles",
        "subset_sizes": sizes,
        "rollouts_per_image": args.rollouts,
        "seed": args.seed,
        "exclude_flagged": not args.include_flagged,
        "n_images": len(rows),
        "n_concepts": len(names),
        "mapped_concepts_per_image": {
            "min": int(counts.min()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
        },
        "chance": {
            "R@1": 1 / len(rows),
            "R@5": 5 / len(rows),
            "R@10": 10 / len(rows),
            "R@20": 20 / len(rows),
            "R@25": 25 / len(rows),
            "R@50": 50 / len(rows),
        },
        "inputs": {
            "trained_html": args.trained_html,
            "trained_html_sha256": sha256(trained_path),
            "frozen_html": args.frozen_html,
            "frozen_html_sha256": sha256(frozen_path),
            "attributes": args.attributes,
            "attributes_sha256": sha256(attrs_path),
            "dictionary": args.dictionary,
            "dictionary_git": args.dictionary_git,
            "profiles_npz": args.profiles_npz,
            "profiles_npz_sha256": sha256(profile_path) if profile_path else None,
        },
        "models": evaluations,
        "paired_comparison_reference": reference,
        "paired_comparison": deltas,
    }

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({m: e["true_attributes"] for m, e in evaluations.items()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
