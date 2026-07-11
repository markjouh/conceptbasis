"""Quick MPS-friendly screen for smooth correlation-weighted orthogonality."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREFIX = "classsplit_smooth_"

# name, mode, tau, floor, power, lambda_orth
CONFIGS = [
    ("hard_t015", "hard", .15, .00, 2, 5),
    ("t010_p2_f000", "smooth", .10, .00, 2, 5),
    ("t010_p4_f000", "smooth", .10, .00, 4, 5),
    ("t015_p2_f000", "smooth", .15, .00, 2, 5),
    ("t015_p4_f000", "smooth", .15, .00, 4, 5),
    ("t015_p8_f000", "smooth", .15, .00, 8, 5),
    ("t020_p2_f000", "smooth", .20, .00, 2, 5),
    ("t020_p4_f000", "smooth", .20, .00, 4, 5),
    ("t020_p8_f000", "smooth", .20, .00, 8, 5),
    ("t015_p4_f001", "smooth", .15, .01, 4, 5),
    ("t020_p4_f001", "smooth", .20, .01, 4, 5),
    ("t015_p4_f005", "smooth", .15, .05, 4, 5),
    ("t015_p4_f001_l3", "smooth", .15, .01, 4, 3),
    ("t015_p4_f001_l8", "smooth", .15, .01, 4, 8),
]

COMMON = [
    "--embed_dim", "320", "--epochs", "30", "--batch", "1024",
    "--lr", "0.001", "--weight_decay", "0.0001",
    "--ema", "0.9", "--seed", "0", "--device", "mps",
    "--image_ids", "data/image_ids.json",
    "--image_embeddings", "data/image_embeddings.npy",
    "--caption_embeddings", "data/caption_embeddings.npy",
    "--labels", "data/labels.parquet",
]


def run(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the training configurations without launching them",
    )
    args = ap.parse_args()
    if args.dry_run:
        for config in CONFIGS:
            print(config)
        return
    if not __import__("torch").backends.mps.is_available():
        raise RuntimeError("this sweep is intended to run with MPS available")

    for name, mode, tau, floor, power, lam in CONFIGS:
        run_name = PREFIX + name
        print("training", name, flush=True)
        command = [sys.executable, "-m", "conceptbasis.train", *COMMON,
                   "--lambda_orth", str(lam), "--corr_exempt", str(tau),
                   "--corr_weighting", mode, "--corr_weight_floor", str(floor),
                   "--corr_weight_power", str(power), "--run_name", run_name]
        output = run(command)
        print(output.strip().splitlines()[-2], flush=True)

    profiles = "outputs/evals/classsplit_smooth_profiles.npz"
    command = [
        sys.executable, "scripts/evaluation/build_groupmean_profiles.py",
        "--embeddings", "data/image_embeddings.npy",
        "--cc0-embeddings", "data/image_embeddings_cc0.npy",
        "--labels", "data/labels.parquet",
        "--attributes", "data/attributes_dev.jsonl", "--include-frozen",
    ]
    for name, *_ in CONFIGS:
        command.extend(["--checkpoint", f"{name}=outputs/checkpoints/{PREFIX}{name}/ckpt.pt"])
    command.extend(["--out", profiles])
    print("building profiles", flush=True)
    run(command)

    metrics_path = "outputs/evals/classsplit_smooth_k14.json"
    run([
        sys.executable, "scripts/evaluation/eval_playground_subset_composability.py",
        "--dictionary", "data/dictionary.json",
        "--profiles-npz", profiles,
        "--subset-sizes", "1,4,8,14", "--rollouts", "24", "--seed", "0",
        "--out", metrics_path,
    ])
    metrics = json.load(open(os.path.join(ROOT, metrics_path)))

    summary = []
    for name, mode, tau, floor, power, lam in CONFIGS:
        history = json.load(open(os.path.join(
            ROOT, "outputs", "checkpoints", PREFIX + name, "history.json")))
        val = history[-1]["dev"]
        curves = metrics["models"][name]["true_attributes"]
        row = {
            "name": name, "mode": mode, "tau": tau, "floor": floor,
            "power": power, "lambda_orth": lam,
            "auroc": val["auroc"], "orth_rms": val["orth_rms"],
            "caption_R1": val["R@k"]["1"],
            "median_rank_k1": curves["1"]["median_rank"],
            "median_rank_k4": curves["4"]["median_rank"],
            "median_rank_k8": curves["8"]["median_rank"],
            "median_rank_k14": curves["14"]["median_rank"],
            "MRR_k14": curves["14"]["mean_reciprocal_rank"],
            "R1_k14": curves["14"]["R@1"],
            "R10_k14": curves["14"]["R@10"],
            "R50_k14": curves["14"]["R@50"],
        }
        # Exploratory balance score. Keep its components visible; do not use as
        # a confirmatory test statistic.
        row["screen_score"] = row["MRR_k14"] * row["auroc"]
        summary.append(row)
    summary.sort(key=lambda x: x["screen_score"], reverse=True)

    out = os.path.join(ROOT, "outputs/evals/classsplit_smooth_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nranked screen")
    for row in summary:
        print(f"{row['name']:20s} score={row['screen_score']:.4f} "
              f"auc={row['auroc']:.3f} orth={row['orth_rms']:.3f} "
              f"med1={row['median_rank_k1']:.0f} med14={row['median_rank_k14']:.0f} "
              f"MRR14={row['MRR_k14']:.3f} R10={100*row['R10_k14']:.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
