"""Stage 5 experiment — Sweep the production reverse-ridge objective.

This is deliberately a small, matched sweep over the two reverse-ridge
hyperparameters.  It uses ``conceptbasis.train`` directly so the selected
configuration and the regular training entry point cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "reverse_sweep_v2_"


@dataclass(frozen=True)
class Config:
    name: str
    ridge_alpha: float
    lambda_orth: float


CONFIGS = [
    Config(f"p0_l{orth:g}", 1e-3, orth)
    for orth in (32.0, 64.0, 128.0, 256.0, 512.0)
]


def run_streaming(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("ep") or line.startswith("saved"):
                print(f"  {line.rstrip()}", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_name(config: Config, seed: int) -> str:
    return PREFIX + (config.name if seed == 0 else f"s{seed}_{config.name}")


def completed_checkpoint(config: Config, epochs: int, seed: int) -> Path | None:
    run_dir = ROOT / "outputs/checkpoints" / run_name(config, seed)
    checkpoint = run_dir / "ckpt.pt"
    config_path = run_dir / "config.json"
    if not checkpoint.exists() or not config_path.exists():
        return None
    saved = json.loads(config_path.read_text())
    actual = saved.get("config", saved)
    expected = {
        "objective": "reverse-ridge",
        "epochs": epochs,
        "ridge_alpha": config.ridge_alpha,
        "lambda_orth": config.lambda_orth,
        "seed": seed,
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            f"existing checkpoint has a mismatched configuration: {run_dir}"
        )
    return checkpoint


def selected_configs(names: str | None) -> list[Config]:
    if not names:
        return CONFIGS
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    selected = [config for config in CONFIGS if config.name in wanted]
    missing = wanted - {config.name for config in selected}
    if missing:
        raise ValueError(f"unknown configurations: {sorted(missing)}")
    return selected


def train(configs: list[Config], *, epochs: int, seed: int) -> None:
    suffix = "" if seed == 0 else f"_s{seed}"
    status_path = ROOT / f"outputs/evals/reverse_sweep_v2{suffix}_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "epochs": epochs,
        "seed": seed,
        "configs": [asdict(config) for config in configs],
        "completed": [],
    }
    for index, config in enumerate(configs, 1):
        checkpoint = completed_checkpoint(config, epochs, seed)
        if checkpoint is not None:
            print(f"[{index}/{len(configs)}] skip {config.name}", flush=True)
        else:
            print(f"[{index}/{len(configs)}] train {config.name}", flush=True)
            command = [
                sys.executable,
                "-m", "conceptbasis.train",
                "--objective", "reverse-ridge",
                "--embed_dim", "320",
                "--epochs", str(epochs),
                "--batch", "1024",
                "--ridge_chunk", "2048",
                "--lr", "0.001",
                "--weight_decay", "0.0001",
                "--ridge_alpha", str(config.ridge_alpha),
                "--lambda_orth", str(config.lambda_orth),
                "--eval_every", str(epochs),
                "--seed", str(seed),
                "--run_name", run_name(config, seed),
            ]
            run_streaming(
                command,
                ROOT / f"outputs/logs/reverse_sweep_v2{suffix}/{config.name}.log",
            )
        status["completed"].append(config.name)
        status_path.write_text(json.dumps(status, indent=2) + "\n")


def evaluate(configs: list[Config], *, epochs: int, seed: int) -> None:
    checkpoints = {
        config.name: completed_checkpoint(config, epochs, seed)
        for config in configs
    }
    missing = [name for name, checkpoint in checkpoints.items() if checkpoint is None]
    if missing:
        raise RuntimeError(f"cannot evaluate incomplete configurations: {missing}")

    suffix = "" if seed == 0 else f"_s{seed}"
    profiles = f"outputs/evals/reverse_sweep_v2{suffix}_profiles.npz"
    profile_command = [
        sys.executable,
        "scripts/evaluation/build_retrieval_profiles.py",
        "--embeddings", "data/image_embeddings.npy",
        "--cc0-embeddings", "data/image_embeddings_cc0.npy",
        "--soft-labels", "data/labels.parquet",
        "--out", profiles,
    ]
    for name, checkpoint in checkpoints.items():
        assert checkpoint is not None
        profile_command.extend(
            ["--checkpoint", f"{name}={checkpoint.relative_to(ROOT)}"]
        )
    run_streaming(
        profile_command,
        ROOT / f"outputs/logs/reverse_sweep_v2{suffix}/build_profiles.log",
    )

    metrics_path = f"outputs/evals/reverse_sweep_v2{suffix}_composability.json"
    run_streaming(
        [
            sys.executable,
            "scripts/evaluation/evaluate_compositional_retrieval.py",
            "--dictionary", "data/dictionary.json",
            "--profiles-npz", profiles,
            "--subset-sizes", "1,2,4,6,8,10,12,14",
            "--rollouts", "24",
            "--rollouts-by-size", "2:25,4:25,6:25,8:25,10:26,12:35,14:129",
            "--seed", str(seed),
            "--out", metrics_path,
        ],
        ROOT / f"outputs/logs/reverse_sweep_v2{suffix}/evaluate.log",
    )
    metrics = json.loads((ROOT / metrics_path).read_text())

    torch = __import__("torch")
    summary = []
    for config in configs:
        checkpoint_path = checkpoints[config.name]
        assert checkpoint_path is not None
        run_dir = checkpoint_path.parent
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = json.loads((run_dir / "history.json").read_text())
        final = checkpoint["final_direction"]
        dev_retrieval = history[-1]["dev"]
        curve = metrics["models"][config.name]["true_attributes"]
        row = asdict(config)
        row["seed"] = seed
        row.update(
            {
                "reverse_orth_rms": final["orth_rms"],
                "reverse_explained": final["explained_fraction"],
                "dev_R@5": dev_retrieval["R@k"]["5"],
                "R@5_k1": curve["1"]["R@5"],
                "R@5_k4": curve["4"]["R@5"],
                "R@5_k8": curve["8"]["R@5"],
                "R@5_k14": curve["14"]["R@5"],
                "MRR_k14": curve["14"]["mean_reciprocal_rank"],
                "mean_rank_k14": curve["14"]["mean_normalized_rank"],
            }
        )
        row["mean_R@5_k4_k8_k14"] = (
            row["R@5_k4"] + row["R@5_k8"] + row["R@5_k14"]
        ) / 3
        summary.append(row)
    summary.sort(key=lambda row: row["mean_R@5_k4_k8_k14"], reverse=True)
    summary_path = ROOT / f"outputs/evals/reverse_sweep_v2{suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nranked reverse-ridge screen", flush=True)
    for row in summary:
        print(
            f"{row['name']:10s} mean={row['mean_R@5_k4_k8_k14']:.3f} "
            f"R5[1,4,8,14]={row['R@5_k1']:.3f},"
            f"{row['R@5_k4']:.3f},{row['R@5_k8']:.3f},{row['R@5_k14']:.3f} "
            f"orth={row['reverse_orth_rms']:.3f} dev={row['dev_R@5']:.3f}",
            flush=True,
        )
    print(f"wrote {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--configs",
        default="p0_l32,p0_l64,p0_l128,p0_l256,p0_l512",
        help="comma-separated pure-reverse configurations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.train_only and args.evaluate_only:
        raise ValueError("--train-only and --evaluate-only are mutually exclusive")
    configs = selected_configs(args.configs)
    if args.dry_run:
        for config in configs:
            print(asdict(config))
        return
    if not args.evaluate_only:
        train(configs, epochs=args.epochs, seed=args.seed)
    if not args.train_only:
        evaluate(configs, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
