"""Stage 6c — Summarize matched multi-seed retrieval evaluations.

Combines the per-seed compositional metrics and checkpoint histories into the
compact tracked result consumed by the README chart generator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from conceptbasis.encoders import write_json_atomic


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse_history_specs(values: list[str]) -> dict[str, str]:
    specs = {}
    for value in values:
        name, separator, template = value.partition("=")
        if not separator or not name or "{seed}" not in template:
            raise ValueError("--history must use NAME=PATH_WITH_{seed}")
        if name in specs:
            raise ValueError(f"duplicate history model: {name}")
        specs[name] = template
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--metrics", required=True)
    parser.add_argument(
        "--history",
        action="append",
        required=True,
        help="NAME=checkpoint/history/path/with/s{seed}/history.json",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integers")
    histories = parse_history_specs(args.history)
    metrics_path = resolve(args.metrics)
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("eval_split") != "dev":
        raise ValueError("summary is restricted to the development split")
    sizes = [int(value) for value in metrics["subset_sizes"]]

    models = {}
    history_inputs = {}
    for family, template in histories.items():
        composition = {}
        for size in sizes:
            values = [
                metrics["models"][f"{family}_s{seed}"]["true_attributes"][str(size)][
                    "R@5"
                ]
                for seed in seeds
            ]
            composition[str(size)] = mean_std(values)

        retrieval = []
        history_inputs[family] = {}
        for seed in seeds:
            path = resolve(template.format(seed=seed))
            history = json.loads(path.read_text())
            retrieval.append(float(history[-1]["dev"]["R@k"]["5"]))
            history_inputs[family][str(seed)] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        models[family] = {
            "ordinary_R@5": mean_std(retrieval),
            "composition_R@5": composition,
        }

    reference_family = next(iter(histories))
    cohort = {
        str(size): metrics["models"][f"{reference_family}_s{seeds[0]}"][
            "true_attributes"
        ][str(size)]["n_images"]
        for size in sizes
    }
    input_labels = metrics["inputs"].get(
        "fixed_labels", metrics["inputs"].get("attributes")
    )
    input_labels_sha256 = metrics["inputs"].get(
        "fixed_labels_sha256", metrics["inputs"].get("attributes_sha256")
    )
    if not input_labels or not input_labels_sha256:
        raise ValueError("metrics do not identify their fixed-label input")

    result = {
        "schema": "conceptbasis.seeded-composability-summary/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development",
        "eval_split": "dev",
        "test_split_read": False,
        "seeds": seeds,
        "subset_sizes": sizes,
        "cohort_images": cohort,
        "mapped_concepts_per_image": metrics["mapped_concepts_per_image"],
        "models": models,
        "inputs": {
            "composability": {
                "path": str(metrics_path.relative_to(ROOT)),
                "sha256": sha256(metrics_path),
            },
            "fixed_labels": input_labels,
            "fixed_labels_sha256": input_labels_sha256,
            "label_field": metrics["inputs"].get(
                "label_field", metrics["inputs"].get("attribute_field", "present")
            ),
            "histories": history_inputs,
        },
    }
    output = resolve(args.out)
    write_json_atomic(output, result)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
