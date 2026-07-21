"""Stage 7 reporting — Render the README retrieval comparison chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--metrics",
        default="research/results/siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_k20.json",
        help="tracked three-model development summary",
    )
    parser.add_argument(
        "--retrieval-metrics",
        default="research/results/matched_retrieval_dev.json",
        help="tracked matched ordinary-retrieval metrics",
    )
    parser.add_argument(
        "--out",
        default="docs/assets/composability-retrieval.svg",
    )
    args = parser.parse_args()

    metrics = json.loads((ROOT / args.metrics).read_text())
    seeded = metrics.get("schema") == "conceptbasis.seeded-composability-summary/v1"
    retrieval = None if seeded else json.loads((ROOT / args.retrieval_metrics).read_text())
    if metrics["eval_split"] != "dev" or (
        retrieval is not None and retrieval["eval_split"] != "dev"
    ):
        raise ValueError("README chart expects development metrics")

    subset_sizes = [int(value) for value in metrics["subset_sizes"]]
    model_specs = [
        ("contrastive", "contrastive", "Contrastive", "circle"),
        ("groupmean", "group_mean", "Group-mean orthogonality", "square"),
        ("reverse", "reverse_ridge", "Reverse-ridge orthogonality", "diamond"),
    ]
    if seeded:
        series = [
            (
                css_class,
                label,
                [metrics["models"][model]["composition_R@5"][str(size)]["mean"] for size in subset_sizes],
                [metrics["models"][model]["composition_R@5"][str(size)]["sample_std"] for size in subset_sizes],
                marker,
            )
            for css_class, model, label, marker in model_specs
        ]
        ordinary = [
            (
                css_class,
                label.replace(" orthogonality", ""),
                metrics["models"][model]["ordinary_R@5"]["mean"],
                metrics["models"][model]["ordinary_R@5"]["sample_std"],
                marker,
            )
            for css_class, model, label, marker in model_specs
        ]
        cohort_sizes = [metrics["cohort_images"][str(size)] for size in subset_sizes]
        seed_count = len(metrics["seeds"])
    else:
        series = [
            (
                css_class,
                label,
                [metrics["models"][model]["true_attributes"][str(size)]["R@5"] for size in subset_sizes],
                [0.0] * len(subset_sizes),
                marker,
            )
            for css_class, model, label, marker in model_specs
        ]
        ordinary = [
            (
                css_class,
                label.replace(" orthogonality", ""),
                retrieval["mean_percent_recall"][model]["R@5"] / 100,
                0.0,
                marker,
            )
            for css_class, model, label, marker in model_specs
        ]
        cohort_sizes = [
            metrics["models"]["reverse_ridge"]["true_attributes"][str(size)]["n_images"]
            for size in subset_sizes
        ]
        for model in ("contrastive", "group_mean"):
            observed = [
                metrics["models"][model]["true_attributes"][str(size)]["n_images"]
                for size in subset_sizes
            ]
            if observed != cohort_sizes:
                raise ValueError("model query cohorts differ")
        seed_count = 5

    width, height = 1120, 448
    top, bottom = 70, 76
    plot_bottom = height - bottom
    plot_height = plot_bottom - top
    comp_left, comp_right = 74, 620
    ordinary_left, ordinary_right = 820, 1100

    def y_pos(value: float) -> float:
        return top + plot_height * (1 - value)

    def comp_x(index: int) -> float:
        return comp_left + index * (comp_right - comp_left) / (len(subset_sizes) - 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Compositional and ordinary retrieval across three training objectives</title>',
        '<desc id="desc">Reverse-ridge orthogonality substantially improves compositional '
        'retrieval while retaining ordinary image-text retrieval performance.</desc>',
        "<style>",
        ".text{fill:#1f2328;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}",
        ".muted{fill:#57606a}.grid{stroke:#d8dee4;stroke-width:1}.axis{stroke:#8c959f;stroke-width:1}",
        ".contrastive{stroke:#6e7781}.point.contrastive{fill:#6e7781}",
        ".groupmean{stroke:#0969da}.point.groupmean{fill:#0969da}",
        ".reverse{stroke:#1a7f37}.point.reverse{fill:#1a7f37}",
        ".band{stroke:none;fill-opacity:.12}.band.contrastive{fill:#6e7781}",
        ".band.groupmean{fill:#0969da}.band.reverse{fill:#1a7f37}",
        ".series{fill:none;stroke-width:3}.point{stroke-width:0}",
        ".bar{fill-opacity:.16;stroke-width:2}.bar.contrastive{fill:#6e7781}",
        ".bar.groupmean{fill:#0969da}.bar.reverse{fill:#1a7f37}",
        ".label{font-size:13px}.small{font-size:12px}.title{font-size:17px;font-weight:500}",
        "@media(prefers-color-scheme:dark){.text{fill:#e6edf3}.muted{fill:#9da7b1}"
        ".grid{stroke:#30363d}.axis{stroke:#6e7681}.contrastive{stroke:#9da7b1}"
        ".point.contrastive{fill:#9da7b1}.bar.contrastive{fill:#9da7b1}"
        ".groupmean{stroke:#58a6ff}.point.groupmean{fill:#58a6ff}.bar.groupmean{fill:#58a6ff}"
        ".reverse{stroke:#3fb950}.point.reverse{fill:#3fb950}.bar.reverse{fill:#3fb950}}",
        "</style>",
        '<text class="text title" x="74" y="30">Compositional retrieval</text>',
        f'<text class="text muted small" x="74" y="50">278-class dev gallery · exhaustive labels · mean ± s.d. over {seed_count} seeds</text>',
        '<text class="text title" x="820" y="30">Ordinary image–text retrieval</text>',
        f'<text class="text muted small" x="820" y="50">3,914 dev pairs · mean ± s.d. over {seed_count} matched seeds</text>',
    ]

    for tick in range(0, 101, 20):
        value = tick / 100
        y = y_pos(value)
        lines.extend(
            [
                f'<line class="grid" x1="{comp_left}" y1="{y:.1f}" '
                f'x2="{comp_right}" y2="{y:.1f}"/>',
                f'<line class="grid" x1="{ordinary_left}" y1="{y:.1f}" '
                f'x2="{ordinary_right}" y2="{y:.1f}"/>',
                f'<text class="text muted small" x="{comp_left-12}" y="{y+4:.1f}" '
                f'text-anchor="end">{tick}%</text>',
            ]
        )

    lines.extend(
        [
            f'<line class="axis" x1="{comp_left}" y1="{top}" '
            f'x2="{comp_left}" y2="{plot_bottom}"/>',
            f'<line class="axis" x1="{comp_left}" y1="{plot_bottom}" '
            f'x2="{comp_right}" y2="{plot_bottom}"/>',
            f'<line class="axis" x1="{ordinary_left}" y1="{top}" '
            f'x2="{ordinary_left}" y2="{plot_bottom}"/>',
            f'<line class="axis" x1="{ordinary_left}" y1="{plot_bottom}" '
            f'x2="{ordinary_right}" y2="{plot_bottom}"/>',
        ]
    )

    for index, size in enumerate(subset_sizes):
        x = comp_x(index)
        lines.extend(
            [
                f'<text class="text small" x="{x:.1f}" y="{plot_bottom+20}" '
                f'text-anchor="middle">{size}</text>',
                (
                    f'<text class="text muted small" x="{x:.1f}" y="{plot_bottom+36}" '
                    f'text-anchor="middle">n={cohort_sizes[index]}</text>'
                    if cohort_sizes[index] != max(cohort_sizes)
                    else ""
                ),
            ]
        )
    lines.extend(
        [
            f'<text class="text muted label" x="{(comp_left+comp_right)/2:.1f}" '
            f'y="{height-12}" text-anchor="middle">Attributes in query (k)</text>',
            f'<text class="text muted label" transform="translate(20 {top+plot_height/2:.1f}) '
            'rotate(-90)" text-anchor="middle">Recall@5</text>',
        ]
    )

    for css_class, label, values, deviations, marker in series:
        upper = [min(1.0, value + deviation) for value, deviation in zip(values, deviations)]
        lower = [max(0.0, value - deviation) for value, deviation in zip(values, deviations)]
        band = " ".join(
            [
                *[
                    f"{'M' if index == 0 else 'L'} {comp_x(index):.1f} {y_pos(value):.1f}"
                    for index, value in enumerate(upper)
                ],
                *[
                    f"L {comp_x(index):.1f} {y_pos(lower[index]):.1f}"
                    for index in range(len(lower) - 1, -1, -1)
                ],
                "Z",
            ]
        )
        lines.append(f'<path class="band {css_class}" d="{band}"/>')
        path = " ".join(
            f"{'M' if index == 0 else 'L'} {comp_x(index):.1f} {y_pos(value):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(f'<path class="series {css_class}" d="{path}"/>')
        for index, value in enumerate(values):
            x, y = comp_x(index), y_pos(value)
            if marker == "circle":
                lines.append(
                    f'<circle class="point {css_class}" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>'
                )
            elif marker == "square":
                lines.append(
                    f'<rect class="point {css_class}" x="{x-3.5:.1f}" y="{y-3.5:.1f}" '
                    'width="7" height="7"/>'
                )
            else:
                lines.append(
                    f'<path class="point {css_class}" d="M {x:.1f} {y-5:.1f} '
                    f'L {x+5:.1f} {y:.1f} L {x:.1f} {y+5:.1f} '
                    f'L {x-5:.1f} {y:.1f} Z"/>'
                )
        final_y = y_pos(values[-1])
        lines.extend(
            [
                f'<line class="series {css_class}" x1="{comp_right}" y1="{final_y:.1f}" '
                f'x2="{comp_right+14}" y2="{final_y:.1f}"/>',
                f'<text class="text label" x="{comp_right+20}" y="{final_y-2:.1f}">{label}</text>',
                f'<text class="text muted small" x="{comp_right+20}" y="{final_y+14:.1f}">'
                f'{values[-1]*100:.1f}% at k={subset_sizes[-1]}</text>',
            ]
        )

    centers = [850, 950, 1050]
    for center, (css_class, label, value, deviation, marker) in zip(centers, ordinary):
        y = y_pos(value)
        whisker_top = y_pos(min(1.0, value + deviation))
        whisker_bottom = y_pos(max(0.0, value - deviation))
        lines.append(
            f'<rect class="bar {css_class}" x="{center-24}" y="{y:.1f}" '
            f'width="48" height="{plot_bottom-y:.1f}"/>'
        )
        lines.extend(
            [
                f'<line class="axis" x1="{center}" y1="{whisker_top:.1f}" x2="{center}" y2="{whisker_bottom:.1f}"/>',
                f'<line class="axis" x1="{center-6}" y1="{whisker_top:.1f}" x2="{center+6}" y2="{whisker_top:.1f}"/>',
                f'<line class="axis" x1="{center-6}" y1="{whisker_bottom:.1f}" x2="{center+6}" y2="{whisker_bottom:.1f}"/>',
            ]
        )
        if marker == "circle":
            lines.append(
                f'<circle class="point {css_class}" cx="{center}" cy="{y:.1f}" r="4"/>'
            )
        elif marker == "square":
            lines.append(
                f'<rect class="point {css_class}" x="{center-4}" y="{y-4:.1f}" '
                'width="8" height="8"/>'
            )
        else:
            lines.append(
                f'<path class="point {css_class}" d="M {center} {y-5.5:.1f} '
                f'L {center+5.5} {y:.1f} L {center} {y+5.5:.1f} '
                f'L {center-5.5} {y:.1f} Z"/>'
            )
        lines.extend(
            [
                f'<text class="text label" x="{center}" y="{y-10:.1f}" '
                f'text-anchor="middle">{value*100:.1f}%</text>',
                f'<text class="text small" x="{center}" y="{plot_bottom+22}" '
                f'text-anchor="middle">{label}</text>',
            ]
        )
    lines.append(
        f'<text class="text muted label" x="{(ordinary_left+ordinary_right)/2:.1f}" '
        f'y="{height-12}" text-anchor="middle">Training objective</text>'
    )

    lines.append("</svg>")
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
