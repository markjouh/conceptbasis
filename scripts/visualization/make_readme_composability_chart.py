"""Render the three-stage ConceptBasis composability chart for the README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="research/results/three_model_dev_composability.json",
        help="tracked three-model development metrics",
    )
    parser.add_argument(
        "--out",
        default="docs/assets/composability-retrieval.svg",
    )
    args = parser.parse_args()

    metrics = json.loads((ROOT / args.metrics).read_text())
    subset_sizes = [int(value) for value in metrics["subset_sizes"]]

    series = [
        (
            "contrastive",
            "Contrastive only",
            [
                metrics["models"]["contrastive"]["true_attributes"][str(size)]["R@5"]
                for size in subset_sizes
            ],
            "circle",
        ),
        (
            "groupmean",
            "+ group-mean orthogonality",
            [
                metrics["models"]["group_mean"]["true_attributes"][str(size)]["R@5"]
                for size in subset_sizes
            ],
            "square",
        ),
        (
            "reverse",
            "+ reverse-ridge orthogonality",
            [
                metrics["models"]["reverse_ridge"]["true_attributes"][str(size)]["R@5"]
                for size in subset_sizes
            ],
            "diamond",
        ),
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

    width, height = 860, 448
    left, right, top, bottom = 74, 232, 70, 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = 0.9

    def x_pos(index: int) -> float:
        return left + index * plot_width / (len(subset_sizes) - 1)

    def y_pos(value: float) -> float:
        return top + plot_height * (1 - value / y_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Compositional retrieval across three training objectives</title>',
        '<desc id="desc">Recall at five as one to fourteen concept attributes are composed. '
        'Reverse-ridge orthogonality performs best, followed by group-mean orthogonality and '
        'contrastive-only training.</desc>',
        "<style>",
        ".text{fill:#1f2328;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}",
        ".muted{fill:#57606a}.grid{stroke:#d8dee4;stroke-width:1}.axis{stroke:#8c959f;stroke-width:1}",
        ".contrastive{stroke:#6e7781}.point.contrastive{fill:#6e7781}",
        ".groupmean{stroke:#0969da}.point.groupmean{fill:#0969da}",
        ".reverse{stroke:#1a7f37}.point.reverse{fill:#1a7f37}",
        ".series{fill:none;stroke-width:3}.point{stroke-width:0}",
        ".label{font-size:13px}.small{font-size:12px}.title{font-size:17px;font-weight:500}",
        "@media(prefers-color-scheme:dark){.text{fill:#e6edf3}.muted{fill:#9da7b1}"
        ".grid{stroke:#30363d}.axis{stroke:#6e7681}.contrastive{stroke:#9da7b1}"
        ".point.contrastive{fill:#9da7b1}.groupmean{stroke:#58a6ff}.point.groupmean{fill:#58a6ff}"
        ".reverse{stroke:#3fb950}.point.reverse{fill:#3fb950}}",
        "</style>",
        '<text class="text title" x="74" y="30">Compositional retrieval improves with cleaner concept directions</text>',
        '<text class="text muted small" x="74" y="50">278-class dev gallery · all classes with ≥k attributes · Recall@5</text>',
    ]

    for tick in range(10):
        value = tick / 10
        y = y_pos(value)
        lines.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{width-right}" y2="{y:.1f}"/>'
        )
        lines.append(
            f'<text class="text muted small" x="{left-12}" y="{y+4:.1f}" '
            f'text-anchor="end">{tick*10}%</text>'
        )

    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height-bottom}" '
            f'x2="{width-right}" y2="{height-bottom}"/>',
        ]
    )
    for index, size in enumerate(subset_sizes):
        x = x_pos(index)
        lines.append(
            f'<text class="text small" x="{x:.1f}" y="{height-bottom+20}" '
            f'text-anchor="middle">{size}</text>'
        )
        lines.append(
            f'<text class="text muted small" x="{x:.1f}" y="{height-bottom+36}" '
            f'text-anchor="middle">n={cohort_sizes[index]}</text>'
        )
    lines.append(
        f'<text class="text muted label" x="{left+plot_width/2:.1f}" y="{height-12}" '
        'text-anchor="middle">Attributes in query (k)</text>'
    )
    lines.append(
        f'<text class="text muted label" transform="translate(20 {top+plot_height/2:.1f}) '
        'rotate(-90)" text-anchor="middle">Recall@5</text>'
    )

    for css_class, label, values, marker in series:
        path = " ".join(
            f"{'M' if index == 0 else 'L'} {x_pos(index):.1f} {y_pos(value):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(f'<path class="series {css_class}" d="{path}"/>')
        for index, value in enumerate(values):
            x, y = x_pos(index), y_pos(value)
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
        lines.append(
            f'<line class="series {css_class}" x1="{width-right}" y1="{final_y:.1f}" '
            f'x2="{width-right+14}" y2="{final_y:.1f}"/>'
        )
        lines.append(
            f'<text class="text label" x="{width-right+20}" y="{final_y-2:.1f}">{label}</text>'
        )
        lines.append(
            f'<text class="text muted small" x="{width-right+20}" y="{final_y+14:.1f}">'
            f'{values[-1]*100:.1f}% at k=14</text>'
        )

    lines.append("</svg>")
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
