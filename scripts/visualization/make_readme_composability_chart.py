"""Render the tracked composability ablation as an SVG for the README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERIES = [
    ("frozen", "Frozen backbone", "frozen"),
    ("clip_only", "Contrastive adapter", "contrastive"),
    ("clip_orth_smooth", "+ orthogonality loss", "orth"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="research/results/composability_ablation_k14.json",
    )
    parser.add_argument(
        "--out",
        default="docs/assets/composability-retrieval.svg",
    )
    args = parser.parse_args()

    metrics = json.loads((ROOT / args.metrics).read_text())
    subset_sizes = [int(value) for value in metrics["subset_sizes"]]
    width, height = 780, 430
    left, right, top, bottom = 74, 118, 70, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = 0.55

    def x_pos(index: int) -> float:
        return left + index * plot_width / (len(subset_sizes) - 1)

    def y_pos(value: float) -> float:
        return top + plot_height * (1 - value / y_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Compositional retrieval recall at ten</title>',
        '<desc id="desc">Recall at ten as more concept attributes are composed. '
        'The adapter trained with orthogonality loss improves substantially over the '
        'frozen backbone and contrastive-only adapter.</desc>',
        "<style>",
        ".text{fill:#1f2328;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}",
        ".muted{fill:#57606a}.grid{stroke:#d8dee4;stroke-width:1}.axis{stroke:#8c959f;stroke-width:1}",
        ".frozen{stroke:#6e7781;fill:#6e7781}.contrastive{stroke:#0969da;fill:#0969da}",
        ".orth{stroke:#1a7f37;fill:#1a7f37}.series{fill:none;stroke-width:3}",
        ".point{stroke-width:0}.label{font-size:13px}.small{font-size:12px}.title{font-size:17px;font-weight:500}",
        "@media(prefers-color-scheme:dark){.text{fill:#e6edf3}.muted{fill:#9da7b1}"
        ".grid{stroke:#30363d}.axis{stroke:#6e7681}.frozen{stroke:#9da7b1;fill:#9da7b1}"
        ".contrastive{stroke:#58a6ff;fill:#58a6ff}.orth{stroke:#3fb950;fill:#3fb950}}",
        "</style>",
        '<text class="text title" x="74" y="30">Compositional retrieval from partial attribute sets</text>',
        '<text class="text muted small" x="74" y="50">Higher is better · 1,854-image gallery</text>',
    ]

    for tick in range(0, 6):
        value = tick / 10
        y = y_pos(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(
            f'<text class="text muted small" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick*10}%</text>'
        )

    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
    ])
    for index, size in enumerate(subset_sizes):
        x = x_pos(index)
        lines.append(
            f'<text class="text muted small" x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle">{size}</text>'
        )
    lines.append(
        f'<text class="text muted label" x="{left+plot_width/2:.1f}" y="{height-12}" text-anchor="middle">Composed attributes (k)</text>'
    )
    lines.append(
        f'<text class="text muted label" transform="translate(20 {top+plot_height/2:.1f}) rotate(-90)" text-anchor="middle">Recall@10</text>'
    )

    for key, label, css_class in SERIES:
        values = [
            metrics["models"][key]["true_attributes"][str(size)]["R@10"]
            for size in subset_sizes
        ]
        points = " ".join(
            f"{x_pos(index):.1f},{y_pos(value):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(f'<polyline class="series {css_class}" points="{points}"/>')
        for index, value in enumerate(values):
            lines.append(
                f'<circle class="point {css_class}" cx="{x_pos(index):.1f}" cy="{y_pos(value):.1f}" r="4"/>'
            )
        final_y = y_pos(values[-1])
        lines.append(
            f'<text class="text label" x="{width-right+12}" y="{final_y+4:.1f}">{values[-1]*100:.1f}%</text>'
        )

    legend_x = [84, 270, 468]
    for (key, label, css_class), x in zip(SERIES, legend_x):
        lines.append(f'<line class="series {css_class}" x1="{x}" y1="{top-4}" x2="{x+24}" y2="{top-4}"/>')
        lines.append(f'<text class="text small" x="{x+32}" y="{top:.1f}">{label}</text>')

    lines.append("</svg>")
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
