import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


JOURNAL_COLUMN_WIDTH_IN = 3.5


def add_box(ax, center, text, width=8.2, height=0.74, face="#F4F7FB", edge="#2F4156",
            fontsize=10.2, weight="normal"):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1F2933",
    )
    return patch


def add_arrow(ax, start, end, color="#2F4156", lw=1.2, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Directory for the workflow figure files.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "tif", "eps", "pdf", "svg"],
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )

    # Build at the journal's one-column width so LaTeX does not shrink the
    # typography after export.
    fig, ax = plt.subplots(figsize=(JOURNAL_COLUMN_WIDTH_IN, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18.3)
    ax.axis("off")

    main_x = 5
    ys = [16.6, 15.35, 14.1, 12.85, 11.6, 10.35, 9.1]
    labels = [
        "Microarray datasets",
        "Data preprocessing",
        "Cross-validated screening",
        "XGBoost feature ranking",
        "Rank aggregation",
        "Top-ranked probes",
        "SVM-RFECV refinement",
    ]

    for y, label in zip(ys, labels):
        add_box(ax, (main_x, y), label)

    for y1, y2 in zip(ys[:-1], ys[1:]):
        add_arrow(ax, (main_x, y1 - 0.38), (main_x, y2 + 0.38))

    consensus_y = 7.8
    add_box(
        ax,
        (main_x, consensus_y),
        "Consensus biomarkers",
        face="#E8F2EC",
        edge="#2F6B4F",
        weight="bold",
    )
    add_arrow(ax, (main_x, ys[-1] - 0.38), (main_x, consensus_y + 0.38), color="#2F6B4F")

    # A grouped vertical layout preserves the parallel downstream analyses
    # without forcing three illegibly narrow boxes across one column.
    group_bottom = 2.55
    group_top = 6.85
    group = FancyBboxPatch(
        (0.65, group_bottom),
        8.7,
        group_top - group_bottom,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        edgecolor="#6B7280",
        facecolor="#FFFFFF",
    )
    ax.add_patch(group)
    ax.text(
        5,
        6.52,
        "Downstream analyses",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color="#374151",
    )

    add_box(
        ax,
        (main_x, 5.65),
        "Post-selection panel\nassessment",
        width=7.4,
        height=0.92,
        face="#EEF4FA",
        edge="#345E8A",
        fontsize=9.7,
    )
    add_box(
        ax,
        (main_x, 4.45),
        "SHAP attribution",
        width=7.4,
        face="#FFF4E6",
        edge="#9A5A17",
        fontsize=9.9,
    )
    add_box(
        ax,
        (main_x, 3.25),
        "Functional enrichment",
        width=7.4,
        face="#F7F0FA",
        edge="#6B4C7D",
        fontsize=9.9,
    )

    interp_pos = (5, 1.45)
    add_box(
        ax,
        interp_pos,
        "Biological interpretation",
        width=8.2,
        face="#F1F1F1",
        edge="#3F3F46",
        weight="bold",
    )

    add_arrow(
        ax,
        (main_x, consensus_y - 0.38),
        (main_x, group_top + 0.02),
        color="#2F6B4F",
    )
    add_arrow(
        ax,
        (main_x, group_bottom - 0.02),
        (main_x, interp_pos[1] + 0.38),
        color="#3F3F46",
    )

    ax.text(
        5,
        17.72,
        "Proposed biomarker-discovery workflow",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#111827",
    )

    fig.tight_layout(pad=0.15)

    base = out_dir / "proposed_workflow"
    for ext in args.formats:
        fig.savefig(base.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved workflow figure files to {out_dir}")


if __name__ == "__main__":
    main()
