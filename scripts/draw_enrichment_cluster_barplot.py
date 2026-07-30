"""Draw enrichment cluster bar plots with gene-count bubbles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SOURCE_COLORS = {
    "GO:BP": "#4C78A8",
    "GO:CC": "#72B7B2",
    "GO:MF": "#54A24B",
    "KEGG_PATHWAY": "#F58518",
    "WIKIPATHWAYS": "#E45756",
    "REACTOME_PATHWAY": "#B279A2",
    "DISGENET": "#9D755D",
}

SOURCE_LABELS = {
    "GO:BP": "GO:BP",
    "GO:CC": "GO:CC",
    "GO:MF": "GO:MF",
    "KEGG_PATHWAY": "KEGG",
    "WIKIPATHWAYS": "WikiPathways",
    "REACTOME_PATHWAY": "Reactome",
    "DISGENET": "DisGeNET",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        clean = col.strip()
        if clean in {";log10(q)", "-log10(q)", "log10(q)", "Log10(q)"}:
            rename[col] = "neg_log10_q"
        elif clean in {"Num_Genes", "Num_genes", "num_genes", "GeneCount"}:
            rename[col] = "num_genes"
        elif clean in {"Enrichment_Term", "Term", "Description"}:
            rename[col] = "term"
        elif clean in {"Source", "source"}:
            rename[col] = "source"
        elif clean in {"Genes", "genes"}:
            rename[col] = "genes"

    df = df.rename(columns=rename)
    required = {"source", "term", "neg_log10_q", "num_genes"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    df["source"] = df["source"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    df["neg_log10_q"] = pd.to_numeric(df["neg_log10_q"], errors="coerce")
    df["num_genes"] = pd.to_numeric(df["num_genes"], errors="coerce")
    df = df.dropna(subset=["neg_log10_q", "num_genes"])
    return df


def read_clusters(path: Path, top_n: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = df.sort_values("neg_log10_q", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    return df.sort_values("neg_log10_q", ascending=True).reset_index(drop=True)


def save_figure(fig: plt.Figure, out_path: Path, formats: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tif", "tiff"}:
            kwargs["dpi"] = 300
        fig.savefig(out_path.with_suffix(f".{suffix}"), **kwargs)


def draw_barplot(
    data: pd.DataFrame,
    title: str,
    out_path: Path,
    formats: list[str],
    bubble_scale: float,
    base_font_size: int,
    compact: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 4,
            "axes.labelsize": base_font_size + 1,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": max(9, base_font_size - 2),
            "legend.title_fontsize": max(10, base_font_size - 1),
        }
    )
    fig_height = max(6.0, 0.47 * len(data) + 1.9)
    fig_width = 10.2 if compact else 12.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_positions = np.arange(len(data))
    colors = [
        SOURCE_COLORS.get(source, "#7A7A7A")
        for source in data["source"]
    ]

    ax.barh(
        y_positions,
        data["neg_log10_q"],
        color=colors,
        edgecolor="white",
        height=0.48,
    )

    bubble_sizes = data["num_genes"] * bubble_scale
    ax.scatter(
        data["neg_log10_q"],
        y_positions,
        s=bubble_sizes,
        c=colors,
        edgecolors="#222222",
        linewidths=0.45,
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["term"])
    ax.set_xlabel("-log10(q)")
    ax.set_title(title, pad=13)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, float(data["neg_log10_q"].max()) * 1.045)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    source_handles = [
        Patch(
            facecolor=SOURCE_COLORS.get(source, "#7A7A7A"),
            label=SOURCE_LABELS.get(source, source),
        )
        for source in sorted(data["source"].unique())
    ]
    gene_counts = np.unique(data["num_genes"].astype(int))
    if len(gene_counts) > 4:
        gene_counts = np.unique(
            np.round(np.linspace(gene_counts.min(), gene_counts.max(), 4)).astype(int)
        )
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=f"{count} genes",
            markerfacecolor="#9A9A9A",
            markeredgecolor="#222222",
            markersize=np.sqrt(count * bubble_scale),
        )
        for count in gene_counts
    ]
    if compact:
        source_legend = fig.legend(
            handles=source_handles,
            title="Source",
            loc="lower center",
            bbox_to_anchor=(0.39, 0.015),
            ncol=min(4, len(source_handles)),
            frameon=True,
            borderaxespad=0.0,
        )
        fig.legend(
            handles=size_handles,
            title="Gene count",
            loc="lower center",
            bbox_to_anchor=(0.83, 0.015),
            ncol=2,
            frameon=True,
            borderaxespad=0.0,
        )
    else:
        source_legend = ax.legend(
            handles=source_handles,
            title="Source",
            loc="lower right",
            frameon=True,
        )
        ax.add_artist(source_legend)
        ax.legend(
            handles=size_handles,
            title="Gene count",
            loc="center right",
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0.16 if compact else 0, 1, 1))
    save_figure(fig, out_path, formats)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters",
        default=(
            "outputs/model_eval/"
            "Binary_Adenocarcinoma/adenocarcinoma_selected_clusters.txt"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "outputs/figures/"
            "adenocarcinoma_selected_clusters_barplot"
        ),
    )
    parser.add_argument("--title", default="Adenocarcinoma Enrichment Clusters")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["eps", "tif"])
    parser.add_argument("--bubble-scale", type=float, default=35.0)
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Use the older wide layout with legends inside the axes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_clusters(Path(args.clusters), args.top_n)
    draw_barplot(
        data=data,
        title=args.title,
        out_path=Path(args.out),
        formats=args.formats,
        bubble_scale=args.bubble_scale,
        base_font_size=args.font_size,
        compact=not args.wide,
    )
    print(f"Plotted {len(data)} enrichment terms")
    print(f"Output prefix: {args.out}")


if __name__ == "__main__":
    main()
