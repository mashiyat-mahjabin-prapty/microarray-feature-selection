# scripts/plot_gene_expression_by_class.py

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GENES = ["ZYX", "MGST1", "CD33", "USP9X", "AKR1C1"]


def clean_class_label(value):
    text = str(value).strip()
    if len(text) >= 3 and text.startswith("b'") and text.endswith("'"):
        return text[2:-1]
    if len(text) >= 3 and text.startswith('b"') and text.endswith('"'):
        return text[2:-1]
    return text


def load_mapping(mapping_csv):
    mapping = pd.read_csv(mapping_csv)

    mapping = mapping.rename(columns={
        "gene_name": "gene_symbol",
        "gene symbol": "gene_symbol",
        "Gene Symbol": "gene_symbol",
        "feature": "probe_id",
        "probe": "probe_id",
        "probe_id": "probe_id",
    })

    if "gene_symbol" not in mapping.columns or "probe_id" not in mapping.columns:
        raise ValueError("Mapping file must contain gene_symbol and probe_id/feature columns")

    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.strip()
    mapping["probe_id"] = mapping["probe_id"].astype(str).str.strip()

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--out", default="leukemia_gene_expression_boxplots")
    parser.add_argument("--genes", nargs="+", default=GENES)
    parser.add_argument("--formats", nargs="+", default=["tif", "eps"])
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    mapping = load_mapping(args.mapping_csv)

    if "CLASS" not in df.columns:
        raise ValueError("Dataset must contain CLASS column")

    gene_to_probe = {}
    for gene in args.genes:
        matches = mapping[mapping["gene_symbol"].str.upper() == gene.upper()]
        probes = [p for p in matches["probe_id"].tolist() if p in df.columns]
        if probes:
            gene_to_probe[gene] = probes

    if not gene_to_probe:
        raise ValueError("None of the requested genes could be mapped to dataset columns")

    plot_df_rows = []
    for gene, probes in gene_to_probe.items():
        expression = df[probes].mean(axis=1)
        for class_name, value in zip(df["CLASS"].map(clean_class_label), expression):
            plot_df_rows.append({
                "gene": gene,
                "class": class_name,
                "expression": value,
            })

    plot_df = pd.DataFrame(plot_df_rows)

    genes = list(gene_to_probe.keys())
    n_genes = len(genes)

    fig, axes = plt.subplots(
        1,
        n_genes,
        figsize=(3.0 * n_genes, 4.2),
        sharey=False,
    )

    if n_genes == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, gene in zip(axes, genes):
        gene_df = plot_df[plot_df["gene"] == gene]
        classes = sorted(gene_df["class"].unique())

        data = [
            gene_df.loc[gene_df["class"] == cls, "expression"].to_numpy()
            for cls in classes
        ]

        try:
            box = ax.boxplot(
                data,
                tick_labels=classes,
                patch_artist=True,
                showfliers=False,
                widths=0.55,
            )
        except TypeError:
            box = ax.boxplot(
                data,
                labels=classes,
                patch_artist=True,
                showfliers=False,
                widths=0.55,
            )

        colors = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        for i, values in enumerate(data, start=1):
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                s=24,
                color="#222222",
                alpha=0.7,
                linewidths=0,
            )

        ax.set_title(gene)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Expression")
    if args.title:
        fig.suptitle(args.title, y=1.03)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        suffix = fmt.lower().lstrip(".")
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tif", "tiff"}:
            kwargs["dpi"] = 300
        fig.savefig(out.with_suffix(f".{suffix}"), **kwargs)

    plot_df.to_csv(out.with_suffix(".csv"), index=False)

    print("Plotted genes:")
    for gene, probes in gene_to_probe.items():
        print(f"{gene}: {', '.join(probes)}")


if __name__ == "__main__":
    main()
