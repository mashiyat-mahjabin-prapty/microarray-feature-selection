import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Match the final two-column manuscript width so exported labels remain close
# to the article body-text size after inclusion.
JOURNAL_TEXT_WIDTH_IN = 7.2
FONT_FAMILY = "serif"
BODY_FONT_SIZE = 8.5

DATASET_LABELS = {
    "Brain Cancer": "Brain cancer",
    "Brain Tumor": "Brain tumor",
    "Breast Cancer": "Breast cancer",
    "Colon Tumor": "Colon tumor",
    "Crohns Disease": "Crohn's disease",
    "Endometrial Cancer": "Endometrial cancer",
    "Gastric Cancer": "Gastric cancer",
    "Leukemia3": "Leukemia (3-class)",
    "Leukemia4": "Leukemia (4-class)",
    "Lung Cancer": "Lung cancer",
    "Ovarian Cancer": "Ovarian cancer",
    "Prostate Cancer": "Prostate cancer",
}


def clean_task(value):
    value = str(value).strip().lower()
    if value in {"binary", "bin"}:
        return "Binary"
    if value in {"multiclass", "multi-class", "multi class"}:
        return "Multiclass"
    return str(value).strip()


def proposed_label(row):
    method = str(row["method"]).lower()
    if "stack" in method:
        return "Proposed Stacking"
    if "lr" in method or "logistic" in method:
        return "Proposed LR"
    return "Proposed method"


def clean_method(value):
    value = str(value).strip()
    value = value.replace("CEFS (Voting)", "CEFS(Voting)")
    if value.lower() == "cefs(voting)":
        return "CEFS(Voting)"
    return value


def clean_dataset(value):
    value = str(value).strip()
    return DATASET_LABELS.get(value, value)


def build_method_colors(data):
    methods = sorted(
        data.loc[~data["is_proposed_bool"], "method_display"].dropna().unique()
    )
    cmap = plt.get_cmap("tab20")
    return {method: cmap(i % cmap.N) for i, method in enumerate(methods)}


def plot_panel(ax, data, task, rng, method_colors):
    sub = data[data["task"] == task].copy()
    if sub.empty:
        ax.set_visible(False)
        return

    # Sort datasets by best proposed accuracy, then keep all remaining.
    proposed = sub[sub["is_proposed_bool"]].copy()
    order = (
        proposed.groupby("dataset")["accuracy"]
        .max()
        .sort_values()
        .index
        .tolist()
    )
    remaining = [d for d in sub["dataset"].drop_duplicates() if d not in order]
    order = order + remaining

    y_lookup = {dataset: i for i, dataset in enumerate(order)}

    lit = sub[~sub["is_proposed_bool"]].copy()
    prop = sub[sub["is_proposed_bool"]].copy()

    # Literature points: method-specific colors, lightly jittered.
    if not lit.empty:
        for method, group in lit.groupby("method_display", sort=True):
            y = group["dataset"].map(y_lookup).to_numpy(dtype=float)
            y = y + rng.normal(0, 0.075, size=len(y))
            ax.scatter(
                group["accuracy"],
                y,
                s=28,
                color=method_colors[method],
                edgecolor="#4A4A4A",
                linewidth=0.45,
                alpha=0.88,
                zorder=2,
            )

    # Proposed LR and Stacking: highlighted.
    if not prop.empty:
        prop["proposed_label"] = prop.apply(proposed_label, axis=1)

        lr = prop[prop["proposed_label"].eq("Proposed LR")]
        if not lr.empty:
            ax.scatter(
                lr["accuracy"],
                lr["dataset"].map(y_lookup).to_numpy(dtype=float) - 0.08,
                s=78,
                marker="^",
                color="#4C78A8",
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )

        stacking = prop[prop["proposed_label"].eq("Proposed Stacking")]
        if not stacking.empty:
            ax.scatter(
                stacking["accuracy"],
                stacking["dataset"].map(y_lookup).to_numpy(dtype=float) + 0.08,
                s=78,
                marker="v",
                color="#F58518",
                edgecolor="black",
                linewidth=1.0,
                zorder=6,
            )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    panel_label = "(a)" if task == "Binary" else "(b)"
    ax.set_title(f"{panel_label} {task} datasets", fontsize=9.5, pad=5)
    ax.set_xlabel("Accuracy", fontsize=BODY_FONT_SIZE)
    ax.set_xlim(0.75, 1.02)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_handles = []
    for method in sorted(lit["method_display"].dropna().unique()):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=method_colors[method],
                markeredgecolor="#4A4A4A",
                markersize=5,
                label=method,
            )
        )

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#4C78A8",
                markeredgecolor="black",
                markersize=6,
                label="Proposed LR",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color="none",
                markerfacecolor="#F58518",
                markeredgecolor="black",
                markersize=6,
                label="Proposed Stacking",
            ),
        ]
    )
    ax.legend(
        handles=legend_handles,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=True,
        fontsize=7.4,
        title_fontsize=8,
        borderaxespad=0.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="outputs/literature_comparison_tidy.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="outputs/literature_comparison_combined",
    )
    parser.add_argument("--formats", nargs="+", default=["tif", "eps", "png"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["task"] = df["task"].map(clean_task)
    df["dataset"] = df["dataset"].map(clean_dataset)
    df["method_display"] = df["method"].map(clean_method)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.dropna(subset=["accuracy", "dataset", "method"]).copy()

    if "is_proposed" not in df.columns:
        df["is_proposed"] = df["method"].astype(str).str.contains(
            "proposed", case=False, na=False
        )

    df["is_proposed_bool"] = (
        df["is_proposed"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"yes", "true", "1", "proposed"})
        | df["method"].astype(str).str.contains("proposed", case=False, na=False)
    )

    rng = np.random.default_rng(42)
    method_colors = build_method_colors(df)

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BODY_FONT_SIZE,
            "axes.labelsize": BODY_FONT_SIZE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(JOURNAL_TEXT_WIDTH_IN, 8.2),
        sharex=True,
    )

    plot_panel(axes[0], df, "Binary", rng, method_colors)
    plot_panel(axes[1], df, "Multiclass", rng, method_colors)
    fig.subplots_adjust(
        left=0.18,
        right=0.99,
        bottom=0.17,
        top=0.93,
        hspace=1.05,
    )

    fig.suptitle(
        "Contextual comparison with reported microarray classification results",
        fontsize=10,
        y=0.985,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"png", "tif", "tiff"}:
            kwargs["dpi"] = 300
        fig.savefig(out_prefix.with_suffix(f".{fmt}"), **kwargs)

    plt.close(fig)


if __name__ == "__main__":
    main()
