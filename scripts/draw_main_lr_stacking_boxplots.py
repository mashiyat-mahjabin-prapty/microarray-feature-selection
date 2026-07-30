import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# The manuscript includes this figure at full two-column width.  Building the
# figure at that width keeps the exported font sizes close to the body text
# instead of shrinking a very large canvas in LaTeX.
JOURNAL_TEXT_WIDTH_IN = 7.2
FONT_FAMILY = "serif"
BODY_FONT_SIZE = 8.5


DATASET_LABELS = {
    "Binary_Adenocarcinoma": "Adenocarcinoma",
    "Binary_BrainTumor": "Brain tumor",
    "Binary_BreastCancer": "Breast cancer",
    "Binary_ColonTumor": "Colon tumor",
    "Binary_Gastric": "Gastric cancer",
    "Binary_Leukemia": "Leukemia",
    "Binary_Lung": "Lung cancer",
    "Binary_Lymphoma": "Lymphoma",
    "Binary_Myeloma": "Myeloma",
    "Binary_OvarianCancer": "Ovarian cancer",
    "Binary_Prostate": "Prostate cancer",
    "Multiclass_BrainCancer": "Brain cancer",
    "Multiclass_Crohns": "Crohn's disease",
    "Multiclass_EndometrialCancer": "Endometrial cancer",
    "Multiclass_Glioma": "Glioma",
    "Multiclass_Leukemia_3": "Leukemia\n(3-class)",
    "Multiclass_Leukemia_4": "Leukemia\n(4-class)",
    "Multiclass_LungCancer": "Lung cancer",
    "Multiclass_Lymphoma": "Lymphoma",
    "Multiclass_MLL": "MLL",
    "Multiclass_SRBCT": "SRBCT",
}

BINARY_ORDER = [
    "Adenocarcinoma",
    "Brain tumor",
    "Breast cancer",
    "Colon tumor",
    "Gastric cancer",
    "Leukemia",
    "Lung cancer",
    "Lymphoma",
    "Myeloma",
    "Ovarian cancer",
    "Prostate cancer",
]

MULTICLASS_ORDER = [
    "Brain cancer",
    "Crohn's disease",
    "Endometrial cancer",
    "Glioma",
    "Leukemia\n(3-class)",
    "Leukemia\n(4-class)",
    "Lung cancer",
    "Lymphoma",
    "MLL",
    "SRBCT",
]


def collect_fold_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "violin_plots":
            continue

        metrics_path = dataset_dir / "ml_fold_metrics.csv"
        if not metrics_path.exists():
            continue

        df = pd.read_csv(metrics_path)
        df = df[df["model"].isin(["lr", "stacking"])].copy()
        if df.empty:
            continue

        df["dataset_key"] = dataset_dir.name
        df["dataset"] = df["dataset_key"].map(DATASET_LABELS).fillna(dataset_dir.name)
        df["task"] = np.where(df["dataset_key"].str.startswith("Binary_"), "Binary", "Multiclass")
        df["model_label"] = df["model"].map({"lr": "LR", "stacking": "Stacking"})
        rows.append(df)

    if not rows:
        raise ValueError(f"No LR/stacking fold metrics found under {results_dir}")

    out = pd.concat(rows, ignore_index=True)
    out["f1_weighted"] = pd.to_numeric(out["f1_weighted"], errors="coerce")
    out = out.dropna(subset=["f1_weighted"])
    return out


def draw_panel(ax, data: pd.DataFrame, task: str, order: list[str], panel_label: str) -> None:
    sub = data[data["task"] == task].copy()

    sns.boxplot(
        data=sub,
        x="dataset",
        y="f1_weighted",
        hue="model_label",
        order=order,
        hue_order=["LR", "Stacking"],
        palette={"LR": "#4C78A8", "Stacking": "#F58518"},
        width=0.62,
        linewidth=1.0,
        fliersize=0,
        ax=ax,
    )

    sns.stripplot(
        data=sub,
        x="dataset",
        y="f1_weighted",
        hue="model_label",
        order=order,
        hue_order=["LR", "Stacking"],
        dodge=True,
        jitter=0.12,
        palette={"LR": "#4C78A8", "Stacking": "#F58518"},
        size=2.3,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.25,
        ax=ax,
    )

    ax.set_title(f"{panel_label} {task} datasets", fontsize=9.5, pad=5)
    ax.set_xlabel("")
    ax.set_ylabel("Weighted F1", fontsize=BODY_FONT_SIZE)
    ax.set_ylim(0.65, 1.03)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.tick_params(axis="x", rotation=32, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Remove duplicated legends from each panel; a shared legend is added later.
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="outputs/model_eval",
    )
    parser.add_argument(
        "--out-prefix",
        default="outputs/figures/lr_stacking_weighted_f1_boxplots",
    )
    parser.add_argument("--formats", nargs="+", default=["tif", "eps", "png"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = collect_fold_metrics(results_dir)

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font=FONT_FAMILY,
        rc={
            "font.size": BODY_FONT_SIZE,
            "axes.labelsize": BODY_FONT_SIZE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(JOURNAL_TEXT_WIDTH_IN, 6.6),
        sharey=True,
    )

    draw_panel(axes[0], data, "Binary", BINARY_ORDER, "(a)")
    draw_panel(axes[1], data, "Multiclass", MULTICLASS_ORDER, "(b)")

    handles, labels = axes[0].get_legend_handles_labels()
    # Boxplot + stripplot both create handles, so keep first two unique labels.
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique and label in {"LR", "Stacking"}:
            unique[label] = handle

    fig.legend(
        unique.values(),
        unique.keys(),
        title="Classifier",
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=8.2,
        title_fontsize=8.2,
    )

    fig.suptitle(
        "Post-selection repeated-CV weighted F1 performance of selected feature panels",
        fontsize=10,
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.20,
        top=0.92,
        hspace=0.58,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        fmt = fmt.lower().lstrip(".")
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tif", "tiff"}:
            kwargs["dpi"] = 300
        fig.savefig(out_prefix.with_suffix(f".{fmt}"), **kwargs)

    plt.close(fig)

    print(f"Saved figure to: {out_prefix}")


if __name__ == "__main__":
    main()
