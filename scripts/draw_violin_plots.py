"""Draw violin plots from supervisor biomarker pipeline fold metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "roc_auc",
    "average_precision",
]

MODEL_ORDER = ["lr", "svm", "rf", "xgb", "voting", "stacking"]
MODEL_LABELS = {
    "lr": "LR",
    "svm": "SVM",
    "rf": "RF",
    "xgb": "XGB",
    "voting": "Voting",
    "stacking": "Stacking",
}
MODEL_FULL_NAMES = {
    "lr": "Logistic Regression",
    "svm": "Support Vector Classifier",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "voting": "Voting Classifier",
    "stacking": "Stacking Classifier",
}
COLORS = [
    "#2F5BEA",
    "#E77716",
    "#2DBE4F",
    "#E01E37",
    "#8E44D9",
    "#9A4F12",
    "#DF65B0",
    "#8C8C8C",
    "#7A7A7A",
    "#17B7D4",
    "#4C78A8",
    "#54A24B",
]


def metric_label(metric: str) -> str:
    return {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "precision_macro": "Macro precision",
        "recall_macro": "Macro recall",
        "f1_macro": "Macro F1",
        "f1_weighted": "Weighted F1",
        "mcc": "MCC",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average precision",
    }.get(metric, metric)


def title_metric_label(metric: str) -> str:
    return {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1_macro": "Macro F1 Score",
        "f1_weighted": "F1 Score",
        "mcc": "MCC",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average Precision",
    }.get(metric, metric_label(metric))


def short_dataset_label(dataset: str) -> str:
    if dataset.startswith("Binary_"):
        return dataset.replace("Binary_", "", 1)
    if dataset.startswith("Multiclass_"):
        return dataset.replace("Multiclass_", "", 1)
    return dataset


def discover_metrics_dirs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/ml_fold_metrics.csv"))


def load_fold_metrics(results_dir: Path) -> pd.DataFrame:
    paths = discover_metrics_dirs(results_dir)
    if not paths:
        raise FileNotFoundError(f"No ml_fold_metrics.csv files found under {results_dir}")

    frames = []
    for path in paths:
        dataset = path.parent.name
        task = dataset.split("_", 1)[0] if "_" in dataset else "Unknown"
        df = pd.read_csv(path)
        df.insert(0, "dataset", dataset)
        df.insert(1, "task", task)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def ordered_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].dropna().astype(str))
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def set_score_axis(ax: plt.Axes, metric: str) -> None:
    if metric == "mcc":
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#808080", linewidth=0.8, linestyle="--", alpha=0.7)
    else:
        ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def set_dataset_score_axis(ax: plt.Axes, metric: str, data: list[np.ndarray]) -> None:
    if metric == "mcc":
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#808080", linewidth=0.8, linestyle="--", alpha=0.7)
    else:
        finite_values = np.concatenate([values[np.isfinite(values)] for values in data])
        lower = min(0.50, float(np.floor(finite_values.min() * 10) / 10) - 0.02)
        ax.set_ylim(max(-0.03, lower), 1.10)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_path: Path, formats: list[str]) -> None:
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        save_kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tif", "tiff"}:
            save_kwargs["dpi"] = 300
        fig.savefig(out_path.with_suffix(f".{suffix}"), **save_kwargs)


def draw_violin_by_model(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    formats: list[str],
) -> None:
    models = ordered_models(df)
    data = []
    labels = []
    colors = []

    for idx, model in enumerate(models):
        values = pd.to_numeric(df.loc[df["model"] == model, metric], errors="coerce").dropna()
        if values.empty:
            continue
        data.append(values.to_numpy())
        labels.append(MODEL_LABELS.get(model, model))
        colors.append(COLORS[idx % len(COLORS)])

    if not data:
        return

    fig_width = max(7.0, 1.15 * len(data))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.78,
    )

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)

    if "cmedians" in parts:
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.6)

    rng = np.random.default_rng(42)
    for pos, values in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=11,
            color="#1A1A1A",
            alpha=0.28,
            linewidths=0,
        )
        ax.scatter(
            pos,
            float(np.mean(values)),
            marker="D",
            s=34,
            color="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=4,
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_label(metric))
    ax.set_xlabel("Classifier")
    ax.set_title(title, pad=12)
    set_score_axis(ax, metric)

    fig.tight_layout()
    save_figure(fig, out_path, formats)
    plt.close(fig)


def draw_dataset_panels(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    formats: list[str],
) -> None:
    dataset_dir = out_dir / "per_dataset" / metric
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for dataset, dataset_df in sorted(df.groupby("dataset")):
        safe_name = dataset.replace("/", "_").replace("\\", "_")
        title = f"{dataset}: {metric_label(metric)}"
        draw_violin_by_model(
            dataset_df,
            metric,
            dataset_dir / f"{safe_name}_{metric}",
            title,
            formats,
        )


def draw_task_panels(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    formats: list[str],
) -> None:
    for task, task_df in sorted(df.groupby("task")):
        title = f"{task} datasets: {metric_label(metric)}"
        draw_violin_by_model(
            task_df,
            metric,
            out_dir / f"{task.lower()}_datasets_by_model_{metric}",
            title,
            formats,
        )


def draw_metric_by_dataset_for_model(
    df: pd.DataFrame,
    metric: str,
    model: str,
    task: str,
    out_path: Path,
    formats: list[str],
    plot_type: str,
) -> None:
    model_df = df[(df["model"] == model) & (df["task"] == task)].copy()
    if model_df.empty:
        return

    data = []
    labels = []
    colors = []
    for idx, (dataset, dataset_df) in enumerate(sorted(model_df.groupby("dataset"))):
        values = pd.to_numeric(dataset_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        data.append(values.to_numpy())
        labels.append(short_dataset_label(dataset))
        colors.append(COLORS[idx % len(COLORS)])

    if not data:
        return

    fig_width = max(9.2, 0.78 * len(data))
    fig, ax = plt.subplots(figsize=(fig_width, 3.6))

    if plot_type in {"violin", "violin_points"}:
        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.82,
        )

        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#222222")
            body.set_linewidth(0.8)
            body.set_alpha(0.95)

        if "cmedians" in parts:
            parts["cmedians"].set_color("#343434")
            parts["cmedians"].set_linewidth(2.0)

    if plot_type in {"box", "box_points"}:
        box = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.7},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#222222")
            patch.set_alpha(0.82)

    if plot_type in {"points", "box_points", "violin_points"}:
        rng = np.random.default_rng(42)
        for pos, values in enumerate(data, start=1):
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=15,
                color="#222222",
                alpha=0.42,
                linewidths=0,
                zorder=3,
            )

    if plot_type == "mean_sd":
        means = [float(np.mean(values)) for values in data]
        stds = [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in data]
        positions = np.arange(1, len(data) + 1)
        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=3,
            zorder=2,
        )
        ax.scatter(
            positions,
            means,
            s=52,
            color=colors,
            edgecolor="#111111",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric_label(metric))
    ax.set_title(
        f"{title_metric_label(metric)} of {MODEL_FULL_NAMES.get(model, model)} "
        f"on Different {task} Datasets",
        fontsize=10,
        pad=12,
    )
    set_dataset_score_axis(ax, metric, data)

    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.8)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(
        0.5,
        0.01,
        MODEL_LABELS.get(model, model),
        ha="center",
        va="bottom",
        fontsize=14,
        family="serif",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats)
    plt.close(fig)


def draw_model_dataset_panels(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    models: list[str] | None = None,
    formats: list[str] | None = None,
    plot_types: list[str] | None = None,
) -> None:
    models_to_plot = models if models is not None else ordered_models(df)
    output_formats = formats if formats is not None else ["png", "pdf"]
    output_plot_types = plot_types if plot_types is not None else ["violin"]

    for plot_type in output_plot_types:
        model_dir = out_dir / "by_model_dataset" / plot_type / metric
        model_dir.mkdir(parents=True, exist_ok=True)
        for task in ["Binary", "Multiclass"]:
            for model in models_to_plot:
                draw_metric_by_dataset_for_model(
                    df,
                    metric,
                    model,
                    task,
                    model_dir / f"{task.lower()}_{model}_{metric}_{plot_type}",
                    output_formats,
                    plot_type,
                )


def make_best_model_table(df: pd.DataFrame, out_dir: Path, primary_metric: str) -> None:
    summary = (
        df.groupby(["dataset", "task", "model"], as_index=False)
        .agg(
            metric_mean=(primary_metric, "mean"),
            metric_std=(primary_metric, "std"),
            n_folds=(primary_metric, "count"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_weighted_mean=("f1_weighted", "mean"),
            mcc_mean=("mcc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            average_precision_mean=("average_precision", "mean"),
        )
    )
    summary = summary.sort_values(
        ["dataset", "metric_mean", "mcc_mean"],
        ascending=[True, False, False],
    )
    best = summary.groupby("dataset", as_index=False).head(1)
    summary.to_csv(out_dir / "model_metric_summary_from_folds.csv", index=False)
    best.to_csv(out_dir / f"best_model_by_dataset_{primary_metric}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="outputs/model_eval",
        help="Directory containing per-dataset supervisor pipeline outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output plot directory. Defaults to <results-dir>/violin_plots.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to plot.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Image formats to save, e.g. png pdf tif eps.",
    )
    parser.add_argument(
        "--primary-metric",
        default="balanced_accuracy",
        help="Metric used to choose best_model_by_dataset table.",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Also draw one violin plot per dataset per metric.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to use for model-vs-dataset plots. Defaults to all models.",
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["violin", "box", "points", "box_points", "mean_sd"],
        choices=["violin", "violin_points", "box", "points", "box_points", "mean_sd"],
        help="Plot styles for model-specific dataset plots.",
    )
    parser.add_argument(
        "--no-model-dataset-plots",
        action="store_true",
        help="Skip model-specific plots with datasets on the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "violin_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.lower().lstrip(".") for fmt in args.formats]

    df = load_fold_metrics(results_dir)
    df.to_csv(out_dir / "combined_fold_metrics.csv", index=False)

    available_metrics = [metric for metric in args.metrics if metric in df.columns]
    missing_metrics = sorted(set(args.metrics).difference(available_metrics))
    if missing_metrics:
        print(f"Skipping missing metrics: {', '.join(missing_metrics)}")

    make_best_model_table(df, out_dir, args.primary_metric)

    for metric in available_metrics:
        draw_violin_by_model(
            df,
            metric,
            out_dir / f"all_datasets_by_model_{metric}",
            f"All datasets: {metric_label(metric)}",
            formats,
        )
        draw_task_panels(df, metric, out_dir, formats)
        if not args.no_model_dataset_plots:
            draw_model_dataset_panels(
                df,
                metric,
                out_dir,
                args.models,
                formats,
                args.plot_types,
            )
        if args.per_dataset:
            draw_dataset_panels(df, metric, out_dir, formats)

    print(f"Loaded {df['dataset'].nunique()} datasets")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
