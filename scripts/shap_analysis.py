"""Post-hoc SHAP analysis for biomarker pipeline outputs.

This script trains one final XGBoost interpretation model using the selected
consensus genes for a dataset, then writes SHAP rankings and figures with gene
symbols as feature names. It is for interpretation only, not performance
estimation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
CLASS_COLORS = [
    "#1E88E5",
    "#D81B60",
    "#8A9A00",
    "#00A676",
    "#7E57C2",
    "#F4511E",
    "#6D4C41",
]


def clean_class_label(value: object) -> str:
    text = str(value).strip()
    if len(text) >= 3 and text.startswith("b'") and text.endswith("'"):
        return text[2:-1]
    if len(text) >= 3 and text.startswith('b"') and text.endswith('"'):
        return text[2:-1]
    return text


def save_figure(fig: plt.Figure, out_path: Path, formats: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tif", "tiff"}:
            kwargs["dpi"] = 300
        fig.savefig(out_path.with_suffix(f".{suffix}"), **kwargs)


def make_xgb(n_classes: int, seed: int, n_jobs: int) -> XGBClassifier:
    if n_classes == 2:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=n_jobs,
        )

    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
    )


def load_dataset_path(dataset_name: str, manifest_path: Path) -> Path:
    manifest = pd.read_csv(manifest_path)
    manifest["dataset_name"] = manifest["task"].astype(str) + "_" + manifest["dataset"].astype(str)
    match = manifest[manifest["dataset_name"] == dataset_name]
    if match.empty:
        raise ValueError(f"{dataset_name} was not found in {manifest_path}")
    dataset_path = Path(match.iloc[0]["path"])
    if dataset_path.is_absolute():
        return dataset_path

    repository_relative = ROOT / dataset_path
    if repository_relative.exists():
        return repository_relative

    raise FileNotFoundError(
        f"The manifest path for {dataset_name} does not resolve inside the "
        f"repository: {dataset_path}"
    )


def path_for_metadata(path: Path | None) -> str | None:
    """Return a portable path for saved run metadata."""
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external-data>/{resolved.name}"


def choose_feature_file(dataset_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    preferred = dataset_dir / "rfecv_feature_consensus_threshold.csv"
    if preferred.exists():
        return preferred

    fallback = dataset_dir / "final_selected_features.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"No RFECV consensus or final selected feature file found in {dataset_dir}"
    )


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "Gene Symbol": "gene_name",
            "gene symbol": "gene_name",
            "gene_symbol": "gene_name",
            "GeneSymbol": "gene_name",
            "symbol": "gene_name",
            "Symbol": "gene_name",
            "gene": "gene_name",
            "Gene": "gene_name",
            "probe": "feature",
            "Probe": "feature",
            "probe_id": "feature",
            "Probe ID": "feature",
            "probe ID": "feature",
            "ProbeID": "feature",
            "ID": "feature",
            "id": "feature",
        }
    )


def is_missing_label(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return (
        series.isna()
        | cleaned.eq("")
        | cleaned.str.lower().isin({"nan", "none", "null", "na", "<na>"})
        | cleaned.eq("-")
    )


def load_label_mapping(label_file: Path) -> pd.DataFrame:
    labels = normalize_feature_columns(pd.read_csv(label_file))
    if "feature" not in labels.columns or "gene_name" not in labels.columns:
        raise ValueError(
            f"{label_file} must contain probe/feature and gene_name/gene_symbol columns"
        )

    labels = labels[["feature", "gene_name"]].copy()
    labels["feature"] = labels["feature"].astype(str).str.strip()
    labels["gene_name"] = labels["gene_name"].astype(str).str.strip()
    labels = labels[labels["feature"].ne("")]
    labels = labels[~is_missing_label(labels["gene_name"])]
    return labels.drop_duplicates("feature", keep="first")


def load_selected_features(
    feature_file: Path,
    min_frequency: float | None,
    label_file: Path | None,
) -> pd.DataFrame:
    features = pd.read_csv(feature_file)
    features = normalize_feature_columns(features)

    if "feature" not in features.columns:
        raise ValueError(f"{feature_file} must contain a feature/probe column")
    if "gene_name" not in features.columns:
        features["gene_name"] = features["feature"]

    features["feature"] = features["feature"].astype(str).str.strip()
    features["gene_name"] = features["gene_name"].astype(str).str.strip()

    if label_file is not None:
        labels = load_label_mapping(label_file)
        features = features.merge(
            labels.rename(columns={"gene_name": "mapped_gene_name"}),
            on="feature",
            how="left",
        )
        has_mapping = ~is_missing_label(features["mapped_gene_name"])
        features.loc[has_mapping, "gene_name"] = features.loc[has_mapping, "mapped_gene_name"]
        features = features.drop(columns=["mapped_gene_name"])

    missing_gene_name = is_missing_label(features["gene_name"])
    features.loc[missing_gene_name, "gene_name"] = features.loc[missing_gene_name, "feature"]
    features = features[features["feature"].ne("")].copy()

    if min_frequency is not None and "frequency" in features.columns:
        features["frequency"] = pd.to_numeric(features["frequency"], errors="coerce")
        features = features[features["frequency"] >= min_frequency].copy()

    if features.empty:
        raise ValueError(f"No usable selected features found in {feature_file}")

    return features.drop_duplicates(["feature", "gene_name"])


def make_gene_level_matrix(
    dataset_df: pd.DataFrame,
    selected_features: pd.DataFrame,
    collapse_duplicate_genes: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = selected_features[selected_features["feature"].isin(dataset_df.columns)].copy()
    if usable.empty:
        missing = selected_features["feature"].head(10).tolist()
        raise ValueError(f"No selected features were found in the dataset columns. Examples: {missing}")

    usable["feature"] = usable["feature"].astype(str).str.strip()
    usable["gene_name"] = usable["gene_name"].astype(str).str.strip()
    missing_gene_name = is_missing_label(usable["gene_name"])
    usable.loc[missing_gene_name, "gene_name"] = usable.loc[missing_gene_name, "feature"]
    usable["gene_name"] = usable["gene_name"].fillna(usable["feature"]).astype(str).str.strip()

    X_probe = dataset_df[usable["feature"].tolist()].copy()
    if collapse_duplicate_genes:
        rename_map = dict(zip(usable["feature"], usable["gene_name"]))
        X_gene_named = X_probe.rename(columns=rename_map)
        X_gene = X_gene_named.T.groupby(level=0).mean().T
    else:
        gene_counts = usable["gene_name"].value_counts()
        display_names = []
        for _, row in usable.iterrows():
            gene = row["gene_name"]
            probe = row["feature"]
            if pd.isna(gene) or str(gene).strip().lower() in {"", "nan", "none", "null", "na", "<na>", "-"}:
                gene = probe
            if gene_counts.get(gene, 1) > 1:
                display_names.append(f"{gene} ({probe})")
            else:
                display_names.append(gene)
        X_gene = X_probe.copy()
        X_gene.columns = display_names

    mapping_used = usable[["feature", "gene_name"]].copy()
    mapping_used = mapping_used.rename(columns={"feature": "probe_id", "gene_name": "gene_symbol"})
    return X_gene, mapping_used


def xgb_shap_contributions(
    model: XGBClassifier,
    X: pd.DataFrame,
    class_names: list[str],
) -> np.ndarray:
    dmatrix = DMatrix(X, feature_names=list(X.columns))
    raw = np.asarray(model.get_booster().predict(dmatrix, pred_contribs=True))
    n_features = X.shape[1]
    n_classes = len(class_names)

    if raw.ndim == 2:
        if n_classes > 2 and raw.shape[1] == n_classes * (n_features + 1):
            raw = raw.reshape(X.shape[0], n_classes, n_features + 1)
            return np.transpose(raw[:, :, :-1], (0, 2, 1))
        return raw[:, :-1]

    if raw.ndim == 3:
        if raw.shape[1] == n_classes and raw.shape[2] == n_features + 1:
            return np.transpose(raw[:, :, :-1], (0, 2, 1))
        if raw.shape[1] == n_features + 1 and raw.shape[2] == n_classes:
            return raw[:, :-1, :]

    raise ValueError(f"Unexpected XGBoost contribution shape: {raw.shape}")


def mean_abs_shap_table(
    shap_values: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    values = np.asarray(shap_values)

    if values.ndim == 2:
        return pd.DataFrame(
            {
                "gene_symbol": feature_names,
                "class_name": "overall",
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

    if values.ndim == 3:
        rows = []
        for class_idx, class_name in enumerate(class_names):
            rows.append(
                pd.DataFrame(
                    {
                        "gene_symbol": feature_names,
                        "class_name": class_name,
                        "mean_abs_shap": np.abs(values[:, :, class_idx]).mean(axis=0),
                    }
                )
            )
        per_class = pd.concat(rows, ignore_index=True)
        overall = (
            per_class.groupby("gene_symbol", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
        )
        overall["class_name"] = "overall"
        return pd.concat([overall, per_class], ignore_index=True)

    raise ValueError(f"Unexpected SHAP value dimensions: {values.shape}")


def draw_custom_bar(
    mean_abs: pd.DataFrame,
    out_prefix: Path,
    formats: list[str],
    max_display: int,
) -> None:
    plot_df = mean_abs[mean_abs["class_name"] == "overall"].head(max_display).copy()
    plot_df = plot_df.sort_values("mean_abs_shap", ascending=True)

    fig_height = max(4.5, 0.35 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    ax.barh(plot_df["gene_symbol"], plot_df["mean_abs_shap"], color="#4C78A8")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_prefix, formats)
    plt.close(fig)


def draw_multiclass_classwise_bar(
    mean_abs: pd.DataFrame,
    class_names: list[str],
    out_prefix: Path,
    formats: list[str],
    max_display: int,
) -> None:
    overall = (
        mean_abs[mean_abs["class_name"] == "overall"]
        .sort_values("mean_abs_shap", ascending=False)
        .head(max_display)
    )
    genes = overall["gene_symbol"].tolist()
    per_class = mean_abs[mean_abs["class_name"].isin(class_names)].copy()

    matrix = (
        per_class.pivot_table(
            index="gene_symbol",
            columns="class_name",
            values="mean_abs_shap",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(index=genes, columns=class_names, fill_value=0.0)
    )
    matrix = matrix.iloc[::-1]

    fig_height = max(4.8, 0.42 * len(matrix) + 1.4)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    left = np.zeros(len(matrix))

    for idx, class_name in enumerate(class_names):
        values = matrix[class_name].to_numpy()
        ax.barh(
            matrix.index,
            values,
            left=left,
            color=CLASS_COLORS[idx % len(CLASS_COLORS)],
            label=class_name,
            edgecolor="white",
            linewidth=0.6,
        )
        left += values

    ax.set_xlabel("mean(|SHAP value|) by class")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.legend(
        title="Class",
        loc="lower right",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, out_prefix, formats)
    plt.close(fig)


def draw_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    class_names: list[str],
    out_dir: Path,
    formats: list[str],
    max_display: int,
) -> None:
    values = np.asarray(shap_values)

    if values.ndim == 2:
        draw_single_beeswarm(
            values,
            X,
            out_dir / "shap_summary_beeswarm",
            formats,
            max_display,
        )
        return

    if values.ndim == 3:
        for class_idx, class_name in enumerate(class_names):
            safe_class = class_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            draw_single_beeswarm(
                values[:, :, class_idx],
                X,
                out_dir / f"shap_summary_beeswarm_{safe_class}",
                formats,
                max_display,
                title=f"Class {class_name}",
            )
        return

    raise ValueError(f"Unexpected SHAP value dimensions: {values.shape}")


def draw_single_beeswarm(
    values: np.ndarray,
    X: pd.DataFrame,
    out_prefix: Path,
    formats: list[str],
    max_display: int,
    title: str | None = None,
) -> None:
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    order = order[::-1]
    labels = [X.columns[idx] for idx in order]

    fig_height = max(4.5, 0.36 * len(order) + 1.4)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    rng = np.random.default_rng(42)

    for y_pos, feature_idx in enumerate(order):
        shap_feature = values[:, feature_idx]
        expr_feature = X.iloc[:, feature_idx].to_numpy(dtype=float)
        finite = np.isfinite(expr_feature)
        if finite.any() and np.nanmax(expr_feature) > np.nanmin(expr_feature):
            colors = (expr_feature - np.nanmin(expr_feature)) / (
                np.nanmax(expr_feature) - np.nanmin(expr_feature)
            )
        else:
            colors = np.full(expr_feature.shape, 0.5)
        jitter = rng.normal(0, 0.08, size=shap_feature.shape[0])
        ax.scatter(
            shap_feature,
            np.full(shap_feature.shape[0], y_pos) + jitter,
            c=colors,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=22,
            edgecolors="none",
        )

    ax.axvline(0, color="#606060", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Feature value")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    fig.tight_layout()
    save_figure(fig, out_prefix, formats)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "outputs" / "model_eval"),
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "outputs" / "dataset_manifest.csv"),
    )
    parser.add_argument("--feature-file", default=None)
    parser.add_argument(
        "--label-file",
        default=None,
        help=(
            "Optional partial probe-to-gene CSV. It must contain a probe/feature "
            "column and a gene_name/gene_symbol column. Unmapped probes keep their probe IDs."
        ),
    )
    parser.add_argument("--min-frequency", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "shap"),
    )
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--formats", nargs="+", default=["tif", "eps", "png"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--collapse-duplicate-genes",
        action="store_true",
        help="Average probes mapping to the same gene before SHAP. By default, probes are modeled separately and only relabeled for display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    dataset_dir = results_dir / args.dataset_name
    feature_file = choose_feature_file(dataset_dir, args.feature_file)
    out_dir = Path(args.out_dir) / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = load_dataset_path(args.dataset_name, Path(args.manifest))
    dataset_df = pd.read_csv(dataset_path)
    if "CLASS" not in dataset_df.columns:
        raise ValueError(f"{dataset_path} does not contain CLASS")

    label_file = Path(args.label_file) if args.label_file else None
    selected_features = load_selected_features(feature_file, args.min_frequency, label_file)
    X_gene, mapping_used = make_gene_level_matrix(
        dataset_df,
        selected_features,
        collapse_duplicate_genes=args.collapse_duplicate_genes,
    )

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(
        imputer.fit_transform(X_gene),
        columns=X_gene.columns,
        index=X_gene.index,
    )

    label_encoder = LabelEncoder()
    y_raw = dataset_df["CLASS"].map(clean_class_label)
    y = label_encoder.fit_transform(y_raw)
    class_names = [clean_class_label(value) for value in label_encoder.classes_]

    model = make_xgb(len(class_names), args.seed, args.n_jobs)
    model.fit(X, y)

    shap_values = xgb_shap_contributions(model, X, class_names)

    mean_abs = mean_abs_shap_table(shap_values, list(X.columns), class_names)
    mean_abs.to_csv(out_dir / "mean_abs_shap_gene_symbols.csv", index=False)
    selected_features.to_csv(out_dir / "selected_features_used.csv", index=False)
    mapping_used.to_csv(out_dir / "probe_to_gene_mapping_used.csv", index=False)
    pd.Series(class_names, name="class_name").to_csv(
        out_dir / "classes.csv",
        index_label="encoded_label",
    )

    draw_custom_bar(
        mean_abs,
        out_dir / "shap_summary_bar",
        args.formats,
        args.max_display,
    )
    if np.asarray(shap_values).ndim == 3:
        draw_multiclass_classwise_bar(
            mean_abs,
            class_names,
            out_dir / "shap_summary_bar_classwise",
            args.formats,
            args.max_display,
        )
    draw_beeswarm(
        shap_values,
        X,
        class_names,
        out_dir,
        args.formats,
        args.max_display,
    )

    with open(out_dir / "shap_run_config.json", "w") as handle:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "dataset_path": path_for_metadata(dataset_path),
                "feature_file": path_for_metadata(feature_file),
                "label_file": path_for_metadata(label_file),
                "n_selected_rows": int(selected_features.shape[0]),
                "n_mapped_probes": int(mapping_used["probe_id"].nunique()),
                "n_gene_symbols_after_collapsing": int(X.shape[1]),
                "collapse_duplicate_genes": bool(args.collapse_duplicate_genes),
                "classes": class_names,
                "model": "XGBClassifier",
                "shap_backend": "xgboost pred_contribs",
                "purpose": "Post-hoc interpretation only; not used for performance estimation.",
            },
            handle,
            indent=2,
        )

    print(f"Done: {args.dataset_name}")
    print(f"Feature file: {feature_file}")
    print(f"Gene symbols used: {X.shape[1]}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
