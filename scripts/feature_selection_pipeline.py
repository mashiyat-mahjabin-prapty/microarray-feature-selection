"""Full-dataset biomarker discovery pipeline.

Pipeline:
1. Rank features by cross-validated XGBoost importance.
2. Keep the top percentage of ranked features.
3. Run RFECV on the reduced feature set.
4. Assess common classifiers by repeated CV on the fixed selected panel.

This is intended for full-cohort biomarker discovery with post-selection
panel assessment, not as an unbiased estimate of de novo pipeline performance.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_SEED = 42
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def path_for_metadata(path: Path) -> str:
    """Return a portable path for saved run metadata."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"<external-data>/{resolved.name}"


def adaptive_splits(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(int(requested), int(counts.min())))


def load_dataset(path: Path) -> tuple[str, pd.DataFrame, np.ndarray, list[str], str]:
    df = pd.read_csv(path)
    if "CLASS" not in df.columns:
        raise ValueError(f"{path} does not contain a CLASS column")

    dataset_name = f"{path.parent.parent.name}_{path.parent.name}"
    X = df.drop(columns=["CLASS"])
    y_raw = df["CLASS"].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    classes = [str(value) for value in encoder.classes_]
    task = "binary" if len(classes) == 2 else "multiclass"
    return dataset_name, X, y, classes, task


def make_xgb(task: str, seed: int, n_jobs: int) -> XGBClassifier:
    eval_metric = "logloss" if task == "binary" else "mlogloss"
    return XGBClassifier(
        colsample_bytree=0.8,
        eval_metric=eval_metric,
        learning_rate=0.05,
        max_depth=3,
        n_estimators=200,
        n_jobs=n_jobs,
        random_state=seed,
        subsample=0.8,
        importance_type="gain",
    )


def xgb_cv_rank_features(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    task: str,
    cv_splits: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    feature_names = np.asarray(X_raw.columns)
    ranks = []
    importances = []
    fold_rows = []

    for fold_idx, (train_idx, _) in enumerate(cv.split(X_raw, y), start=1):
        print(f'Running XGBoost CV fold {fold_idx}/{cv_splits}...')
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(
            imputer.fit_transform(X_raw.iloc[train_idx]),
            columns=X_raw.columns,
            index=X_raw.index[train_idx],
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y[train_idx])
        model = make_xgb(task, seed + fold_idx, n_jobs)
        model.fit(X_train, y[train_idx], sample_weight=sample_weight)

        importance = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(importance)[::-1]
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)

        importances.append(importance)
        ranks.append(rank.astype(float))
        fold_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "feature": feature_names,
                    "rank": rank,
                    "importance": importance,
                }
            )
        )

    rank_matrix = np.vstack(ranks)
    importance_matrix = np.vstack(importances)

    ranked = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_rank": rank_matrix.mean(axis=0),
            "std_rank": rank_matrix.std(axis=0, ddof=1) if rank_matrix.shape[0] > 1 else 0.0,
            "mean_importance": importance_matrix.mean(axis=0),
            "std_importance": importance_matrix.std(axis=0, ddof=1)
            if importance_matrix.shape[0] > 1
            else 0.0,
        }
    )
    ranked = ranked.sort_values(
        ["mean_rank", "mean_importance", "feature"],
        ascending=[True, False, True],
    )
    fold_ranked = pd.concat(fold_rows, ignore_index=True)
    return ranked, fold_ranked


def rfecv_select_features(
    X_top_raw: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int,
    step: float | int,
    scoring: str,
    min_features_to_select: int,
    seed: int,
    n_jobs: int,
) -> tuple[list[str], RFECV, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", class_weight="balanced", random_state=seed)),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    selector = RFECV(
        estimator=pipeline,
        cv=cv,
        importance_getter="named_steps.svm.coef_",
        min_features_to_select=min(min_features_to_select, X_top_raw.shape[1]),
        n_jobs=n_jobs,
        scoring=scoring,
        step=step,
        verbose=1,
    )
    print(f"Running RFECV with {cv_splits} CV splits...")
    selector.fit(X_top_raw, y)

    selected = X_top_raw.columns[selector.support_].tolist()
    cv_results = pd.DataFrame(
        {
            key: value
            for key, value in selector.cv_results_.items()
            if np.asarray(value).ndim <= 1
        }
    )
    fold_selected, fold_consensus = rfecv_fold_feature_consensus(
        selector=selector,
        feature_names=list(X_top_raw.columns),
        cv_splits=cv_splits,
    )
    return selected, selector, cv_results, fold_selected, fold_consensus


def rfecv_fold_feature_consensus(
    selector: RFECV,
    feature_names: list[str],
    cv_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract RFECV-selected features from each CV fold when sklearn exposes them."""
    feature_array = np.asarray(feature_names)
    selected_n = int(selector.n_features_)
    n_features_grid = np.asarray(selector.cv_results_.get("n_features", []))
    rows = []

    for split_idx in range(cv_splits):
        support_key = f"split{split_idx}_support"
        ranking_key = f"split{split_idx}_ranking"
        if support_key not in selector.cv_results_:
            continue

        support_matrix = np.asarray(selector.cv_results_[support_key])
        if support_matrix.ndim != 2:
            continue

        if n_features_grid.size == support_matrix.shape[0]:
            matching = np.where(n_features_grid == selected_n)[0]
            if matching.size:
                row_idx = int(matching[-1])
            else:
                row_idx = int(np.argmin(np.abs(n_features_grid - selected_n)))
        else:
            support_counts = support_matrix.sum(axis=1)
            matching = np.where(support_counts == selected_n)[0]
            if matching.size:
                row_idx = int(matching[-1])
            else:
                row_idx = int(np.argmin(np.abs(support_counts - selected_n)))

        support = support_matrix[row_idx].astype(bool)
        selected_features = feature_array[support]

        ranking_values = np.full(feature_array.shape[0], np.nan)
        if ranking_key in selector.cv_results_:
            ranking_matrix = np.asarray(selector.cv_results_[ranking_key])
            if ranking_matrix.ndim == 2 and row_idx < ranking_matrix.shape[0]:
                ranking_values = ranking_matrix[row_idx]

        for feature in selected_features:
            feature_idx = int(np.where(feature_array == feature)[0][0])
            rows.append(
                {
                    "rfecv_fold": split_idx + 1,
                    "feature": feature,
                    "rank": ranking_values[feature_idx],
                    "n_features_at_selected_step": int(support.sum()),
                }
            )

    fold_selected = pd.DataFrame(rows)
    if fold_selected.empty:
        fold_consensus = pd.DataFrame(
            columns=["feature", "count", "n_folds", "frequency", "mean_rank"]
        )
        return fold_selected, fold_consensus

    fold_consensus = (
        fold_selected.groupby("feature", as_index=False)
        .agg(
            count=("rfecv_fold", "nunique"),
            mean_rank=("rank", "mean"),
        )
    )
    fold_consensus["n_folds"] = cv_splits
    fold_consensus["frequency"] = fold_consensus["count"] / cv_splits
    fold_consensus = fold_consensus.sort_values(
        ["frequency", "mean_rank", "feature"],
        ascending=[False, True, True],
    )
    return fold_selected, fold_consensus


def make_lr(task: str, seed: int) -> LogisticRegression:
    solver = "liblinear" if task == "binary" else "lbfgs"
    return LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        solver=solver,
    )


def classifier_factories(task: str, seed: int, n_jobs: int, stack_cv: int) -> dict[str, BaseEstimator]:
    lr = make_lr(task, seed)
    svm = SVC(class_weight="balanced", probability=True, random_state=seed)
    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=500,
        n_jobs=n_jobs,
        random_state=seed,
    )
    xgb = make_xgb(task, seed, n_jobs)

    def wrapped(model: BaseEstimator) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    voting = VotingClassifier(
        estimators=[
            ("lr", wrapped(lr)),
            ("svm", wrapped(svm)),
            ("rf", wrapped(rf)),
            ("xgb", wrapped(xgb)),
        ],
        voting="soft",
        n_jobs=n_jobs,
    )

    stacking = StackingClassifier(
        cv=stack_cv,
        estimators=[
            ("lr", wrapped(lr)),
            ("svm", wrapped(svm)),
            ("rf", wrapped(rf)),
            ("xgb", wrapped(xgb)),
        ],
        final_estimator=make_lr(task, seed),
        n_jobs=n_jobs,
        stack_method="predict_proba",
    )

    return {
        "lr": wrapped(lr),
        "svm": wrapped(svm),
        "rf": wrapped(rf),
        "xgb": wrapped(xgb),
        "voting": voting,
        "stacking": stacking,
    }


def predict_scores(model: BaseEstimator, X_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    classes: list[str],
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_score is None:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
        return metrics

    try:
        if len(classes) == 2:
            score_1d = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
            metrics["roc_auc"] = roc_auc_score(y_true, score_1d)
            metrics["average_precision"] = float(
                np.clip(average_precision_score(y_true, score_1d), 0.0, 1.0)
            )
        else:
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y_true)
            metrics["roc_auc"] = roc_auc_score(
                y_bin,
                y_score,
                average="macro",
                multi_class="ovr",
            )
            metrics["average_precision"] = float(
                np.clip(
                    average_precision_score(
                        y_bin,
                        y_score,
                        average="macro",
                    ),
                    0.0,
                    1.0,
                )
            )
    except ValueError:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


def evaluate_panel(
    X_panel: pd.DataFrame,
    y: np.ndarray,
    classes: list[str],
    task: str,
    cv_splits: int,
    repeats: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Evaluating panel with {cv_splits} CV splits and {repeats} repeats...")
    stack_cv = adaptive_splits(y, min(5, cv_splits))
    models = classifier_factories(task, seed, n_jobs, stack_cv)
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=repeats,
        random_state=seed,
    )

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_panel, y), start=1):
        X_train = X_panel.iloc[train_idx]
        X_test = X_panel.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, model in models.items():
            print(f"Running {model_name} on fold {fold_idx}/{cv_splits * repeats}...")
            started = time.perf_counter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = predict_scores(model, X_test)
            metrics = score_predictions(y_test, y_pred, y_score, classes)
            metrics.update(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "fit_eval_seconds": time.perf_counter() - started,
                }
            )
            rows.append(metrics)

    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby("model")
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return fold_metrics, summary


def run(args: argparse.Namespace) -> None:
    dataset_path = Path(args.csv)
    dataset_name, X_raw, y, classes, task = load_dataset(dataset_path)
    out_dir = Path(args.out_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_splits = adaptive_splits(y, args.xgb_cv_splits)
    rfecv_splits = adaptive_splits(y, args.rfecv_cv_splits)
    ml_splits = adaptive_splits(y, args.ml_cv_splits)

    print(f"Running XGBoost CV with {xgb_splits} splits...")
    ranked, fold_ranked = xgb_cv_rank_features(
        X_raw,
        y,
        task,
        cv_splits=xgb_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    if args.top_n is not None:
        n_top = min(max(1, int(args.top_n)), X_raw.shape[1])
        print(f"Selecting top {n_top} XGBoost-ranked features for RFECV...")
    else:
        n_top = max(1, int(math.ceil(X_raw.shape[1] * args.top_pct)))
        print(f"Selecting top {args.top_pct:.0%} features ({n_top}) for RFECV...")
    top_features = ranked.head(n_top)["feature"].tolist()
    X_top = X_raw[top_features]

    fold_ranked["xgb_top_candidate"] = fold_ranked["rank"] <= n_top
    xgb_consensus = (
        fold_ranked.groupby("feature", as_index=False)
        .agg(
            count=("xgb_top_candidate", "sum"),
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
    )
    xgb_consensus["n_folds"] = xgb_splits
    xgb_consensus["frequency"] = xgb_consensus["count"] / xgb_splits
    xgb_consensus = xgb_consensus.sort_values(
        ["frequency", "mean_rank", "mean_importance", "feature"],
        ascending=[False, True, False, True],
    )
    xgb_consensus_50 = xgb_consensus[xgb_consensus["frequency"] >= args.consensus_threshold].copy()

    print(f"Running RFECV with {rfecv_splits} CV splits...")
    selected, selector, rfecv_results, rfecv_fold_selected, rfecv_consensus = rfecv_select_features(
        X_top,
        y,
        cv_splits=rfecv_splits,
        step=args.rfecv_step,
        scoring=args.rfecv_scoring,
        min_features_to_select=args.min_features_to_select,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    print(f"Final selected features: {len(selected)}")
    rfecv_consensus_50 = rfecv_consensus[
        rfecv_consensus["frequency"] >= args.consensus_threshold
    ].copy()
    X_panel = X_raw[selected]
    fold_metrics, summary = evaluate_panel(
        X_panel,
        y,
        classes,
        task,
        cv_splits=ml_splits,
        repeats=args.ml_repeats,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    ranked.to_csv(out_dir / "xgb_cv_ranked_features.csv", index=False)
    fold_ranked.to_csv(out_dir / "xgb_fold_feature_ranks.csv", index=False)
    ranked.head(n_top).to_csv(out_dir / "xgb_top_features.csv", index=False)
    xgb_consensus.to_csv(out_dir / "xgb_top_feature_consensus_all.csv", index=False)
    xgb_consensus_50.to_csv(out_dir / "xgb_top_feature_consensus_threshold.csv", index=False)
    rfecv_results.to_csv(out_dir / "rfecv_cv_results.csv", index=False)
    rfecv_fold_selected.to_csv(out_dir / "rfecv_selected_features_by_fold.csv", index=False)
    rfecv_consensus.to_csv(out_dir / "rfecv_feature_consensus_all.csv", index=False)
    rfecv_consensus_50.to_csv(out_dir / "rfecv_feature_consensus_threshold.csv", index=False)
    fold_metrics.to_csv(out_dir / "ml_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "ml_summary_metrics.csv", index=False)

    selected_df = ranked[ranked["feature"].isin(selected)].copy()
    selected_df["rfecv_selected"] = True
    selected_df = selected_df.merge(
        xgb_consensus[["feature", "count", "frequency"]],
        on="feature",
        how="left",
    )
    selected_df = selected_df.rename(
        columns={"count": "xgb_top_count", "frequency": "xgb_top_frequency"}
    )
    selected_df = selected_df.merge(
        rfecv_consensus[["feature", "count", "frequency"]],
        on="feature",
        how="left",
    )
    selected_df = selected_df.rename(
        columns={"count": "rfecv_fold_count", "frequency": "rfecv_fold_frequency"}
    )
    selected_df = selected_df.sort_values(["mean_rank", "mean_importance", "feature"])
    selected_df.to_csv(out_dir / "final_selected_features.csv", index=False)

    pd.Series(classes, name="class_name").to_csv(
        out_dir / "classes.csv",
        index_label="encoded_label",
    )

    config = {
        "dataset": dataset_name,
        "csv": path_for_metadata(dataset_path),
        "task": task,
        "classes": classes,
        "n_samples": int(X_raw.shape[0]),
        "n_original_features": int(X_raw.shape[1]),
        "xgb_cv_splits_requested": args.xgb_cv_splits,
        "xgb_cv_splits_used": xgb_splits,
        "top_pct": args.top_pct,
        "top_n_requested": args.top_n,
        "n_xgb_top_features": len(top_features),
        "consensus_threshold": args.consensus_threshold,
        "n_xgb_consensus_features": int(xgb_consensus_50.shape[0]),
        "n_rfecv_consensus_features": int(rfecv_consensus_50.shape[0]),
        "rfecv_cv_splits_requested": args.rfecv_cv_splits,
        "rfecv_cv_splits_used": rfecv_splits,
        "rfecv_step": args.rfecv_step,
        "rfecv_scoring": args.rfecv_scoring,
        "n_final_selected_features": len(selected),
        "ml_cv_splits_requested": args.ml_cv_splits,
        "ml_cv_splits_used": ml_splits,
        "ml_repeats": args.ml_repeats,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "interpretation": (
            "Full-cohort biomarker discovery with post-selection repeated-CV "
            "assessment of the fixed feature panel."
        ),
    }
    with open(out_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    print(f"Done: {dataset_name}")
    print(f"XGB top features: {len(top_features)}")
    print(f"RFECV final selected features: {len(selected)}")
    print(f"Output: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Dataset CSV with a CLASS column")
    parser.add_argument("--out-dir", default="outputs/model_eval")
    parser.add_argument("--xgb-cv-splits", type=int, default=5)
    parser.add_argument("--top-pct", type=float, default=0.20)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Use a fixed number of XGBoost-ranked features before RFECV. Overrides --top-pct.",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.50,
        help="Frequency threshold for XGBoost top-feature consensus reporting.",
    )
    parser.add_argument("--rfecv-cv-splits", type=int, default=5)
    parser.add_argument("--rfecv-step", type=float, default=0.20)
    parser.add_argument("--rfecv-scoring", default="balanced_accuracy")
    parser.add_argument("--min-features-to-select", type=int, default=5)
    parser.add_argument("--ml-cv-splits", type=int, default=5)
    parser.add_argument("--ml-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
