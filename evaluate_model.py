"""Held-out evaluation, plots, and model interpretation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from config import RANDOM_STATE


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    candidate_models: dict[str, Any] | None = None,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate one selected model once on the untouched test partition."""
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "f2": fbeta_score(y_test, predictions, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "decision_threshold": decision_threshold,
    }

    print(
        "\n=== Untouched held-out test metrics "
        f"(validation-tuned threshold = {decision_threshold:.2f}) ==="
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("\nConfusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    CalibrationDisplay.from_predictions(
        y_test, probabilities, n_bins=min(10, max(3, len(y_test) // 10)), ax=axes[2]
    )
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-recall curve")
    axes[2].set_title("Calibration curve")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_curves.png", dpi=160)
    plt.close(fig)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    print("\nPermutation importance (test ROC-AUC decrease):")
    print(importance_table.head(10).to_string(index=False))
    importance_table.to_csv(output_dir / "permutation_importance.csv", index=False)
    _plot_tree_importances(candidate_models or {"Selected model": model}, output_dir)
    return metrics


def tune_decision_threshold(
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    output_dir: Path,
) -> tuple[float, pd.DataFrame]:
    """Choose a validation cutoff balancing accuracy and high-risk F1 equally."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(X_validation)[:, 1]
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_validation, predictions)),
                "precision": float(
                    precision_score(y_validation, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_validation, predictions, zero_division=0)),
                "f1": float(f1_score(y_validation, predictions, zero_division=0)),
                "selection_score": float(
                    (
                        accuracy_score(y_validation, predictions)
                        + f1_score(y_validation, predictions, zero_division=0)
                    )
                    / 2
                ),
            }
        )
    tradeoff = pd.DataFrame(rows)
    best = tradeoff.sort_values(
        ["selection_score", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected_threshold = float(best["threshold"])
    tradeoff.to_csv(output_dir / "threshold_tradeoff.csv", index=False)

    # Show regular threshold intervals plus the selected operating point.
    shown = tradeoff[
        np.isclose((tradeoff["threshold"] * 100) % 10, 0, atol=1e-8)
        | np.isclose(tradeoff["threshold"], selected_threshold)
    ].drop_duplicates(subset="threshold")
    print("\n=== Validation threshold trade-off ===")
    print(
        shown.sort_values("threshold").to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print(
        "\nSelected validation cutoff (highest mean of accuracy and F1, "
        "then recall and precision): "
        f"{selected_threshold:.2f} "
        f"(accuracy={best['accuracy']:.3f}, precision={best['precision']:.3f}, "
        f"recall={best['recall']:.3f}, F1={best['f1']:.3f}, "
        f"joint score={best['selection_score']:.3f})"
    )
    return selected_threshold, tradeoff


def _plot_tree_importances(
    models: dict[str, Any], output_dir: Path
) -> None:
    """Print and save intrinsic importances for each fitted tree candidate."""
    importance_tables: list[pd.DataFrame] = []
    for model_name, model in models.items():
        classifier = model.named_steps["classifier"]
        if not hasattr(classifier, "feature_importances_"):
            continue
        names = model.named_steps["preprocessor"].get_feature_names_out()
        importance_tables.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "feature": names,
                    "importance": classifier.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
        )
    if not importance_tables:
        print("\nNo selected candidate has intrinsic tree feature importances.")
        return

    table = pd.concat(importance_tables, ignore_index=True)
    print("\nTree-based feature importance:")
    for model_name, model_table in table.groupby("model", sort=False):
        print(f"\n{model_name}:")
        print(
            model_table.head(15)
            .drop(columns="model")
            .to_string(index=False)
        )
    table.to_csv(output_dir / "tree_feature_importance.csv", index=False)
    fig, axes = plt.subplots(
        1, len(importance_tables), figsize=(8 * len(importance_tables), 6), squeeze=False
    )
    for axis, model_table in zip(axes[0], importance_tables):
        top = model_table.head(15).sort_values("importance")
        axis.barh(top["feature"], top["importance"])
        axis.set_title(model_table["model"].iloc[0])
        axis.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "tree_feature_importance.png", dpi=160)
    plt.close(fig)

