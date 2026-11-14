"""Create ROC and precision-recall plots for the fixed held-out test split.

Run with: python plot_roc_pr_curves.py

This script reproduces the original stratified test split and only calls
predict_proba on that test partition. It does not refit the model, tune a
threshold, or select a model based on test results.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import DATA_PATH, MODEL_PATH, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from data_preprocessing import clean_dataset, load_dataset


def main() -> None:
    """Plot class-discrimination curves on the untouched held-out test rows."""
    raw_data = load_dataset(str(DATA_PATH))
    features, target = clean_dataset(raw_data)
    _, X_test, _, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    try:
        model = joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not load trained model at {MODEL_PATH}: {exc}") from exc

    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name="CardioRiskML",
        ax=axis,
    )
    axis.set_title("ROC Curve — Untouched Test Set")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    roc_path = OUTPUT_DIR / "roc_curve.png"
    figure.savefig(roc_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        name="CardioRiskML",
        ax=axis,
    )
    axis.set_title("Precision–Recall Curve — Untouched Test Set")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    pr_path = OUTPUT_DIR / "precision_recall_curve.png"
    figure.savefig(pr_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"Untouched held-out test rows: {len(y_test)}")
    print(f"Positive-class prevalence: {y_test.mean():.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Average precision (PR-AUC): {average_precision:.3f}")
    print(f"ROC curve saved to: {roc_path}")
    print(f"Precision-recall curve saved to: {pr_path}")


if __name__ == "__main__":
    main()
