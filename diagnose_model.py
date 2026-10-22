"""Leakage-safe diagnostics for the saved CardioRiskML classifier.

Run with: python diagnose_model.py

All diagnostic models are fitted on model-training rows only. The validation
partition is used for diagnostics and permutation importance; the held-out test
partition is used only for final reporting. No diagnostic writes or changes the
saved model, threshold, or test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from config import (
    DATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from data_preprocessing import clean_dataset, inspect_dataset, load_dataset
from train_models import (
    _build_pipeline,
    _model_specs,
    _refit_candidate,
    _selection_score,
    compare_models,
)


DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
TRAIN_SIZES = np.linspace(0.1, 1.0, 10)
DIAGNOSTIC_THRESHOLD = 0.5


def _load_data_path() -> Path:
    """Prefer requested data/heart.csv, falling back to this repo's root CSV."""
    requested_path = DATA_PATH.parent / "data" / "heart.csv"
    if requested_path.is_file():
        return requested_path
    if DATA_PATH.is_file():
        return DATA_PATH
    raise FileNotFoundError(
        f"Could not find dataset at {requested_path} or configured path {DATA_PATH}."
    )


def _metric_scorer(
    metric: Callable[..., float], threshold: float
) -> Callable[[Any, pd.DataFrame, pd.Series], float]:
    """Build a scorer using the saved binary threshold, without fitting anything."""

    def score(estimator: Any, features: pd.DataFrame, labels: pd.Series) -> float:
        probabilities = estimator.predict_proba(features)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        if metric is accuracy_score:
            return float(accuracy_score(labels, predictions))
        return float(metric(labels, predictions, zero_division=0))

    return score


def _evaluate_partition(
    model: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a fitted model without changing it or fitting on the partition."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _nested_cv_model_selection(
    X_fit: pd.DataFrame, y_fit: pd.Series
) -> dict[str, float]:
    """Estimate model-selection performance with inner tuning and outer scoring.

    Each outer fold independently tunes every candidate using only its outer
    training fold, then scores the chosen pipeline on the outer held-out fold.
    This avoids reusing folds that selected the saved model's hyperparameters.
    """
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "roc_auc": "roc_auc",
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0),
    }
    fold_rows: list[dict[str, Any]] = []
    model_specs = _model_specs()

    for fold_number, (inner_train_indices, outer_test_indices) in enumerate(
        outer_cv.split(X_fit, y_fit), start=1
    ):
        X_inner_train = X_fit.iloc[inner_train_indices]
        y_inner_train = y_fit.iloc[inner_train_indices]
        X_outer_test = X_fit.iloc[outer_test_indices]
        y_outer_test = y_fit.iloc[outer_test_indices]
        fold_searches: dict[str, GridSearchCV] = {}
        for name, (estimator, parameter_grid, calibrate) in model_specs.items():
            inner_cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold_number
            )
            search = GridSearchCV(
                estimator=_build_pipeline(estimator, calibrate=calibrate),
                param_grid=parameter_grid,
                scoring=scoring,
                refit=_refit_candidate,
                cv=inner_cv,
                n_jobs=-1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_inner_train, y_inner_train)
            fold_searches[name] = search

        selected_name = max(
            fold_searches,
            key=lambda candidate: _selection_score(
                fold_searches[candidate].cv_results_,
                int(fold_searches[candidate].best_index_),
            ),
        )
        selected_model = fold_searches[selected_name].best_estimator_
        outer_metrics = _evaluate_partition(
            selected_model, X_outer_test, y_outer_test, DIAGNOSTIC_THRESHOLD
        )
        fold_rows.append(
            {
                "outer_fold": fold_number,
                "selected_model": selected_name,
                **outer_metrics,
            }
        )
        print(
            f"  Nested outer fold {fold_number}: selected {selected_name}; "
            f"accuracy={outer_metrics['accuracy']:.3f}, "
            f"ROC-AUC={outer_metrics['roc_auc']:.3f}"
        )

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(DIAGNOSTIC_DIR / "nested_cv_folds.csv", index=False)
    means = {
        metric: float(fold_table[metric].mean())
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }
    return means


def _current_pipeline(
    X_fit: pd.DataFrame, y_fit: pd.Series
) -> tuple[Any, str, float]:
    """Load the existing fitted model, or select/refit using training rows only."""
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)
        threshold = float(getattr(model, "decision_threshold_", 0.5))
        model_name = type(
            getattr(model.named_steps.get("classifier"), "estimator", None)
            or model.named_steps.get("classifier")
        ).__name__
        print(f"\nLoaded existing fitted pipeline: {MODEL_PATH}")
        print(f"Stored binary decision threshold: {threshold:.3f}")
        return model, model_name, threshold

    print(f"\nNo saved model found at {MODEL_PATH}; selecting a model on training data.")
    searches = compare_models(X_fit, y_fit)
    selected_name = max(
        searches, key=lambda name: searches[name].selection_score
    )
    return (
        searches[selected_name].search.best_estimator_,
        selected_name,
        0.5,
    )


def _save_learning_curve(
    model: Any, X_fit: pd.DataFrame, y_fit: pd.Series, threshold: float
) -> pd.DataFrame:
    """Plot train/CV scores as the training fraction increases."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    curve_sizes, train_scores, validation_scores = learning_curve(
        clone(model),
        X_fit,
        y_fit,
        train_sizes=TRAIN_SIZES,
        cv=cv,
        scoring=_metric_scorer(accuracy_score, threshold),
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    rows = pd.DataFrame(
        {
            "training_rows": curve_sizes,
            "training_fraction_of_cv_fold": curve_sizes / curve_sizes[-1],
            "train_accuracy_mean": train_scores.mean(axis=1),
            "train_accuracy_std": train_scores.std(axis=1),
            "cv_accuracy_mean": validation_scores.mean(axis=1),
            "cv_accuracy_std": validation_scores.std(axis=1),
        }
    )
    rows.to_csv(DIAGNOSTIC_DIR / "learning_curve.csv", index=False)

    # A persistent train/CV gap suggests variance; similarly low scores suggest
    # bias. The curves are fit only on subsets of X_fit, never on validation/test.
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        curve_sizes,
        rows["train_accuracy_mean"],
        marker="o",
        label="Training accuracy",
    )
    axis.fill_between(
        curve_sizes,
        rows["train_accuracy_mean"] - rows["train_accuracy_std"],
        rows["train_accuracy_mean"] + rows["train_accuracy_std"],
        alpha=0.15,
    )
    axis.plot(
        curve_sizes,
        rows["cv_accuracy_mean"],
        marker="o",
        label="Cross-validation accuracy",
    )
    axis.fill_between(
        curve_sizes,
        rows["cv_accuracy_mean"] - rows["cv_accuracy_std"],
        rows["cv_accuracy_mean"] + rows["cv_accuracy_std"],
        alpha=0.15,
    )
    axis.set(
        xlabel="Training rows in each CV fold",
        ylabel=f"Accuracy at threshold {threshold:.2f}",
        title="Learning curve (training-only CV)",
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(DIAGNOSTIC_DIR / "learning_curve.png", dpi=160)
    plt.close(figure)
    return rows


def _important_parameters(model: Any) -> list[tuple[str, list[Any]]]:
    """Choose a small validation-curve grid based on the fitted classifier."""
    classifier = model.named_steps.get("classifier")
    if hasattr(classifier, "estimator"):
        classifier = classifier.estimator.named_steps.get("classifier")

    if isinstance(classifier, LogisticRegression):
        return [("classifier__C", [0.01, 0.1, 1.0, 10.0, 100.0])]
    if isinstance(classifier, DummyClassifier):
        return []
    if type(classifier).__name__ == "KNeighborsClassifier":
        return [
            ("classifier__n_neighbors", [3, 5, 7, 9, 11, 15, 17, 21]),
            ("classifier__weights", ["uniform", "distance"]),
        ]
    if type(classifier).__name__ == "RandomForestClassifier":
        return [
            ("classifier__max_depth", [2, 3, 4, 6, None]),
            ("classifier__min_samples_leaf", [1, 2, 4, 8]),
        ]
    if type(classifier).__name__ == "GradientBoostingClassifier":
        return [
            ("classifier__n_estimators", [50, 100, 150, 250]),
            ("classifier__learning_rate", [0.01, 0.05, 0.1, 0.2]),
        ]
    if type(classifier).__name__ == "SVC":
        return [("classifier__estimator__classifier__C", [0.1, 0.5, 1, 5, 10])]
    return []


def _save_validation_curves(
    model: Any,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Plot important hyperparameters with train/CV scores on training folds."""
    parameters = _important_parameters(model)
    if not parameters:
        print("No supported hyperparameter curve found for this saved estimator.")
        return pd.DataFrame()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict[str, Any]] = []
    figure, axes = plt.subplots(
        1, len(parameters), figsize=(7 * len(parameters), 5), squeeze=False
    )
    for axis, (parameter, values) in zip(axes[0], parameters):
        # Avoid KNN values larger than a training fold; validation_curve then
        # remains a fair comparison rather than failing on small folds.
        if parameter.endswith("n_neighbors"):
            maximum = int(len(X_fit) * 4 / 5)
            values = [value for value in values if value <= maximum]
        train_scores, validation_scores = validation_curve(
            clone(model),
            X_fit,
            y_fit,
            param_name=parameter,
            param_range=values,
            cv=cv,
            scoring=_metric_scorer(accuracy_score, threshold),
            n_jobs=-1,
            error_score="raise",
        )
        train_mean = train_scores.mean(axis=1)
        validation_mean = validation_scores.mean(axis=1)
        for value, train_score, cv_score in zip(values, train_mean, validation_mean):
            records.append(
                {
                    "parameter": parameter,
                    "value": str(value),
                    "train_accuracy_mean": float(train_score),
                    "cv_accuracy_mean": float(cv_score),
                    "cv_accuracy_std": float(validation_scores.std(axis=1)[values.index(value)]),
                }
            )
        x_axis = np.arange(len(values))
        axis.plot(x_axis, train_mean, marker="o", label="Training accuracy")
        axis.plot(x_axis, validation_mean, marker="o", label="CV accuracy")
        axis.set_xticks(x_axis, [str(value) for value in values])
        axis.set(
            xlabel=parameter,
            ylabel=f"Accuracy at threshold {threshold:.2f}",
            title=f"Validation curve: {parameter}",
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(DIAGNOSTIC_DIR / "validation_curves.png", dpi=160)
    plt.close(figure)
    result = pd.DataFrame(records)
    result.to_csv(DIAGNOSTIC_DIR / "validation_curves.csv", index=False)
    return result


def _permutation_importance(
    model: Any, X_validation: pd.DataFrame, y_validation: pd.Series
) -> pd.DataFrame:
    """Rank raw patient features using validation ROC-AUC permutation drops."""
    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_validation.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    importance.to_csv(DIAGNOSTIC_DIR / "permutation_importance.csv", index=False)
    return importance


def _compare_simple_and_complex(
    best_model: Any,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Compare a regularized logistic baseline with the selected model in CV."""
    logistic, _, _ = _model_specs()["Logistic Regression"]
    candidates = {
        "Regularized Logistic Regression (simple)": _build_pipeline(logistic),
        f"Saved best model ({type(best_model.named_steps.get('classifier')).__name__})": best_model,
        "Majority-class baseline": _build_pipeline(DummyClassifier(strategy="prior")),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorer = _metric_scorer(accuracy_score, threshold)
    rows: list[dict[str, float | str]] = []
    for name, estimator in candidates.items():
        scores = cross_validate(
            clone(estimator),
            X_fit,
            y_fit,
            cv=cv,
            scoring={"accuracy": scorer},
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_accuracy_std": float(scores["test_accuracy"].std()),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "cv_accuracy_mean", ascending=False, ignore_index=True
    )
    table.to_csv(DIAGNOSTIC_DIR / "simple_vs_complex.csv", index=False)
    return table


def _print_diagnosis(
    rows: int,
    class_share: float,
    metric_table: pd.DataFrame,
    learning_curve_table: pd.DataFrame,
    importance: pd.DataFrame,
    complexity: pd.DataFrame,
) -> None:
    """Print transparent heuristic flags and evidence-based recommendations."""
    train_accuracy = float(metric_table.loc["train", "accuracy"])
    validation_accuracy = float(metric_table.loc["validation", "accuracy"])
    cv_accuracy = float(metric_table.loc["cross_validation", "accuracy"])
    train_gap = train_accuracy - cv_accuracy
    validation_gap = train_accuracy - validation_accuracy
    final_cv = float(learning_curve_table["cv_accuracy_mean"].iloc[-1])
    top_five_gain = float(importance.head(5)["importance_mean"].clip(lower=0).sum())
    simple_accuracy = float(
        complexity.loc[
            complexity["model"].str.startswith("Regularized Logistic"),
            "cv_accuracy_mean",
        ].iloc[0]
    )
    complex_accuracy = float(
        complexity.loc[
            complexity["model"].str.startswith("Saved best"),
            "cv_accuracy_mean",
        ].iloc[0]
    )

    print("\n=== Diagnosis (data-dependent heuristics, not clinical conclusions) ===")
    flags: list[str] = []
    if train_gap >= 0.10 or validation_gap >= 0.10:
        flags.append("High variance / overfitting")
        print(
            f"- High variance / overfitting: flagged; train-CV gap={train_gap:.3f}, "
            f"train-validation gap={validation_gap:.3f} (0.10 heuristic)."
        )
    else:
        print(
            f"- High variance / overfitting: not strongly indicated by this heuristic; "
            f"train-CV gap={train_gap:.3f}, train-validation gap={validation_gap:.3f}."
        )

    if train_accuracy < 0.75 and train_gap < 0.10:
        flags.append("High bias / underfitting")
        print(
            f"- High bias / underfitting: flagged; train accuracy={train_accuracy:.3f} "
            "is low and the train-CV gap is small."
        )
    else:
        print(
            f"- High bias / underfitting: not strongly indicated; training accuracy="
            f"{train_accuracy:.3f}, CV accuracy={cv_accuracy:.3f}."
        )

    if class_share < 0.30:
        flags.append("Class imbalance issue")
        print(
            f"- Class imbalance issue: flagged; minority class is {class_share:.1%} "
            "(below the 30% heuristic)."
        )
    else:
        print(
            f"- Class imbalance issue: not strongly indicated; minority class is "
            f"{class_share:.1%}."
        )

    if top_five_gain <= 0.01:
        flags.append("Poor feature quality")
        print(
            f"- Poor feature quality: possible; top-five validation permutation "
            f"importance sum={top_five_gain:.3f}. Negative/near-zero importance "
            "does not prove the feature itself is clinically poor."
        )
    else:
        print(
            f"- Poor feature quality: not clearly indicated by permutation importance; "
            f"top-five positive importance sum={top_five_gain:.3f}."
        )

    if rows < 1000 or final_cv < 0.80:
        flags.append("Insufficient data")
        print(
            f"- Insufficient data: flagged as a reliability concern; {rows} unique "
            f"rows remain and final learning-curve CV accuracy is {final_cv:.3f}. "
            "The 1,000-row cutoff is a rough engineering heuristic, not a power analysis."
        )
    else:
        print(
            f"- Insufficient data: not flagged by this rough row-count/score heuristic; "
            f"{rows} unique rows remain."
        )

    print("\n=== Next three specific improvements ===")
    recommendations: list[str] = []
    if rows < 1000:
        recommendations.append(
            "Collect or obtain an independent, larger, representative cohort; "
            "deduplicate by patient and split by patient/site before evaluation."
        )
    if train_gap >= 0.10:
        recommendations.append(
            "Reduce variance by tuning stronger regularization or simpler model "
            "capacity using training-only nested/repeated CV."
        )
    if class_share < 0.30:
        recommendations.append(
            "Evaluate class-weighted and training-fold-only resampling approaches "
            "against recall, precision, PR-AUC, and calibration."
        )
    if simple_accuracy > complex_accuracy + 0.02:
        recommendations.append(
            "Prefer the simpler regularized logistic model: it exceeds the current "
            "best model's CV accuracy by more than 0.02."
        )
    elif abs(simple_accuracy - complex_accuracy) <= 0.02:
        recommendations.append(
            "Benchmark regularized Logistic Regression against the saved model in "
            "nested CV on recall, F1/F2, and calibration; their CV accuracy is "
            f"nearly tied ({simple_accuracy:.3f} vs {complex_accuracy:.3f}), so "
            "the simpler model may generalize better."
        )
    else:
        recommendations.append(
            "Inspect the saved model's validation curve and tune one capacity "
            "parameter at a time within nested CV before retaining the more "
            "complex candidate."
        )
    if top_five_gain <= 0.01:
        recommendations.append(
            "Audit feature definitions, coding, and collection quality with a "
            "validated codebook; assess external data rather than dropping features "
            "from this small validation sample."
        )
    if not recommendations:
        recommendations.append(
            "Prioritize external validation and confidence intervals; this small "
            "dataset cannot establish that a different algorithm will exceed 90%."
        )
    while len(recommendations) < 3:
        recommendations.append(
            "Prespecify recall, precision, calibration, and subgroup acceptance "
            "criteria, then assess them on an independent cohort."
        )
    for recommendation in recommendations[:3]:
        print(f"- {recommendation}")

    if flags:
        print("\nFlagged diagnosis categories: " + "; ".join(flags))
    else:
        print("\nNo requested diagnosis category was strongly flagged by heuristics.")


def main() -> None:
    """Run all diagnostics without fitting any preprocessing on full data."""
    dataset_path = _load_data_path()
    raw_data = load_dataset(str(dataset_path))
    print(f"Dataset path: {dataset_path}")
    inspect_dataset(raw_data)

    features, target = clean_dataset(raw_data)
    print("\nClass distribution after exact deduplication and cleaning:")
    print(target.value_counts().sort_index().to_string())
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    print(
        f"\nReproduced stratified partitions (seed={RANDOM_STATE}): "
        f"model-training={len(X_fit)}, validation={len(X_validation)}, "
        f"untouched-test={len(X_test)}."
    )
    print(
        "Only model-training rows fit estimators and preprocessing. Validation "
        "rows are used for diagnostic comparison/permutation importance; test "
        "rows are only scored, never used to select or fit anything."
    )

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    model, model_name, saved_threshold = _current_pipeline(X_fit, y_fit)
    print(
        f"Diagnostic comparisons use the fixed 0.50 classifier cutoff. The saved "
        f"operating cutoff ({saved_threshold:.3f}) was selected using this "
        "validation partition, so validation metrics at that cutoff would be "
        "optimistically biased. The saved cutoff is therefore applied only to "
        "the untouched test score below."
    )

    # A fixed 0.50 cutoff keeps train/CV/validation/test comparisons consistent.
    # The validation-selected deployment cutoff is not used to score validation.
    partition_metrics = {
        "train": _evaluate_partition(
            model, X_fit, y_fit, DIAGNOSTIC_THRESHOLD
        ),
        "validation": _evaluate_partition(
            model, X_validation, y_validation, DIAGNOSTIC_THRESHOLD
        ),
        "test": _evaluate_partition(model, X_test, y_test, DIAGNOSTIC_THRESHOLD),
    }

    # Nested CV repeats candidate model and hyperparameter selection inside
    # each outer training fold, avoiding reusing model-selection folds to score.
    print("\n=== Nested cross-validation (training partition only) ===")
    nested_cv_metrics = _nested_cv_model_selection(X_fit, y_fit)
    partition_metrics["cross_validation"] = nested_cv_metrics
    operational_test_metrics = _evaluate_partition(
        model, X_test, y_test, saved_threshold
    )
    partition_metrics["test_at_saved_operating_threshold"] = operational_test_metrics
    metric_table = pd.DataFrame(partition_metrics).T.loc[
        ["train", "cross_validation", "validation", "test"]
    ]
    metric_table.to_csv(DIAGNOSTIC_DIR / "partition_metrics.csv", index_label="partition")
    print(
        f"\n=== {model_name} performance "
        f"(fixed diagnostic threshold={DIAGNOSTIC_THRESHOLD:.2f}) ==="
    )
    print(
        metric_table.to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )
    print(
        "Training accuracy is in-sample. CV is nested: candidate models and "
        "hyperparameters are selected inside each outer training fold, with "
        "the outer fold used only for scoring. Validation and test rows were "
        "not fitted."
    )
    print(
        f"\nSaved-threshold test performance (threshold={saved_threshold:.3f}; "
        "threshold selected on validation, test untouched):"
    )
    print(
        pd.Series(operational_test_metrics).to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )

    curve = _save_learning_curve(model, X_fit, y_fit, DIAGNOSTIC_THRESHOLD)
    print("\nLearning-curve results (saved to diagnostics/learning_curve.csv):")
    print(
        curve[["training_rows", "train_accuracy_mean", "cv_accuracy_mean"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    validation_curves = _save_validation_curves(
        model, X_fit, y_fit, DIAGNOSTIC_THRESHOLD
    )
    if not validation_curves.empty:
        print("\nValidation curves (saved to diagnostics/validation_curves.csv):")
        print(validation_curves.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    importance = _permutation_importance(model, X_validation, y_validation)
    print("\nTop 10 validation permutation features (ROC-AUC decrease):")
    print(importance.head(10).to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    complexity = _compare_simple_and_complex(
        model, X_fit, y_fit, DIAGNOSTIC_THRESHOLD
    )
    print("\n=== Simple vs complex models (training-only CV accuracy) ===")
    print(complexity.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    minority_share = float(target.value_counts(normalize=True).min())
    _print_diagnosis(
        rows=len(features),
        class_share=minority_share,
        metric_table=metric_table,
        learning_curve_table=curve,
        importance=importance,
        complexity=complexity,
    )
    print(f"\nAll diagnostic tables/plots saved under: {DIAGNOSTIC_DIR}")


if __name__ == "__main__":
    main()
