"""Cross-validated model comparison and hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer, recall_score

from config import (
    CATEGORICAL_FEATURES,
    CV_REPEATS,
    CV_SPLITS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SELECTION_WEIGHTS,
)
from feature_engineering import ClinicalFeatureEngineer


@dataclass
class ModelSearchResult:
    """Best fitted search and its cross-validation metrics."""

    search: GridSearchCV
    roc_auc: float
    recall: float
    f2: float
    selection_score: float


def _build_pipeline(estimator: Any, calibrate: bool = False) -> Pipeline:
    numeric_features = [name for name in NUMERIC_FEATURES]
    categorical_features = [name for name in CATEGORICAL_FEATURES]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    base_pipeline = Pipeline(
        [
            ("feature_engineering", ClinicalFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    if calibrate:
        return Pipeline(
            [
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=base_pipeline, cv=3, ensemble=False
                    ),
                )
            ]
        )
    return base_pipeline


def _selection_score(cv_results: dict[str, Any], candidate_index: int) -> float:
    """Combine CV ROC-AUC, high-risk recall, and F2 for search refitting."""
    return sum(
        weight * float(cv_results[f"mean_test_{metric}"][candidate_index])
        for metric, weight in SELECTION_WEIGHTS.items()
    )


def _refit_candidate(cv_results: dict[str, Any]) -> int:
    """Choose hyperparameters using the documented weighted screening objective."""
    scores = [
        _selection_score(cv_results, index)
        for index in range(len(cv_results["params"]))
    ]
    return max(range(len(scores)), key=scores.__getitem__)


def _model_specs() -> dict[str, tuple[Any, dict[str, list[Any]], bool]]:
    return {
        "Logistic Regression": (
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
            False,
        ),
        "Support Vector Machine": (
            SVC(class_weight="balanced"),
            {
                "classifier__estimator__classifier__C": [0.5, 1.0, 5.0],
                "classifier__estimator__classifier__kernel": ["rbf"],
            },
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            ),
            {
                "classifier__n_estimators": [150],
                "classifier__max_depth": [None, 6],
                "classifier__min_samples_leaf": [1, 2],
            },
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2],
            },
            False,
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [5, 9, 15],
                "classifier__weights": ["uniform", "distance"],
            },
            False,
        ),
    }


def compare_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, ModelSearchResult]:
    """Tune each candidate using repeated stratified CV on training data only."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    scoring = {
        "roc_auc": "roc_auc",
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0),
    }
    results: dict[str, ModelSearchResult] = {}

    for name, (estimator, parameter_grid, calibrate) in _model_specs().items():
        print(f"\nTuning {name}...")
        search = GridSearchCV(
            estimator=_build_pipeline(estimator, calibrate=calibrate),
            param_grid=parameter_grid,
            scoring=scoring,
            refit=_refit_candidate,
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_index = int(search.best_index_)
        cv_results = search.cv_results_
        metric_values = {
            metric: float(cv_results[f"mean_test_{metric}"][best_index])
            for metric in scoring
        }
        results[name] = ModelSearchResult(
            search=search,
            roc_auc=metric_values["roc_auc"],
            recall=metric_values["recall"],
            f2=metric_values["f2"],
            selection_score=_selection_score(cv_results, best_index),
        )
        print(
            f"  CV ROC-AUC={metric_values['roc_auc']:.3f}, "
            f"high-risk recall={metric_values['recall']:.3f}, "
            f"F2={metric_values['f2']:.3f}"
        )
        print(f"  Best parameters: {search.best_params_}")
    return results


def results_table(results: dict[str, ModelSearchResult]) -> pd.DataFrame:
    """Return the comparison table sorted by CV ROC-AUC, then recall."""
    table = pd.DataFrame(
        [
            {
                "model": name,
                "cv_roc_auc": result.roc_auc,
                "cv_high_risk_recall": result.recall,
                "cv_f2": result.f2,
                "selection_score": result.selection_score,
            }
            for name, result in results.items()
        ]
    )
    return table.sort_values(
        ["cv_roc_auc", "cv_high_risk_recall"], ascending=False
    ).reset_index(drop=True)
