"""SHAP explanations for individual High-category patient predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import FEATURE_COLUMNS, MEDIUM_RISK_MAX, MODEL_PATH, OUTPUT_DIR, RANDOM_STATE
from data_preprocessing import normalize_patient_record


def _explain_patient(
    record: dict[str, Any] | pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    background_data: pd.DataFrame | None = None,
    output_dir: str | Path | None = OUTPUT_DIR,
    n_permutations: int = 10,
    require_high: bool = False,
    artifact_prefix: str = "patient",
) -> dict[str, Any]:
    """Explain a patient's positive-class probability using probability-scale SHAP.

    The fitted model must contain its training-only ``shap_background_`` sample,
    or the caller must pass ``background_data`` drawn from training data.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    try:
        model = joblib.load(model_path)
    except (OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Could not load fitted model at {model_path}: {exc}") from exc

    patient = normalize_patient_record(record)
    probability = float(model.predict_proba(patient)[0, 1])
    category = (
        "Low"
        if probability < 0.20
        else "Medium"
        if probability <= MEDIUM_RISK_MAX
        else "High"
    )
    if require_high and category != "High":
        raise ValueError(
            "SHAP High-risk explanation requires a High category prediction "
            f"(probability > {MEDIUM_RISK_MAX:.2f}); got {probability:.3f}."
        )

    reference = background_data
    if reference is None:
        reference = getattr(model, "shap_background_", None)
    if not isinstance(reference, pd.DataFrame) or reference.empty:
        raise ValueError(
            "No training background data is available. Retrain the project to "
            "save its training-only SHAP background, or pass background_data."
        )
    missing = sorted(set(FEATURE_COLUMNS) - set(reference.columns))
    if missing:
        raise ValueError(f"SHAP background is missing feature columns: {missing}")
    reference = reference.loc[:, FEATURE_COLUMNS].copy()
    patient = patient.loc[:, FEATURE_COLUMNS]

    background_values = reference.to_numpy(dtype=float)
    patient_values = patient.to_numpy(dtype=float)

    def predict_positive_probability(values: np.ndarray) -> np.ndarray:
        rows = pd.DataFrame(values, columns=FEATURE_COLUMNS)
        return model.predict_proba(rows)[:, 1]

    explainer = shap.Explainer(
        predict_positive_probability,
        masker=shap.maskers.Independent(background_values),
        algorithm="permutation",
        feature_names=FEATURE_COLUMNS,
        seed=RANDOM_STATE,
    )
    shap_result = explainer(
        patient_values,
        max_evals=(2 * len(FEATURE_COLUMNS) + 1) * n_permutations,
        silent=True,
    )
    values = np.asarray(shap_result.values, dtype=float).reshape(-1)
    base_probability = float(np.asarray(shap_result.base_values).reshape(-1)[0])

    contributions = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "patient_value": patient.iloc[0].to_numpy(),
            "shap_value": values,
        }
    ).sort_values("shap_value", ascending=False, ignore_index=True)
    positive_reasons = contributions.loc[
        contributions["shap_value"] > 0
    ].head(5)
    risk_reducing_factors = contributions.loc[
        contributions["shap_value"] < 0
    ].sort_values("shap_value").head(5)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        contributions.to_csv(
            destination / f"{artifact_prefix}_shap_values.csv", index=False
        )
        shap.plots.waterfall(shap_result[0], max_display=10, show=False)
        figure = plt.gcf()
        figure.savefig(
            destination / f"{artifact_prefix}_shap_waterfall.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    return {
        "probability": probability,
        "risk_category": category,
        "binary_high_risk_threshold": float(
            getattr(model, "decision_threshold_", 0.5)
        ),
        "binary_high_risk_prediction": probability
        >= float(getattr(model, "decision_threshold_", 0.5)),
        "shap_base_probability": base_probability,
        "shap_reconstruction_probability": base_probability + float(values.sum()),
        "positive_reasons": positive_reasons.to_dict(orient="records"),
        "risk_reducing_factors": risk_reducing_factors.to_dict(orient="records"),
        "all_feature_contributions": contributions.to_dict(orient="records"),
    }


def explain_patient_prediction(
    record: dict[str, Any] | pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    background_data: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    n_permutations: int = 10,
) -> dict[str, Any]:
    """Explain a patient prediction in any configured risk category."""
    return _explain_patient(
        record,
        model_path,
        background_data,
        output_dir,
        n_permutations,
        require_high=False,
        artifact_prefix="patient",
    )


def explain_high_risk_patient(
    record: dict[str, Any] | pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    background_data: pd.DataFrame | None = None,
    output_dir: str | Path | None = OUTPUT_DIR,
    n_permutations: int = 10,
) -> dict[str, Any]:
    """Explain a patient's High (>0.50) probability using probability-scale SHAP."""
    return _explain_patient(
        record,
        model_path,
        background_data,
        output_dir,
        n_permutations,
        require_high=True,
        artifact_prefix="high_risk",
    )
