"""Load a saved pipeline and estimate risk for a new patient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import LOW_RISK_MAX, MEDIUM_RISK_MAX, MODEL_PATH
from data_preprocessing import normalize_patient_record


def risk_category(probability: float) -> str:
    """Map a probability to the configured Low/Medium/High bands."""
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability <= MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def predict_patient_risk(
    record: dict[str, Any] | pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, str | float]:
    """Return estimated probability, risk band, and a clinician-facing message."""
    try:
        model = joblib.load(model_path)
        patient = normalize_patient_record(record)
        probability = float(model.predict_proba(patient)[0, 1])
        decision_threshold = float(getattr(model, "decision_threshold_", 0.5))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RuntimeError(f"Could not generate patient risk prediction: {exc}") from exc

    category = risk_category(probability)
    messages = {
        "Low": "Estimated risk is low; continue routine clinical assessment.",
        "Medium": (
            "Estimated risk is intermediate; review the full clinical context "
            "and consider further assessment."
        ),
        "High": (
            "Estimated risk is high; prompt clinician review and appropriate "
            "confirmatory assessment are advised."
        ),
    }
    return {
        "probability": probability,
        "risk_category": category,
        "high_risk_prediction": probability >= decision_threshold,
        "decision_threshold": decision_threshold,
        "recommended_message": messages[category],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict cardiovascular risk from a one-row patient CSV or JSON."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV containing one patient row.")
    source.add_argument("--json", type=str, help="Patient feature mapping as JSON.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    try:
        record: dict[str, Any] | pd.DataFrame
        if args.csv:
            rows = pd.read_csv(args.csv)
            record = rows
        else:
            parsed = json.loads(args.json)
            if not isinstance(parsed, dict):
                raise ValueError("--json must contain a JSON object.")
            record = parsed
        print(json.dumps(predict_patient_risk(record, args.model), indent=2))
    except (OSError, pd.errors.ParserError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
