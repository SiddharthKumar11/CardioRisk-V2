"""Streamlit interface for the CardioRiskML screening prototype."""

from __future__ import annotations

from typing import Any

import joblib
import streamlit as st

from config import (
    CLINICAL_BOUNDS,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    MODEL_PATH,
)
from data_preprocessing import normalize_patient_record
from predict_risk import predict_patient_risk
from shap_explainability import explain_patient_prediction


st.set_page_config(
    page_title="CardioRiskML | Patient risk",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def _load_model() -> Any:
    return joblib.load(MODEL_PATH)


def _number_default(feature: str, fallback: float, model: Any) -> float:
    """Use a training-background median for a sensible valid form default."""
    background = getattr(model, "shap_background_", None)
    if background is not None and feature in background:
        value = float(background[feature].median())
        if CLINICAL_BOUNDS[feature][0] <= value <= CLINICAL_BOUNDS[feature][1]:
            return value
    return fallback


def _coded_default(feature: str, model: Any) -> int:
    background = getattr(model, "shap_background_", None)
    if background is not None and feature in background:
        return int(background[feature].mode(dropna=True).iloc[0])
    return int(CLINICAL_BOUNDS[feature][0])


def _select_coded_value(label: str, feature: str, model: Any) -> int:
    minimum, maximum = CLINICAL_BOUNDS[feature]
    choices = list(range(int(minimum), int(maximum) + 1))
    selected = st.selectbox(
        label,
        choices,
        index=choices.index(_coded_default(feature, model)),
        format_func=lambda value: f"Code {value}",
        help=(
            "Select the dataset's numeric code. The source file does not provide "
            "a verified human-readable codebook for this feature."
        ),
    )
    return int(selected)


def _render_results(record: dict[str, Any]) -> None:
    with st.spinner("Estimating risk and calculating SHAP explanation..."):
        prediction = predict_patient_risk(record)
        explanation = explain_patient_prediction(
            record,
            output_dir=None,
            n_permutations=5,
        )

    probability = float(prediction["probability"])
    category = str(prediction["risk_category"])
    metric_columns = st.columns(3)
    metric_columns[0].metric("Predicted probability", f"{probability:.1%}")
    metric_columns[1].metric("Risk category", category)
    metric_columns[2].metric(
        "Binary high-risk flag",
        "Yes" if prediction["high_risk_prediction"] else "No",
        help=(
            "Uses the separate validation-tuned decision threshold "
            f"({float(prediction['decision_threshold']):.2f}); the risk category "
            "uses the configured probability bands."
        ),
    )
    st.progress(probability, text="Estimated model probability")

    if category == "High":
        st.error(str(prediction["recommended_message"]))
    elif category == "Medium":
        st.warning(str(prediction["recommended_message"]))
    else:
        st.success(str(prediction["recommended_message"]))

    st.subheader("What influenced this estimate?")
    st.caption(
        "SHAP contributions show how each input moved this model's estimate "
        "relative to its training-data background. They are not causal or "
        "clinical explanations."
    )
    reasons = explanation["positive_reasons"]
    reducing = explanation["risk_reducing_factors"]
    reason_columns = st.columns(2)
    with reason_columns[0]:
        st.markdown("**Factors pushing probability higher**")
        if reasons:
            st.dataframe(reasons, hide_index=True, width="stretch")
        else:
            st.write("No positive SHAP contributions for this patient.")
    with reason_columns[1]:
        st.markdown("**Factors pushing probability lower**")
        if reducing:
            st.dataframe(reducing, hide_index=True, width="stretch")
        else:
            st.write("No negative SHAP contributions for this patient.")

    with st.expander("View all feature contributions"):
        st.dataframe(
            explanation["all_feature_contributions"],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "SHAP base probability: "
            f"{float(explanation['shap_base_probability']):.1%}; "
            "base plus all feature contributions reconstructs the model estimate."
        )


st.title("CardioRiskML")
st.subheader("Heart-disease risk screening")
st.write(
    "Enter the available patient measurements to view the model estimate and "
    "patient-specific SHAP contributions."
)
st.warning(
    "Decision-support prototype only—not a diagnosis or medical device. "
    "Clinical decisions require clinician assessment and independent validation."
)
st.caption(
    f"Risk bands: Low < {LOW_RISK_MAX:.0%}; "
    f"Medium {LOW_RISK_MAX:.0%}–{MEDIUM_RISK_MAX:.0%}; "
    f"High > {MEDIUM_RISK_MAX:.0%}."
)

if not MODEL_PATH.is_file():
    st.error(f"Model not found at {MODEL_PATH}. Train it first by running `python main.py`.")
    st.stop()
try:
    fitted_model = _load_model()
except (OSError, ValueError, EOFError, AttributeError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

with st.form("patient_form"):
    st.subheader("Patient details")
    clinical_columns = st.columns(3)
    with clinical_columns[0]:
        age_min, age_max = CLINICAL_BOUNDS["age"]
        age = st.number_input(
            "Age (years)",
            min_value=int(age_min),
            max_value=int(age_max),
            value=int(_number_default("age", 55, fitted_model)),
            step=1,
        )
        trestbps_min, trestbps_max = CLINICAL_BOUNDS["trestbps"]
        trestbps = st.number_input(
            "Resting blood pressure (mmHg)",
            min_value=float(trestbps_min),
            max_value=float(trestbps_max),
            value=_number_default("trestbps", 130, fitted_model),
            step=1.0,
        )
        chol_min, chol_max = CLINICAL_BOUNDS["chol"]
        chol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=float(chol_min),
            max_value=float(chol_max),
            value=_number_default("chol", 240, fitted_model),
            step=1.0,
        )
        thalach_min, thalach_max = CLINICAL_BOUNDS["thalach"]
        thalach = st.number_input(
            "Maximum heart rate",
            min_value=float(thalach_min),
            max_value=float(thalach_max),
            value=_number_default("thalach", 150, fitted_model),
            step=1.0,
        )
        oldpeak_min, oldpeak_max = CLINICAL_BOUNDS["oldpeak"]
        oldpeak = st.number_input(
            "Oldpeak (ST depression)",
            min_value=float(oldpeak_min),
            max_value=float(oldpeak_max),
            value=_number_default("oldpeak", 1.0, fitted_model),
            step=0.1,
        )
    with clinical_columns[1]:
        sex = _select_coded_value("Sex (coded)", "sex", fitted_model)
        cp = _select_coded_value("Chest pain type (coded)", "cp", fitted_model)
        fbs = _select_coded_value(
            "Fasting blood sugar (coded)", "fbs", fitted_model
        )
        restecg = _select_coded_value("Resting ECG (coded)", "restecg", fitted_model)
    with clinical_columns[2]:
        exang = _select_coded_value(
            "Exercise-induced angina (coded)", "exang", fitted_model
        )
        slope = _select_coded_value("ST segment slope (coded)", "slope", fitted_model)
        ca = _select_coded_value("Major vessels (ca, coded)", "ca", fitted_model)
        thal = _select_coded_value("Thalassemia (coded)", "thal", fitted_model)

    submitted = st.form_submit_button(
        "Estimate risk", type="primary", use_container_width=True
    )

if submitted:
    patient_record: dict[str, Any] = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    try:
        normalized_record = normalize_patient_record(patient_record)
        _render_results(normalized_record.iloc[0].to_dict())
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not calculate the patient estimate: {exc}")

st.divider()
st.caption(
    "The input codes are passed as stored in the supplied dataset. Confirm that "
    "your local dataset uses the same coding scheme before interpreting results."
)
