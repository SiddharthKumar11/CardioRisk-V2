"""End-to-end training, evaluation, interpretation, and model persistence."""

from __future__ import annotations

import joblib
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    SELECTION_WEIGHTS,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from data_preprocessing import (
    clean_dataset,
    inspect_dataset,
    load_dataset,
    report_statistical_outliers,
)
from evaluate_model import evaluate_model, tune_decision_threshold
from predict_risk import predict_patient_risk
from shap_explainability import explain_high_risk_patient
from train_models import compare_models, results_table


def main() -> None:
    raw_data = load_dataset(str(DATA_PATH))
    inspect_dataset(raw_data)
    features, target = clean_dataset(raw_data)

    print("\nCleaned class distribution:")
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
        f"\nStratified partitions: model-training={len(X_fit)}, "
        f"threshold-validation={len(X_validation)}, untouched-test={len(X_test)}; "
        "learned preprocessing is fitted inside training CV pipelines."
    )
    report_statistical_outliers(X_train)

    searches = compare_models(X_fit, y_fit)
    comparison = results_table(searches)
    print("\n=== Model comparison (sorted by CV ROC-AUC, then high-risk recall) ===")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = max(
        searches,
        key=lambda name: searches[name].selection_score,
    )
    best_model = searches[best_name].search.best_estimator_
    print(
        f"\nSelected model: {best_name} "
        f"(weighted CV objective: {searches[best_name].selection_score:.3f}; "
        f"weights={SELECTION_WEIGHTS})"
    )
    shap_background = X_fit.sample(
        n=min(50, len(X_fit)), random_state=RANDOM_STATE
    ).copy()
    best_model.shap_background_ = shap_background

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    decision_threshold, _ = tune_decision_threshold(
        best_model, X_validation, y_validation, OUTPUT_DIR
    )
    best_model.decision_threshold_ = decision_threshold
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        OUTPUT_DIR,
        candidate_models={
            name: result.search.best_estimator_
            for name, result in searches.items()
        },
        decision_threshold=decision_threshold,
    )
    print(f"\nSaved held-out metrics to console and plots/tables under {OUTPUT_DIR}.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    high_category_positions = (test_probabilities > 0.5).nonzero()[0]
    if len(high_category_positions):
        patient = X_test.iloc[[int(high_category_positions[0])]].copy()
        patient_source = "untouched test partition"
    else:
        fit_probabilities = best_model.predict_proba(X_fit)[:, 1]
        high_fit_positions = (fit_probabilities > 0.5).nonzero()[0]
        if not len(high_fit_positions):
            raise RuntimeError(
                "No High-category example is available for the requested SHAP demo."
            )
        patient = X_fit.iloc[[int(high_fit_positions[0])]].copy()
        patient_source = "model-training partition (no High-category test example)"

    prediction = predict_patient_risk(patient, MODEL_PATH)
    print(f"\n=== Example patient prediction ({patient_source}) ===")
    print(f"Predicted probability: {prediction['probability']:.3f}")
    print(f"Risk category: {prediction['risk_category']}")
    print(f"Doctor message: {prediction['recommended_message']}")
    if prediction["risk_category"] == "High":
        explanation = explain_high_risk_patient(patient, MODEL_PATH)
        print(
            "Top SHAP reasons increasing predicted disease probability "
            "(relative to the training background):"
        )
        for reason in explanation["positive_reasons"]:
            print(
                f"  {reason['feature']}={reason['patient_value']}: "
                f"SHAP {reason['shap_value']:+.3f}"
            )
        print(
            "Top risk-reducing SHAP contributions: "
            f"{explanation['risk_reducing_factors']}"
        )
        print(
            "SHAP base + contributions: "
            f"{explanation['shap_base_probability']:.3f} + "
            f"{sum(item['shap_value'] for item in explanation['all_feature_contributions']):+.3f} "
            f"= {explanation['shap_reconstruction_probability']:.3f}"
        )
        print(
            "Saved patient SHAP values and waterfall plot under "
            f"{OUTPUT_DIR}."
        )
    else:
        print(
            "No SHAP explanation produced because the example did not meet "
            "the High-category probability threshold."
        )
    print(
        "\nClinical note: This educational decision-support prototype is not "
        "a diagnostic device; predictions require clinician interpretation."
    )
    print(f"Test ROC-AUC={metrics['roc_auc']:.3f}, high-risk recall={metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
