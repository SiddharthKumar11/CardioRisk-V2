# CardioRiskML

CardioRiskML is a small, reproducible Python project for exploring heart-disease
risk prediction from tabular patient data. It is an educational decision-support
prototype, not a medical device, diagnostic service, or substitute for clinician
judgment.

## Dataset assumptions

- The CSV is `heart.csv` in this project directory (or update `DATA_PATH` in
  `config.py`).
- It contains the 13 numeric feature columns used by the supplied file and a
  binary `target` (`1` = heart disease/high-risk outcome; `0` = no heart disease).
- The supplied dataset has 1,025 rows and 14 columns, with no missing cells,
  723 exact duplicate rows, and target counts of 499 zeros and 526 ones.
  Duplicate removal leaves 302 unique records. Dataset version and duplication
  patterns should be checked before interpreting clinical performance.
- The dataset contains coded categories (such as `cp` and `thal`); these are
  treated as categorical, not continuous measurements.

Cleaning converts feature values to numeric, marks values outside broad
plausibility bounds or invalid categorical codes as missing, and drops rows
without a valid binary target. It removes exact duplicates before splitting:
otherwise identical records could appear in both train and test partitions.
statistical outliers are flagged using training-partition IQR fences but not
automatically removed, because plausible clinical extremes should be retained.
All imputers, encoders, scalers, and classifiers are fitted only within training
folds. Missing numeric values are median-imputed; categorical values use the
most frequent category.

## Installation

Use Python 3.10 or newer, then install dependencies from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Project files

- `config.py`: dataset paths, feature schema, plausibility limits, CV settings,
  and risk thresholds.
- `data_preprocessing.py`: raw inspection, CSV loading, duplicate removal,
  schema checks, and patient-input normalization.
- `feature_engineering.py`: pipeline-integrated, stateless clinical feature.
- `train_models.py`: preprocessing pipelines, repeated stratified CV, and
  hyperparameter/model comparison.
- `evaluate_model.py`: held-out metrics, plots, permutation importance, and
  tree-based feature importance.
- `predict_risk.py`: reusable prediction API and command-line CSV interface.
- `shap_explainability.py`: individual High-category SHAP probability
  explanations.
- `app.py`: Streamlit patient-entry and risk explanation interface.
- `main.py`: end-to-end training and evaluation entry point.

## Train and evaluate

Place the dataset at `heart.csv`, then run:

```powershell
python main.py
```

The program inspects the raw CSV, cleans duplicate/invalid records, performs a
stratified held-out split, and compares Logistic Regression, Support Vector
Machine, Random Forest, Gradient Boosting, and K-Nearest Neighbors. Hyperparameter
selection uses RepeatedStratifiedKFold (5 splits, 3 repeats) on training data.
The refit objective weights CV ROC-AUC (0.40), high-risk recall (0.35), and
high-risk F2 (0.25). The printed comparison table is sorted by ROC-AUC, then
recall. The selected fitted pipeline is saved to
`models/cardio_risk_model.joblib`.

The development partition is split again into model-training and validation
subsets. Validation probabilities select the binary high-risk decision cutoff
by maximizing the equal-weighted mean of accuracy and high-risk F1, breaking
ties by recall and then precision. The complete precision/recall/F1/accuracy
trade-off across thresholds 0.00 to 1.00 is saved
to `outputs/threshold_tradeoff.csv`. The selected cutoff is stored with the
pipeline and exposed as `high_risk_prediction` by the prediction API. The
configured three probability groups (Low/Medium/High) remain independent of
this binary cutoff.

The untouched test set is evaluated once using the validation-selected cutoff
for binary metrics. Curves and interpretation tables are written to `outputs/`.

## Diagnose model performance

Run the leakage-safe diagnostic report with:

```powershell
python diagnose_model.py
```

It reproduces the stratified train/validation/test partitions from `main.py`,
uses nested CV for model-selection performance, and writes learning curves,
validation curves, partition metrics, nested-CV fold scores, and validation-set
permutation importance under `outputs/diagnostics/`. It does not modify or
refit the saved model. The test partition is used only for final scoring; the
saved threshold (selected on validation) is reported separately on test so its
validation score is not presented as unbiased.

## Make a prediction

Import the function and provide a Python mapping:

```python
from predict_risk import predict_patient_risk

patient = {
    "age": 58, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
}
result = predict_patient_risk(patient)
print(result)
```

Alternatively, provide a one-row CSV:

```powershell
python predict_risk.py --csv patient.csv
```

The Python function also accepts a dictionary directly, as shown above. Risk
thresholds in `config.py` are configurable: Low is probability below 0.20,
Medium is 0.20 through 0.50 inclusive, and High is greater than 0.50. Missing
input fields are imputed by the saved pipeline; invalid/out-of-range values are
treated as missing. The output also includes `high_risk_prediction`, which
applies the separate decision cutoff tuned on the validation partition.

## Streamlit doctor interface

Install the project dependencies (including Streamlit), train the model, then
launch the web app:

```powershell
python -m pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```

The form accepts all 13 dataset features and displays the probability, three
risk bands, the separately tuned binary high-risk flag, and SHAP contributions
that increase or decrease the estimate. Coded categorical fields are shown as
numeric codes because the supplied dataset does not include a verified
human-readable codebook.

## Evaluation metrics

- **Accuracy:** fraction of all predictions that are correct. It can obscure
  missed cases when class balance changes.
- **Precision:** fraction of predicted high-risk patients who have a positive
  outcome.
- **Recall (sensitivity):** fraction of positive/high-risk outcomes detected.
  For screening, missed high-risk patients can be especially consequential.
- **F1 / F2:** harmonic means of precision and recall; F2 weights recall more
  strongly than F1.
- **ROC-AUC:** ability to rank positive cases above negative cases across
  classification thresholds.
- **PR-AUC:** precision-recall performance across thresholds; useful when
  positive cases are uncommon.
- **Confusion matrix / classification report:** counts and per-class measures
  that expose false negatives and false positives.
- **Calibration curve:** checks whether predictions near (for example) 0.20
  correspond to roughly 20% observed positive outcomes. Calibration matters
  when probabilities are interpreted as risk.

Recall and probability calibration matter more than accuracy alone for medical
screening: accuracy may look high while positive cases are missed, and an
uncalibrated probability can mislead risk-based decisions. This dataset is small
and has substantial duplicates; reported metrics are not evidence of clinical
validity. Independent, representative external validation and clinical review
are required before any real-world use.

## Interpretation and safety

Permutation importance reports the held-out test ROC-AUC decrease when an input
feature is shuffled. Intrinsic feature importance is printed for the fitted
Random Forest and Gradient Boosting candidates.

`explain_high_risk_patient(record)` in `shap_explainability.py` explains a
patient whose Low/Medium/High category is **High** (predicted probability over
0.50). It uses SHAP's permutation explainer on the saved model's positive-class
probability and a background sample drawn only from model-training rows. It
returns the largest probability-increasing and probability-decreasing feature
contributions, and writes `outputs/high_risk_shap_values.csv` plus a waterfall
plot. The model artifact created by `main.py` includes this training-only
background sample. If loading another model, pass `background_data` containing
training rows with the expected feature columns.

For example, after training:

```python
from shap_explainability import explain_high_risk_patient

explanation = explain_high_risk_patient(patient)
print(explanation["probability"])
print(explanation["positive_reasons"])
```

SHAP values are contributions relative to the chosen training background,
expressed on the positive-class probability scale. They describe the model's
behavior, not causation, patient-specific clinical mechanisms, or validated
medical reasons. A binary prediction can use the separately tuned threshold;
the explanation function requires High risk category (probability > 0.50).
All thresholds, recommendations, model outputs, and explanations require
clinician interpretation.
