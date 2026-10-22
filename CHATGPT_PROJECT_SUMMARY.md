# CardioRiskML: Project Summary for Further Review

This document summarizes the implemented heart-disease risk project and its
saved diagnostic results. It is intended as context for an AI assistant or
technical reviewer. Treat the reported performance as exploratory: the dataset
is small, has many exact duplicate records, and is not an external clinical
validation cohort.

## Dataset

- Source file in this workspace: `heart.csv` (the diagnostic script also checks
  `data/heart.csv` first, if present).
- Raw dataset: **1,025 rows × 14 columns**.
- Columns: 13 input features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`,
  `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`) and binary
  `target`.
- Raw target counts: 499 class 0 and 526 class 1.
- Missing values in the supplied raw CSV: none.
- Exact duplicate rows: **723**. After exact-duplicate removal, 302 rows remain
  (138 class 0, 164 class 1).
- The supplied data has numeric category codes but no verified clinical
  codebook. Code values should not be given human-readable clinical meanings
  without checking the dataset documentation.

## Peak nested-CV fold metrics

The highest observed accuracy was in **outer fold 5** of nested
cross-validation. That fold selected **Random Forest**:

| Measure | Peak fold result |
|---|---:|
| Accuracy | **97.37%** |
| Precision | **95.24%** |
| Recall / sensitivity | **100.00%** |
| F1-score | **97.56%** |
| ROC-AUC | 98.06% |

This is a **single fold's peak**, not the overall cross-validation accuracy,
test accuracy, or expected performance on new patients. The five outer-fold
accuracies were 89.74%, 89.74%, 73.68%, 92.11%, and 97.37%. Their unweighted
mean accuracy was **88.53%** (range: 73.68%–97.37%). The fold metrics are in
`outputs/diagnostics/nested_cv_folds.csv`.

### Other key results for context

- Fixed 0.50 threshold, nested-CV mean: accuracy **88.53%**, precision
  **85.82%**, recall **95.24%**, F1 **90.21%**.
- Fixed 0.50 threshold, untouched held-out test: accuracy **78.69%**, precision
  **79.41%**, recall **81.82%**, F1 **80.60%**, ROC-AUC **87.39%**.
- Validation-tuned 0.34 operating cutoff, untouched held-out test: accuracy
  **80.33%**, precision **78.43%**, recall **87.88%**, F1 **82.93%**.
- The saved-model comparison CSV reports CV ROC-AUC, recall, F2, and a weighted
  selection score; it does not report accuracy. Do not interpret its ROC-AUC
  values as accuracy.
- Threshold 0.34 was selected using the validation subset by maximizing the
  equal-weighted mean of accuracy and F1, with recall and precision as tie
  breakers. The three probability bands remain Low < 0.20, Medium 0.20–0.50,
  and High > 0.50.

## Methodology

1. Inspect the raw CSV schema, data types, missingness, duplicate count, class
   balance, and descriptive statistics.
2. Convert features to numeric, drop exact duplicate rows, remove rows with an
   invalid/missing target, and mark values outside broad configured
   plausibility bounds as missing. Statistical outliers are flagged for review,
   not automatically removed.
3. Use stratified splits with `random_state=42`: 20% held out for test; the
   remaining 80% is divided into model-training and threshold-validation
   subsets (validation is 20% of the development subset). In the diagnostic
   run this produced 192 model-training, 49 validation, and 61 test records.
4. Keep learned preprocessing in scikit-learn pipelines: numeric median
   imputation and scaling; categorical most-frequent imputation and one-hot
   encoding. Preprocessing is fit within model-training CV folds, not on the
   full dataset or held-out test data.
5. Compare and tune candidate models using stratified cross-validation. The
   nested diagnostic uses five outer folds; model/hyperparameter selection is
   performed with three inner folds on each outer training fold. Outer folds
   are used only for scoring. Accuracy, precision, recall, F1, and ROC-AUC are
   summarized. The outer-fold classification metrics use a fixed 0.50 cutoff.
6. Select the saved pipeline using a CV weighted objective of ROC-AUC (0.40),
   high-risk recall (0.35), and F2 (0.25). The validation subset is used for
   binary threshold selection; the test partition is left untouched until final
   evaluation.
7. Report learning curves, validation curves, validation-set permutation
   importance, and simple-vs-complex comparisons as diagnostics. Permutation
   importance on this small validation subset is noisy and is not proof that a
   feature is or is not clinically useful.

The phrase **“peak accuracy”** should always be qualified as “peak accuracy in
one nested-CV fold,” alongside the mean and range. For a paper, do not report
97.37% as the project's overall CV accuracy. Prefer reporting the five-fold
mean and spread, plus the untouched test result and its decision threshold.

## Models compared

- Logistic Regression (`class_weight="balanced"`).
- Support Vector Machine (`class_weight="balanced"`), with probability
  calibration inside the pipeline.
- Random Forest (`class_weight="balanced"`).
- scikit-learn Gradient Boosting.
- K-Nearest Neighbors.

The nested-CV peak fold selected Random Forest. The saved main-run model was
K-Nearest Neighbors. These are different model-selection outcomes in their
respective evaluation procedures; do not conflate the peak-fold model with the
final saved pipeline. No SMOTE was applied. Class weights were used for
estimators that support them.

## Software stack

- Python.
- pandas and NumPy for tabular data handling.
- scikit-learn for preprocessing, pipelines, model selection, classifiers, and
  evaluation.
- matplotlib and seaborn for plots.
- SHAP for individual prediction explanations.
- joblib for pipeline persistence.
- Streamlit for the clinician-facing prototype interface.

The project requirements are listed in `requirements.txt`. It uses a
conventional tabular machine-learning stack; it does **not** implement a deep
neural network.

## Outputs and source files

- `outputs/diagnostics/nested_cv_folds.csv`: outer-fold results and per-fold
  selected model.
- `outputs/diagnostics/partition_metrics.csv`: training, CV, validation, and
  test metrics at the diagnostic 0.50 cutoff.
- `outputs/diagnostics/learning_curve.png` and `learning_curve.csv`.
- `outputs/diagnostics/validation_curves.png` and `validation_curves.csv`.
- `outputs/diagnostics/permutation_importance.csv`.
- `outputs/model_comparison.csv`: model CV ROC-AUC, recall, F2, and selection
  score.
- `models/cardio_risk_model.joblib`: saved trained pipeline.
- `main.py`, `train_models.py`, `evaluate_model.py`, and `diagnose_model.py`:
  training, evaluation, and diagnostic workflows.
- `app.py`: Streamlit interface.

## Suggested concise wording for a report

> “Across five outer folds of nested cross-validation, mean accuracy was
> 88.53% (range: 73.68%–97.37%). The peak fold accuracy was 97.37% (Random
> Forest; precision 95.24%, recall 100.00%, F1-score 97.56%). Accuracy on the
> untouched test set was 78.69% at a 0.50 classification threshold. At the
> validation-selected 0.34 threshold, test accuracy was 80.33% (precision
> 78.43%, recall 87.88%, F1-score 82.93%).”

These estimates are based on a small, duplicate-heavy dataset and should not be
presented as evidence of clinical readiness. Independent external validation,
appropriate patient/site-level splitting where applicable, uncertainty
intervals, calibration assessment, and clinical review are required.
