"""Project configuration and dataset schema."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
TARGET_COLUMN = "target"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "heart.csv"
MODEL_PATH = PROJECT_DIR / "models" / "cardio_risk_model.joblib"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
ENGINEERED_NUMERIC_FEATURES = ["max_hr_percent_predicted"]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    *ENGINEERED_NUMERIC_FEATURES,
]

# Values outside these broad plausibility bounds are treated as unknown (NaN),
# then imputed inside the training-fold pipeline. Valid extreme values remain.
CLINICAL_BOUNDS = {
    "age": (18, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (60, 260),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (30, 240),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
}
VALID_CATEGORIES = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3, 4},
    "thal": {0, 1, 2, 3},
}

LOW_RISK_MAX = 0.20
MEDIUM_RISK_MAX = 0.50

# This ranking emphasizes discrimination and identifying high-risk patients.
SELECTION_WEIGHTS = {"roc_auc": 0.40, "recall": 0.35, "f2": 0.25}

CV_SPLITS = 5
CV_REPEATS = 3
