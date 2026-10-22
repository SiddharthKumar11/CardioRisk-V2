"""Dataset loading, inspection, and stateless data-quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import (
    CLINICAL_BOUNDS,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_CATEGORIES,
)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset, reporting file and parsing errors clearly."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not load dataset at {path!r}: {exc}") from exc


def inspect_dataset(data: pd.DataFrame) -> None:
    """Print raw-data shape, schema, missingness, duplicates, classes, and stats."""
    print("\n=== Raw dataset inspection ===")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values per column:")
    print(data.isna().sum().to_string())
    print(f"\nExact duplicate rows: {int(data.duplicated().sum())}")
    if TARGET_COLUMN in data.columns:
        print("\nClass distribution:")
        print(data[TARGET_COLUMN].value_counts(dropna=False).sort_index().to_string())
    print("\nBasic statistics:")
    print(data.describe(include="all").to_string())


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean invalid records and exact duplicates without learning from data."""
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = data.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    print(f"\nRemoved {before - len(frame)} exact duplicate rows.")

    invalid_outcomes = frame[TARGET_COLUMN].notna() & ~frame[TARGET_COLUMN].isin([0, 1])
    invalid_target_rows = int(frame[TARGET_COLUMN].isna().sum() + invalid_outcomes.sum())
    if invalid_target_rows:
        print(f"Dropping {invalid_target_rows} rows with missing or invalid target values.")
    frame = frame.loc[~invalid_outcomes & frame[TARGET_COLUMN].notna()].copy()

    invalid_counts: dict[str, int] = {}
    for column, (minimum, maximum) in CLINICAL_BOUNDS.items():
        invalid = frame[column].notna() & ~frame[column].between(minimum, maximum)
        if column in VALID_CATEGORIES:
            invalid |= frame[column].notna() & ~frame[column].isin(VALID_CATEGORIES[column])
        count = int(invalid.sum())
        if count:
            invalid_counts[column] = count
            frame.loc[invalid, column] = np.nan
    if invalid_counts:
        print(f"Marked clinically implausible feature values missing: {invalid_counts}")
    else:
        print("No values outside the configured clinical plausibility bounds.")
    print(
        "Statistical outliers are not automatically deleted: clinically possible "
        "extremes are retained, and model preprocessing imputes only invalid/missing "
        "values using statistics learned from each training fold."
    )

    features = frame.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    target = frame[TARGET_COLUMN].astype(int).reset_index(drop=True)
    if target.nunique() != 2:
        raise ValueError("The target must contain both binary classes 0 and 1.")
    return features, target


def report_statistical_outliers(training_features: pd.DataFrame) -> None:
    """Report training-partition IQR outliers for review without removing them."""
    print("\nIQR outlier review on training partition (flagged values retained):")
    for column in CONTINUOUS_FEATURES:
        values = training_features[column].dropna()
        lower_quartile = values.quantile(0.25)
        upper_quartile = values.quantile(0.75)
        spread = upper_quartile - lower_quartile
        lower_fence = lower_quartile - 1.5 * spread
        upper_fence = upper_quartile + 1.5 * spread
        flagged = int(((values < lower_fence) | (values > upper_fence)).sum())
        print(
            f"  {column}: {flagged} outside [{lower_fence:.2f}, "
            f"{upper_fence:.2f}]"
        )


def normalize_patient_record(record: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Convert a patient mapping or one-row CSV DataFrame to model input."""
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise ValueError("A patient DataFrame must contain exactly one row.")
        patient = record.copy()
    elif isinstance(record, dict):
        patient = pd.DataFrame([record])
    else:
        raise TypeError("Patient input must be a dictionary or one-row DataFrame.")

    unexpected = sorted(set(patient.columns) - set(FEATURE_COLUMNS) - {TARGET_COLUMN})
    if unexpected:
        raise ValueError(f"Unexpected patient feature columns: {unexpected}")
    patient = patient.drop(columns=[TARGET_COLUMN], errors="ignore")
    for column in FEATURE_COLUMNS:
        if column not in patient:
            patient[column] = np.nan
        patient[column] = pd.to_numeric(patient[column], errors="coerce")
        minimum, maximum = CLINICAL_BOUNDS[column]
        invalid = patient[column].notna() & ~patient[column].between(minimum, maximum)
        if column in VALID_CATEGORIES:
            invalid |= patient[column].notna() & ~patient[column].isin(
                VALID_CATEGORIES[column]
            )
        patient.loc[invalid, column] = np.nan
    return patient.loc[:, FEATURE_COLUMNS]
