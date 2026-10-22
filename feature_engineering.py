"""Leakage-safe, stateless feature derivation for patient records."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add maximum-heart-rate percentage relative to the age-based estimate."""

    def fit(self, X: pd.DataFrame, _y: object = None) -> "ClinicalFeatureEngineer":
        del _y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer expects a pandas DataFrame.")
        required = {"age", "thalach"}
        if not required.issubset(X.columns):
            raise ValueError(f"Missing inputs for feature engineering: {required - set(X.columns)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        transformed = X.copy()
        estimated_maximum = 220 - pd.to_numeric(transformed["age"], errors="coerce")
        observed_maximum = pd.to_numeric(transformed["thalach"], errors="coerce")
        transformed["max_hr_percent_predicted"] = observed_maximum / estimated_maximum
        transformed.loc[estimated_maximum <= 0, "max_hr_percent_predicted"] = np.nan
        return transformed

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.append(names, "max_hr_percent_predicted")
