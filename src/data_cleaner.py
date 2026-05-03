"""Data cleaning utilities for ESS/LSMS household datasets.

Includes profiling, missing-value handling, outlier treatment, and de-duplication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


@dataclass
class CleaningSummary:
    dropped_sparse_cols: list[str] = field(default_factory=list)
    imputation_strategy: str = "median_mode"
    duplicate_rows_removed: int = 0
    outlier_caps: Dict[str, tuple[float, float]] = field(default_factory=dict)


class DataCleaner:
    """Production-friendly cleaner with explicit missing-value controls."""

    def __init__(self) -> None:
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.summary = CleaningSummary()

    def missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        rep = pd.DataFrame({
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(3),
            "dtype": df.dtypes.astype(str).values,
        })
        return rep.sort_values("missing_pct", ascending=False)

    def drop_sparse_columns(self, df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
        out = df.copy()
        sparse = out.columns[out.isna().mean() > threshold].tolist()
        self.summary.dropped_sparse_cols = sparse
        return out.drop(columns=sparse, errors="ignore")

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        use_knn_for_numeric: bool = False,
        knn_neighbors: int = 5,
    ) -> pd.DataFrame:
        """Impute missing values explicitly for numeric and categorical columns."""
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = out.select_dtypes(exclude=[np.number]).columns.tolist()

        if num_cols:
            if use_knn_for_numeric:
                self.numeric_imputer = KNNImputer(n_neighbors=knn_neighbors)
                out[num_cols] = self.numeric_imputer.fit_transform(out[num_cols])
                self.summary.imputation_strategy = f"knn_numeric(k={knn_neighbors})+{categorical_strategy}_categorical"
            else:
                self.numeric_imputer = SimpleImputer(strategy=numeric_strategy)
                out[num_cols] = self.numeric_imputer.fit_transform(out[num_cols])
                self.summary.imputation_strategy = f"{numeric_strategy}_numeric+{categorical_strategy}_categorical"

        if cat_cols:
            self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            out[cat_cols] = self.categorical_imputer.fit_transform(out[cat_cols])

        return out

    def cap_outliers_iqr(self, df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
            out[col] = out[col].clip(lower=low, upper=high)
            self.summary.outlier_caps[col] = (float(low), float(high))
        return out

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
        before = len(df)
        out = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        self.summary.duplicate_rows_removed = int(before - len(out))
        return out

    def clean(
        self,
        df: pd.DataFrame,
        sparse_threshold: float = 0.70,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        use_knn_for_numeric: bool = False,
    ) -> pd.DataFrame:
        out = self.drop_sparse_columns(df, threshold=sparse_threshold)
        out = self.handle_missing_values(
            out,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            use_knn_for_numeric=use_knn_for_numeric,
        )
        out = self.cap_outliers_iqr(out)
        out = self.remove_duplicates(out)
        return out
