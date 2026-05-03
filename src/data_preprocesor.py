"""Backwards-compatible preprocessing module.

This file intentionally keeps the legacy filename `data_preprocesor.py` while
exposing a complete scikit-learn pipeline API.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """Reusable train/inference preprocessor with fit/transform API."""

    def __init__(self, target_columns: list[str] | None = None) -> None:
        self.target_columns = target_columns or []
        self.pipeline: ColumnTransformer | None = None
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []

    def build(self, df: pd.DataFrame) -> ColumnTransformer:
        X = df.drop(columns=self.target_columns, errors="ignore")
        self.categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.numeric_columns = [c for c in X.columns if c not in self.categorical_columns]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        self.pipeline = ColumnTransformer([
            ("num", numeric_pipe, self.numeric_columns),
            ("cat", categorical_pipe, self.categorical_columns),
        ])
        return self.pipeline

    def fit(self, df: pd.DataFrame):
        if self.pipeline is None:
            self.build(df)
        X = df.drop(columns=self.target_columns, errors="ignore")
        self.pipeline.fit(X)
        return self

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            raise RuntimeError("Call fit or build first.")
        X = df.drop(columns=self.target_columns, errors="ignore")
        return self.pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
