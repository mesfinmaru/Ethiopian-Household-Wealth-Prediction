from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(df: pd.DataFrame, target_cols: list[str]) -> ColumnTransformer:
    feat = df.drop(columns=target_cols, errors="ignore")
    cat_cols = feat.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in feat.columns if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
