from __future__ import annotations

import numpy as np
import pandas as pd


def add_household_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hh_size" not in out.columns:
        member_cols = [c for c in out.columns if "member" in c and out[c].dtype != "O"]
        out["hh_size"] = out[member_cols].sum(axis=1) if member_cols else 1

    child = out.get("num_children", 0)
    elder = out.get("num_elderly", 0)
    working = out.get("num_working_age", out["hh_size"] - child - elder)
    out["dependency_ratio"] = (child + elder) / np.maximum(working, 1)

    asset_cols = [c for c in out.columns if any(k in c for k in ["own_", "asset_", "radio", "tv", "fridge"]) ]
    if asset_cols:
        out["asset_index"] = out[asset_cols].fillna(0).astype(float).sum(axis=1)

    housing_cols = [c for c in out.columns if any(k in c for k in ["wall", "roof", "floor", "water", "toilet", "electric"]) ]
    if housing_cols:
        out["housing_quality_score"] = out[housing_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0).mean(axis=1)

    for col in ["region", "zone", "urban"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    leakage_patterns = ["cons", "consumption_agg", "food_exp", "nonfood_exp", "total_exp"]
    leakage = [c for c in out.columns if any(p in c for p in leakage_patterns) and c not in ["total_consumption"]]
    out = out.drop(columns=leakage, errors="ignore")
    return out


def build_targets(df: pd.DataFrame, poverty_line: float | None = None) -> pd.DataFrame:
    out = df.copy()
    total_col = "total_consumption" if "total_consumption" in out.columns else "consumption"
    out["consumption_per_capita"] = out[total_col] / np.maximum(out["hh_size"], 1)
    line = out["consumption_per_capita"].median() if poverty_line is None else poverty_line
    out["poor"] = (out["consumption_per_capita"] < line).astype(int)
    return out
