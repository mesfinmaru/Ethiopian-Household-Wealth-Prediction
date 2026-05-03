"""
data_cleaner.py — CRISP-DM Phase 2: Data Cleaning
Imputation, outlier handling, and feature validation.

Imputation strategy by feature type:
  Ordinal housing (roof/wall/floor/water/toilet/fuel)
    → settlement-level mode (rural/urban households differ systematically)
  Binary assets (owns_phone/tv/fridge, has_electricity)
    → 0 (absence of data = likely not owned, esp. in early waves)
  Continuous demographics (head_age, hh_size)
    → region-level median
  Labour participation
    → 0 (absent = not working for pay)
  Shocks
    → 0 (absent shock data = not affected)
  W4/W5-only columns for W1/W3 rows
    → NaN kept for tree models; flagged with indicator column
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer

warnings.filterwarnings("ignore")

from config import (
    ALL_FEATURES, ASSET_FEAT, DEMO_COLS, DERIVED_FEAT, ENTERPRISE_F,
    GEO_COLS, HEAD_COLS, HOUSING_FEAT, LABOUR_FEAT, SHOCK_FEAT, TARGET,
)


class DataCleaner:
    """
    Clean the merged multi-wave ESS dataset.

    Usage
    -----
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(df)
        report   = cleaner.report()   # DataFrame for notebook display
    """

    def __init__(self) -> None:
        self._log: list[dict] = []
        self._imputers: dict = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full cleaning pipeline.

        Steps
        -----
        1. Drop rows with missing target
        2. Impute by feature group
        3. Cap numeric outliers (IQR × 3 per wave)
        4. Remove zero-variance columns
        5. Add wave-coverage indicator columns

        Returns cleaned DataFrame.
        """
        df = df.copy()
        df = self._drop_missing_target(df)
        df = self._impute(df)
        df = self._cap_outliers(df)
        df = self._drop_zero_variance(df)
        df = self._add_coverage_flags(df)
        return df

    def report(self) -> pd.DataFrame:
        """Return cleaning log as a DataFrame for notebook display."""
        return pd.DataFrame(self._log)

    def missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Missing value summary per column."""
        miss = df.isnull().sum()
        pct  = (miss / len(df) * 100).round(2)
        out  = pd.DataFrame({"n_missing": miss, "pct_missing": pct})
        return out[out["n_missing"] > 0].sort_values("pct_missing", ascending=False)

    # ── Cleaning steps ────────────────────────────────────────────────────

    def _drop_missing_target(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[TARGET]).copy()
        n = before - len(df)
        self._log.append({"step": "drop_missing_target", "rows_removed": n,
                          "rows_remaining": len(df)})
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute each feature group with its appropriate strategy."""
        n_before = df.isnull().sum().sum()

        # ── Binary assets: 0 (no ownership data = not owned)
        binary_cols = [c for c in ASSET_FEAT + ["has_electricity"]
                       if c in df.columns]
        df[binary_cols] = df[binary_cols].fillna(0)

        # ── Shocks: 0 (no shock data = not affected)
        shock_cols = [c for c in SHOCK_FEAT if c in df.columns]
        df[shock_cols] = df[shock_cols].fillna(0)

        # ── Labour: 0 (no record = not working for pay)
        labour_cols = [c for c in LABOUR_FEAT if c in df.columns]
        df[labour_cols] = df[labour_cols].fillna(0)

        # ── Enterprise: 0 (W4/W5 have no enterprise section)
        ent_cols = [c for c in ENTERPRISE_F if c in df.columns]
        df[ent_cols] = df[ent_cols].fillna(0)

        # ── Ordinal housing: settlement-level mode
        housing_ordinal = [c for c in ["roof","wall","floor","water","toilet","fuel"]
                           if c in df.columns]
        for col in housing_ordinal:
            if df[col].isnull().any():
                mode_fill = df.groupby("settlement")[col].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                )
                df[col] = df[col].fillna(mode_fill).fillna(df[col].median())

        # ── Rooms: settlement-level median
        if "rooms" in df.columns and df["rooms"].isnull().any():
            df["rooms"] = df["rooms"].fillna(
                df.groupby("settlement")["rooms"].transform("median")
            )

        # ── housing_score: recompute from filled components if needed
        if "housing_score" in df.columns and df["housing_score"].isnull().any():
            df["housing_score"] = df["housing_score"].fillna(
                df.groupby("settlement")["housing_score"].transform("median")
            )

        # ── Head demographics: region-level median
        head_cont = [c for c in ["head_age","head_age_sq","head_edu_level"]
                     if c in df.columns]
        for col in head_cont:
            if df[col].isnull().any():
                df[col] = df[col].fillna(
                    df.groupby("region")[col].transform("median")
                ).fillna(df[col].median())

        # ── Head binary: mode
        head_bin = [c for c in ["head_sex","is_female_headed","head_literate"]
                    if c in df.columns]
        for col in head_bin:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

        # ── Household size & adulteq: should have no missing (from cons_agg)
        for col in ["hh_size","adulteq"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # ── Derived features: recompute after imputation
        df = self._recompute_derived(df)

        n_after = df.isnull().sum().sum()
        self._log.append({"step": "impute", "nulls_filled": int(n_before - n_after),
                          "nulls_remaining": int(n_after)})
        return df

    def _recompute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute derived columns after imputation."""
        if "hh_size" in df.columns:
            df["log_hh_size"] = np.log1p(df["hh_size"])
        if all(c in df.columns for c in ["hh_size","adulteq"]):
            df["dependency_ratio"] = (
                (df["hh_size"] - df["adulteq"]).clip(lower=0)
                / df["hh_size"].replace(0, np.nan)
            ).fillna(0)
        if "asset_count" in df.columns and "hh_size" in df.columns:
            df["assets_per_member"] = (
                df["asset_count"] / df["hh_size"].replace(0, np.nan)
            ).fillna(0)
        return df

    def _cap_outliers(
        self,
        df: pd.DataFrame,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """
        IQR-based outlier capping per wave on continuous features.
        Uses multiplier=3.0 (conservative — only extreme outliers).
        Does NOT affect binary/ordinal/target columns.
        """
        continuous = ["head_age","rooms","hh_avg_weeks_worked",
                      "hh_n_workers","enterprise_asset_count",
                      "housing_score","asset_count","n_shocks"]
        continuous = [c for c in continuous if c in df.columns]

        capped = 0
        for col in continuous:
            for wave in df["wave"].unique():
                mask = df["wave"] == wave
                q1   = df.loc[mask, col].quantile(0.25)
                q3   = df.loc[mask, col].quantile(0.75)
                iqr  = q3 - q1
                lo   = q1 - multiplier * iqr
                hi   = q3 + multiplier * iqr
                before = df.loc[mask, col].copy()
                df.loc[mask, col] = df.loc[mask, col].clip(lo, hi)
                capped += (before != df.loc[mask, col]).sum()

        self._log.append({"step": "cap_outliers",
                          "values_capped": int(capped),
                          "method": f"IQR × {multiplier} per wave"})
        return df

    def _drop_zero_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove numeric columns with near-zero variance."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target and IDs
        excl = [TARGET, "household_id", "wave", "zone_id"]
        check_cols = [c for c in num_cols if c not in excl]

        if not check_cols:
            return df

        sel = VarianceThreshold(threshold=1e-4)
        sel.fit(df[check_cols].fillna(0))
        kept    = [c for c, s in zip(check_cols, sel.get_support()) if s]
        dropped = [c for c in check_cols if c not in kept]

        if dropped:
            df = df.drop(columns=dropped)
            self._log.append({"step": "drop_zero_variance",
                              "dropped": dropped})
        return df

    def _add_coverage_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wave-coverage indicator columns for features not in all waves.
        These flags help tree models learn that 0 ≠ 'not applicable'.
        """
        # housing_available: W1/W3 have richer housing; W4/W5 have limited
        df["has_full_housing"] = df["wave"].isin([1, 2, 3]).astype(int)
        # enterprise_available: only W1/W2/W3 have sect7
        df["has_enterprise_data"] = df["wave"].isin([1, 2, 3]).astype(int)
        self._log.append({"step": "add_coverage_flags",
                          "flags_added": ["has_full_housing","has_enterprise_data"]})
        return df

    # ── Convenience methods ───────────────────────────────────────────────

    def knn_impute(
        self,
        df: pd.DataFrame,
        cols: list[str],
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """
        Optional KNN imputation for specific columns.
        More accurate than median for spatially structured data but slower.
        """
        df = df.copy()
        valid = [c for c in cols if c in df.columns]
        imp = KNNImputer(n_neighbors=n_neighbors)
        df[valid] = imp.fit_transform(df[valid])
        self._imputers["knn"] = imp
        return df