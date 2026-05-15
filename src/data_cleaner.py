"""
data_cleaner.py
═══════════════════════════════════════════════════════════════════════════════
Wraps MissingValueHandler with outlier detection, variance filtering,
and coverage flags. Follows the class-reference DataPreprocessor structure

Pipeline: drop_missing_target → impute → outliers → coverage_flags → low_variance
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from missing_value_handler import MissingValueHandler


class DataCleaner:
    """
    Full data cleaning pipeline for the merged ESS multi-wave dataset.

    Follows class-reference interface from Chapter 2:
      load_data, describe_data, detect_missing_values,
      impute_missing, detect_outliers, handle_outliers,
      fit_transform (main pipeline)

    Usage
    -----
        cleaner   = DataCleaner()
        df_clean  = cleaner.fit_transform(df_raw)
        cleaner.report()            # cleaning log
        cleaner.missing_report(df)  # missing value summary
    """

    def __init__(self):
        self.handler = MissingValueHandler()
        self._log    = []

    # ── Step-by-step methods (class-reference interface) ──────────────────────

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the combined CSV produced by build_all_waves()."""
        from pathlib import Path
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run build_all_waves() first.")
        df = pd.read_csv(path, low_memory=False)
        for col in ("region", "zone_label"):
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def describe_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numeric summary statistics (Chapter 2: data understanding step)."""
        return df.describe().round(3)

    def detect_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return missing value report with recommended strategies."""
        return self.handler.missing_report(df)

    def drop_missing_target(self, df: pd.DataFrame,
                            target: str = "cons_quint") -> pd.DataFrame:
        """
        Drop rows where the prediction target is NaN.
        CRISP-DM: target availability is a prerequisite for supervised learning.
        """
        before = len(df)
        df     = df.dropna(subset=[target]).copy()
        n      = before - len(df)
        self._log.append({"step":"drop_missing_target",
                          "rows_dropped":n, "rows_remaining":len(df)})
        return df

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full ESS missing-value pipeline via MissingValueHandler.ess_pipeline().
        Seven-step strategy (wave coverage → MNAR flag → W2 gaps → group →
        mode → KNN → global median).
        """
        before = df.isnull().sum().sum()
        df     = self.handler.ess_pipeline(df)
        after  = df.isnull().sum().sum()
        self._log.append({"step":"impute_missing",
                          "nulls_filled":int(before-after),
                          "nulls_remaining":int(after)})
        return df

    def detect_outliers(self, df: pd.DataFrame,
                        cols: list = None,
                        multiplier: float = 3.0) -> pd.DataFrame:
        """
        IQR-based outlier detection per wave (Chapter 2: outlier analysis).
        Returns summary DataFrame of affected values per feature per wave.
        Uses IQR × 3.0 (conservative — only flags extreme outliers).
        """
        if cols is None:
            cols = ["head_age","rooms","hh_n_workers",
                    "enterprise_asset_count","housing_score","n_shocks"]
        cols = [c for c in cols if c in df.columns]
        rows = []
        for col in cols:
            for wave in df["wave"].unique():
                sub      = df.loc[df["wave"] == wave, col].dropna()
                q1, q3   = sub.quantile(0.25), sub.quantile(0.75)
                iqr      = q3 - q1
                lo, hi   = q1 - multiplier * iqr, q3 + multiplier * iqr
                n_out    = int(((sub < lo) | (sub > hi)).sum())
                if n_out > 0:
                    rows.append({"feature":col,"wave":int(wave),
                                 "n_outliers":n_out,
                                 "lower_bound":round(lo,2),
                                 "upper_bound":round(hi,2)})
        return pd.DataFrame(rows)

    def handle_outliers(self, df: pd.DataFrame,
                        cols: list = None,
                        multiplier: float = 3.0) -> pd.DataFrame:
        """
        Cap outliers at IQR × multiplier per wave (winsorisation).
        Conservative default (3.0) preserves legitimate extreme values.
        """
        if cols is None:
            cols = ["head_age","rooms","hh_n_workers",
                    "enterprise_asset_count","housing_score","n_shocks"]
        cols   = [c for c in cols if c in df.columns]
        capped = 0
        for col in cols:
            for wave in df["wave"].unique():
                mask     = df["wave"] == wave
                q1, q3   = df.loc[mask, col].quantile([0.25, 0.75])
                iqr      = q3 - q1
                lo, hi   = q1 - multiplier * iqr, q3 + multiplier * iqr
                before   = df.loc[mask, col].copy()
                df.loc[mask, col] = df.loc[mask, col].clip(lo, hi)
                capped  += int((before != df.loc[mask, col]).sum())
        self._log.append({"step":"handle_outliers",
                          "values_capped":capped,
                          "method":f"IQR×{multiplier} per wave"})
        return df

    def add_coverage_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary flags marking which waves have richer data coverage.
        Helps tree models distinguish 'absent feature' from 'zero value',
        which are semantically different for structural absences.
        """
        df = df.copy()
        df["has_full_housing"]    = df["wave"].isin([1, 2, 3]).astype(int)
        df["has_enterprise_data"] = df["wave"].isin([1, 2, 3]).astype(int)
        self._log.append({"step":"add_coverage_flags",
                          "flags":["has_full_housing","has_enterprise_data"]})
        return df

    def drop_zero_variance(self, df: pd.DataFrame,
                           threshold: float = 1e-4) -> pd.DataFrame:
        """
        Remove numeric columns with near-zero variance (uninformative features).
        Chapter 2: feature validation — variance threshold filtering.
        """
        excl  = ["cons_quint","household_id","wave","zone_id"]
        num   = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in excl]
        if not num:
            return df
        sel     = VarianceThreshold(threshold=threshold)
        sel.fit(df[num].fillna(0))
        kept    = [c for c, s in zip(num, sel.get_support()) if s]
        dropped = [c for c in num if c not in kept]
        if dropped:
            df = df.drop(columns=dropped)
            self._log.append({"step":"drop_zero_variance","dropped":dropped})
        return df

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full CRISP-DM Phase 2 cleaning pipeline:
          1. drop_missing_target
          2. impute_missing  (7-step ESS pipeline)
          3. handle_outliers (IQR×3 per wave)
          4. add_coverage_flags
          5. drop_zero_variance
        """
        df = self.drop_missing_target(df)
        df = self.impute_missing(df)
        df = self.handle_outliers(df)
        df = self.add_coverage_flags(df)
        df = self.drop_zero_variance(df)

        # Drop panel-link columns from the exported clean dataset.
        # They are useful for diagnostics, but not for modeling, and they are
        # the only remaining source of nulls after the cleaning pipeline.
        id_cols = [c for c in ("household_id_w1",) if c in df.columns]
        if id_cols:
            df = df.drop(columns=id_cols)
            self._log.append({"step": "drop_identifier_columns",
                              "dropped": id_cols})
        return df

    # ── Reporting ──────────────────────────────────────────────────────────────

    def report(self) -> pd.DataFrame:
        """Return cleaning log as a DataFrame (each step is one row)."""
        return pd.DataFrame(self._log)

    def missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to MissingValueHandler.missing_report()."""
        return self.handler.missing_report(df)

    def stb_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """sidetable-style missing summary (n_missing, pct, cumulative)."""
        miss  = df.isnull().sum()
        pct   = (miss / len(df) * 100).round(2)
        out   = pd.DataFrame({"n_missing":miss,"pct_missing":pct})
        out   = out[out["n_missing"]>0].sort_values("pct_missing",ascending=False)
        out["cumulative_pct"] = out["pct_missing"].cumsum().round(2)
        return out
