"""
missing_value_handler.py
═══════════════════════════════════════════════════════════════════════════════
MissingValueHandler — class-reference-compatible imputation for ESS data.

CRISP-DM Phase 2 (Data Preparation): Handles missing values using multiple
strategies appropriate for multi-wave survey data.

Class reference interface (from Chapter 2 Data Preparation):
  mean_impute, median_impute, mode_impute, constant_impute,
  knn_impute, iterative_impute, forward_fill, backward_fill,
  interpolate_impute, impute (dispatcher), transform

ESS-specific extensions:
  group_impute      — within-group (region × wave) medians
  flag_and_fill     — MNAR indicator + constant fill
  wave_coverage_fill— zero-fill structurally absent W4/W5 features
  handle_w2_gaps    — cross-wave donor fill for W2-sparse columns
  ess_pipeline      — recommended full pipeline for ESS data

W2 missing column strategy (run AFTER mixing all waves):
  ─────────────────────────────────────────────────────────
  REASON TO IMPUTE AFTER MIXING:
    W2 has structurally sparse columns because SPSS truncates variable names
    to 8 characters. The rename fixes in sav_reader/config resolve MOST gaps.
    For any remaining NaN in W2, using W1/W3 as "donor waves" (temporally
    adjacent, same survey design) gives more accurate group medians than
    imputing from W2 alone.

  hh_avg_weeks_worked : fill W2 NaN with 0 (not employed = 0 weeks worked)
  All continuous cols : group_impute using W1/W3 donors via region×settlement
═══════════════════════════════════════════════════════════════════════════════
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

warnings.filterwarnings("ignore")


class MissingValueHandler:
    """
    Multi-strategy missing value handler for the ESS multi-wave dataset.

    Attributes
    ----------
    imputer : fitted sklearn imputer (set after calling impute())
    method  : name of the last method applied
    log_    : list of dicts recording each imputation step (for audit)
    """

    def __init__(self):
        self.imputer = None
        self.method  = None
        self.log_    = []

    # ── Class-reference interface methods ─────────────────────────────────────

    def mean_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with column means (sensitive to outliers)."""
        self.imputer = SimpleImputer(strategy="mean")
        self.method  = "mean"
        return pd.DataFrame(self.imputer.fit_transform(X),
                            columns=X.columns, index=X.index)

    def median_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill with column medians — robust to outliers (Chapter 2 recommended)."""
        self.imputer = SimpleImputer(strategy="median")
        self.method  = "median"
        return pd.DataFrame(self.imputer.fit_transform(X),
                            columns=X.columns, index=X.index)

    def mode_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill with most frequent value — suitable for ordinal/categorical."""
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.method  = "mode"
        return pd.DataFrame(self.imputer.fit_transform(X),
                            columns=X.columns, index=X.index)

    def constant_impute(self, X: pd.DataFrame, fill_value=0) -> pd.DataFrame:
        """Fill with a constant — used for MNAR features (e.g. absent assets=0)."""
        self.imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        self.method  = "constant"
        return pd.DataFrame(self.imputer.fit_transform(X),
                            columns=X.columns, index=X.index)

    def knn_impute(self, X: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        K-Nearest Neighbours imputation — preserves local data structure.
        More accurate than mean/median for spatially clustered survey data.
        Chapter 2 reference: KNNImputer from sklearn.impute.
        """
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.method  = "knn"
        return pd.DataFrame(self.imputer.fit_transform(X),
                            columns=X.columns, index=X.index)

    def iterative_impute(self, X: pd.DataFrame,
                         max_iter: int = 10,
                         random_state: int = 42) -> pd.DataFrame:
        """
        MICE iterative imputation using RandomForest estimator.
        Most accurate but slowest — use for key continuous features.
        Chapter 2 reference: IterativeImputer (enable_iterative_imputer).
        """
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50,
                                            random_state=random_state),
            max_iter=max_iter, random_state=random_state,
        )
        self.method = "iterative"
        return pd.DataFrame(self.imputer.fit_transform(X),
                            columns=X.columns, index=X.index)

    def forward_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward fill — useful for panel data sorted by household and wave."""
        self.method = "ffill"
        return X.ffill()

    def backward_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        """Backward fill — complement to forward_fill for panel data."""
        self.method = "bfill"
        return X.bfill()

    def interpolate_impute(self, X: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        """Interpolation — suitable for temporal trend features across waves."""
        self.method = f"interpolate_{method}"
        return X.interpolate(method=method)

    def impute(self, X: pd.DataFrame, method: str = "median", **kwargs) -> pd.DataFrame:
        """
        Dispatcher: apply the chosen imputation strategy.

        Parameters
        ----------
        X      : DataFrame (numeric columns)
        method : one of 'mean','median','mode','constant','knn',
                 'iterative','ffill','bfill','interpolate'
        **kwargs : method-specific args (n_neighbors, fill_value, max_iter, etc.)
        """
        _map = {
            "mean":        lambda x: self.mean_impute(x),
            "median":      lambda x: self.median_impute(x),
            "mode":        lambda x: self.mode_impute(x),
            "constant":    lambda x: self.constant_impute(x, kwargs.get("fill_value", 0)),
            "knn":         lambda x: self.knn_impute(x, kwargs.get("n_neighbors", 5)),
            "iterative":   lambda x: self.iterative_impute(
                               x, kwargs.get("max_iter", 10),
                               kwargs.get("random_state", 42)),
            "ffill":       lambda x: self.forward_fill(x),
            "bfill":       lambda x: self.backward_fill(x),
            "interpolate": lambda x: self.interpolate_impute(
                               x, kwargs.get("interpolation_method", "linear")),
        }
        if method not in _map:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(_map)}")
        return _map[method](X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the FITTED imputer to new data (e.g. test set).
        Ensures test data uses training-set statistics — prevents leakage.
        Chapter 2 principle: fit on train only, transform both.
        """
        if self.imputer is None:
            raise ValueError("No imputer fitted. Call impute() first.")
        if self.method in ("ffill", "bfill"):
            return self.forward_fill(X) if self.method == "ffill" else self.backward_fill(X)
        if self.method and self.method.startswith("interpolate"):
            m = self.method.split("_", 1)[1] if "_" in self.method else "linear"
            return X.interpolate(method=m)
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)

    # ── ESS-specific survey-aware methods ─────────────────────────────────────

    def group_impute(self, df: pd.DataFrame, col: str,
                     group_cols: list, fallback: str = "median") -> pd.DataFrame:
        """
        Impute one column using within-group medians (survey-aware strategy).

        Households in the same region/settlement share socioeconomic context —
        group medians are more accurate than global medians for survey data.

        Fallback chain: group median → wave median → global median.

        Parameters
        ----------
        df         : full merged DataFrame (must contain group_cols)
        col        : column to impute
        group_cols : e.g. ["wave","region"] or ["wave","region","settlement"]
        fallback   : 'median' or 'mean'
        """
        df     = df.copy()
        before = df[col].isnull().sum()
        if before == 0:
            return df

        grp = df.groupby(group_cols, observed=True)[col].transform("median")
        df[col] = df[col].fillna(grp)

        if df[col].isnull().any() and "wave" in df.columns:
            wave_med = df.groupby("wave")[col].transform("median")
            df[col]  = df[col].fillna(wave_med)

        if df[col].isnull().any():
            val = df[col].median() if fallback == "median" else df[col].mean()
            df[col] = df[col].fillna(val)

        after = df[col].isnull().sum()
        self.log_.append({"step": "group_impute", "col": col,
                          "filled": int(before - after), "remaining": int(after)})
        return df

    def flag_and_fill(self, df: pd.DataFrame, cols: list,
                      fill_value=0) -> pd.DataFrame:
        """
        Add binary missingness indicator then fill with constant.

        Used for MNAR (Missing Not At Random) features where absence is
        informative. Adds '<col>_was_missing' before filling.

        Example: has_electricity=NaN in W4/W5 → likely no electricity.
        """
        df = df.copy()
        for col in [c for c in cols if c in df.columns]:
            if df[col].isnull().any():
                df[f"{col}_was_missing"] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(fill_value)
                self.log_.append({"step": "flag_and_fill", "col": col,
                                  "fill_value": fill_value})
        return df

    def wave_coverage_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zero-fill features structurally absent due to survey wave design.

        W4/W5 have no enterprise section → enterprise columns NaN = not collected.
        W1/W2/W3 have no electricity question in sect8 (inferred from lighting).
        Absence = question not asked (MNAR) → fill with 0, not median.
        """
        df        = df.copy()
        zero_cols = [
            "owns_phone","owns_tv","owns_fridge","has_electricity",
            "hh_any_wage_earner","hh_n_workers",
            "has_nonfarm_enterprise","enterprise_asset_count",
            "experienced_drought","experienced_illness",
            "experienced_death","experienced_crop_loss","n_shocks",
        ]
        for col in [c for c in zero_cols if c in df.columns]:
            n = df[col].isnull().sum()
            if n > 0:
                df[col] = df[col].fillna(0)
                self.log_.append({"step": "wave_coverage_fill", "col": col,
                                  "filled": int(n)})
        return df

    def handle_w2_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle features with elevated missing rates specifically in Wave 2.

        Strategy (run AFTER mixing all 5 waves):
          1. hh_avg_weeks_worked → 0  (NaN in W2 means not employed)
          2. All other continuous W2 NaNs → group medians from W1/W3 donors
             using region × settlement grouping
          3. Fallback: global W1/W3 median

        Why after mixing: W1/W3 (temporally adjacent, same survey design)
        provide statistically appropriate donor data for filling W2 gaps.
        """
        df         = df.copy()
        w2_mask    = df["wave"] == 2
        donor_mask = df["wave"].isin([1, 3])
        group_cols = ["region", "settlement"]

        # Step 1: weeks worked → 0 for W2 (NaN = not working for pay)
        if "hh_avg_weeks_worked" in df.columns:
            mask  = w2_mask & df["hh_avg_weeks_worked"].isnull()
            df.loc[mask, "hh_avg_weeks_worked"] = 0
            self.log_.append({"step": "w2_weeks_fill", "filled": int(mask.sum())})

        # Step 2: continuous W2 gaps → W1/W3 group medians
        cont_cols = ["head_age","head_age_sq","head_edu_level",
                     "rooms","housing_score","hh_n_workers"]
        for col in [c for c in cont_cols
                    if c in df.columns and df.loc[w2_mask, c].isnull().any()]:
            donors   = df.loc[donor_mask, group_cols + [col]].copy()
            avail_gc = [g for g in group_cols if g in donors.columns]
            grp_med  = (donors.groupby(avail_gc, observed=True)[col]
                               .median().reset_index().rename(columns={col:"_fill"}))
            tmp = df.loc[w2_mask].merge(grp_med, on=avail_gc, how="left")
            before = int(df.loc[w2_mask, col].isnull().sum())
            nan_idx = df.index[w2_mask & df[col].isnull()]
            fill_vals = tmp.loc[tmp.index.isin(nan_idx), "_fill"].values
            df.loc[nan_idx[:len(fill_vals)], col] = fill_vals[:before]

            # Fallback: global W1/W3 median
            still_nan = w2_mask & df[col].isnull()
            if still_nan.any():
                fb = df.loc[donor_mask, col].median()
                df.loc[still_nan, col] = fb

            after = int(df.loc[w2_mask, col].isnull().sum())
            self.log_.append({"step":"w2_cross_wave_fill","col":col,
                              "filled":before-after,"remaining":after})
        return df

    def ess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recommended full missing-value pipeline for the ESS dataset.

        CRISP-DM Phase 2 pipeline order:
        ──────────────────────────────────────────────────────
        Step 1: wave_coverage_fill  → zero-fill structural absences (MNAR)
        Step 2: flag_and_fill       → MNAR indicators before filling
        Step 3: handle_w2_gaps      → cross-wave donor fill for W2
        Step 4: group_impute        → wave×region medians for continuous
        Step 5: mode fill           → settlement-level mode for housing ordinals
        Step 6: knn_impute          → KNN for remaining continuous NaN
        Step 7: global median       → final fallback
        ──────────────────────────────────────────────────────
        Each step is logged for audit via imputation_log().
        """
        df = df.copy()

        # Step 1: structural zero-fills (wave design absences)
        df = self.wave_coverage_fill(df)

        # Step 2: MNAR flag + fill
        mnar_cols = ["has_electricity","enterprise_asset_count","has_nonfarm_enterprise"]
        df = self.flag_and_fill(df, mnar_cols, fill_value=0)

        # Step 3: W2-specific cross-wave gap filling
        df = self.handle_w2_gaps(df)

        # Step 4: continuous demographics → wave × region group median
        for col in ["head_age","head_edu_level","rooms","housing_score"]:
            if col in df.columns and df[col].isnull().any():
                df = self.group_impute(df, col, group_cols=["wave","region"])

        # Step 5: housing ordinals → settlement-level mode
        for col in ["roof","wall","floor","water","toilet","fuel"]:
            if col in df.columns and df[col].isnull().any():
                if "settlement" in df.columns:
                    mode_fill = df.groupby("settlement", observed=True)[col].transform(
                        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                    )
                    df[col] = df[col].fillna(mode_fill)
                df[col] = df[col].fillna(df[col].median())

        # Step 6: KNN for remaining continuous NaN
        knn_targets = [c for c in ["head_age","head_age_sq","head_edu_level",
                                   "rooms","housing_score","housing_quality_idx",
                                   "adults_ratio","dependency_ratio"]
                       if c in df.columns and df[c].isnull().any()]
        if knn_targets:
            filled = self.knn_impute(df[knn_targets].copy(), n_neighbors=5)
            df[knn_targets] = filled.values

        # Step 7: global median fallback for any remaining numeric NaN
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        return df

    # ── Reporting ──────────────────────────────────────────────────────────────

    def missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary of missing values per column with recommended strategy.
        Columns: feature, n_missing, pct_missing, dtype, recommended_strategy.
        """
        rows = []
        for col in df.columns:
            n = df[col].isnull().sum()
            if n == 0:
                continue
            pct   = round(n / len(df) * 100, 2)
            dtype = str(df[col].dtype)
            uniq  = df[col].nunique()

            if uniq <= 2 or col in ("has_electricity","owns_phone","owns_tv",
                                     "owns_fridge","has_nonfarm_enterprise"):
                strategy = "flag_and_fill(0) [MNAR]"
            elif col in ("roof","wall","floor","water","toilet","fuel","settlement"):
                strategy = "mode by settlement"
            elif col in ("head_age","rooms","housing_score","head_edu_level"):
                strategy = "group_impute(wave × region)"
            elif col == "hh_avg_weeks_worked":
                strategy = "0 for W2 [not working]; group_impute others"
            elif dtype in ("object","category"):
                strategy = "mode"
            else:
                strategy = "knn or median"

            rows.append({"feature":col, "n_missing":n, "pct_missing":pct,
                         "dtype":dtype, "recommended_strategy":strategy})

        return (pd.DataFrame(rows)
                .sort_values("pct_missing", ascending=False)
                .reset_index(drop=True))

    def imputation_log(self) -> pd.DataFrame:
        """Return DataFrame log of all imputation steps performed."""
        return pd.DataFrame(self.log_) if self.log_ else pd.DataFrame()
