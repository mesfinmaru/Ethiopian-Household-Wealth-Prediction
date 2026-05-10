"""
data_preprocessor.py
═══════════════════════════════════════════════════════════════════════════════
DataPreprocessor — sklearn ColumnTransformer pipeline + stratified splits.
CRISP-DM Phase 3: Data Preparation (Encoding + Scaling + Splitting)

Feature treatment (Chapter 2 reference):
  Continuous  → median impute → StandardScaler
  Binary      → constant(0) impute → passthrough
  Ordinal     → most_frequent impute → OrdinalEncoder
  Nominal     → most_frequent impute → OneHotEncoder (drop first)

Stratified splits: train(65%) / val(15%) / test(20%)
Stratification on cons_quint preserves quintile distribution across splits.
═══════════════════════════════════════════════════════════════════════════════
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                   OneHotEncoder, StandardScaler)

from config import MODEL_DIR, RANDOM_STATE, TARGET, TEST_SIZE, VAL_SIZE, MIN_REGION_N


class DataPreprocessor:
    """
    Prepare the cleaned + feature-engineered ESS dataset for machine learning.

    Usage
    -----
        dp     = DataPreprocessor()
        splits = dp.fit(df)

        # splits keys:
        #   X_train, X_val, X_test  : processed numpy arrays
        #   y_train, y_val, y_test  : int arrays (cons_quint 1–5)
        #   feature_names           : output feature names (post-OHE)
        #   split_sizes             : {'train':N, 'val':N, 'test':N}

        X_new = dp.transform(new_df)  # inference on new data
        dp.save()                      # persist to models/
        dp.load()                      # reload for inference
    """

    # ── Feature column groups (Chapter 2: feature type classification) ─────────
    CONTINUOUS = [
        "hh_size","adulteq","head_age","head_age_sq",
        "hh_n_workers","hh_avg_weeks_worked",
        "enterprise_asset_count","rooms","housing_score",
        "assets_per_member","log_hh_size","dependency_ratio",
        "adults_ratio","roof_quality","floor_quality",
        "housing_quality_idx","modern_asset_score","labour_intensity",
        "n_shocks","shock_breadth",
    ]
    BINARY = [
        "head_sex","is_female_headed","head_literate",
        "hh_any_wage_earner","has_nonfarm_enterprise",
        "has_electricity","owns_phone","owns_tv","owns_fridge",
        "post_covid","is_tigray_conflict",
        "has_full_housing","has_enterprise_data",
        "has_electricity_was_missing","has_nonfarm_enterprise_was_missing",
        "is_large_hh","is_single_person","improved_water",
        "improved_sanitation","clean_fuel","has_any_modern_asset",
        "is_fully_dependent","is_multi_shock","is_urban","is_addis",
        "is_peripheral","head_prime_working_age","head_elderly",
        "educated_prime_head","urban_conflict",
    ]
    ORDINAL  = ["roof","wall","floor","water","toilet","fuel","settlement",
                "head_edu_level"]
    NOMINAL  = ["region","wave"]
    TARGET   = "cons_quint"

    def __init__(self, random_state: int = RANDOM_STATE):
        self.rs             = random_state
        self.pipeline_      = None
        self.label_encoder_ = LabelEncoder()
        self.feature_names_ = []
        self._fitted        = False

    def fit(self, df: pd.DataFrame,
            test_size: float = TEST_SIZE,
            val_size: float  = VAL_SIZE) -> dict:
        """
        Fit the ColumnTransformer pipeline and create stratified splits.

        Parameters
        ----------
        df        : cleaned + feature-engineered DataFrame
        test_size : fraction for final held-out test set (default 0.20)
        val_size  : fraction of remaining data for validation (default 0.15)

        Returns
        -------
        dict : X_train/val/test, y_train/val/test, feature_names,
               label_classes, n_features, split_sizes
        """
        df = df.dropna(subset=[self.TARGET]).copy()

        cont = [c for c in self.CONTINUOUS if c in df.columns]
        bin_ = [c for c in self.BINARY    if c in df.columns]
        ord_ = [c for c in self.ORDINAL   if c in df.columns]
        nom  = [c for c in self.NOMINAL   if c in df.columns]
        self.feature_names_ = cont + bin_ + ord_ + nom

        self.pipeline_ = ColumnTransformer([
            ("cont", Pipeline([
                ("imp",   SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), cont),
            ("bin", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value=0)),
            ]), bin_),
            ("ord", Pipeline([
                ("imp",    SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)),
            ]), ord_),
            ("nom", Pipeline([
                ("imp",    SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False, drop="first")),
            ]), nom),
        ], remainder="drop")

        X = df[self.feature_names_]
        y = df[self.TARGET].values.astype(int)
        self.label_encoder_.fit(y)

        # Stratified train / (val+test) split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.rs)
        val_adj = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adj,
            stratify=y_temp, random_state=self.rs)

        X_train_p = self.pipeline_.fit_transform(X_train)
        X_val_p   = self.pipeline_.transform(X_val)
        X_test_p  = self.pipeline_.transform(X_test)

        self._fitted = True

        return {
            "X_train":       X_train_p,
            "X_val":         X_val_p,
            "X_test":        X_test_p,
            "y_train":       y_train,
            "y_val":         y_val,
            "y_test":        y_test,
            "feature_names": self._get_output_names(cont, bin_, ord_, nom),
            "label_classes": self.label_encoder_.classes_,
            "n_features":    X_train_p.shape[1],
            "split_sizes":   {"train":len(y_train),"val":len(y_val),"test":len(y_test)},
        }

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted pipeline to new data (inference)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = df[[c for c in self.feature_names_ if c in df.columns]]
        return self.pipeline_.transform(X)

    def fit_region(self, df: pd.DataFrame, region: str,
                   test_size: float = TEST_SIZE) -> dict:
        """
        Build train/test split for a single region.
        Returns None if region has fewer than MIN_REGION_N households.
        Used for per-region modelling and regional wealth comparison.
        """
        sub = df[df["region"].astype(str) == region].copy()
        if len(sub) < MIN_REGION_N:
            return None
        if not self._fitted:
            self.fit(df)
        X = sub[[c for c in self.feature_names_ if c in sub.columns]]
        y = sub[self.TARGET].values.astype(int)
        strat = y if pd.Series(y).value_counts().min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=self.rs)
        return {"X_train": self.pipeline_.transform(X_tr),
                "X_test":  self.pipeline_.transform(X_te),
                "y_train": y_tr, "y_test": y_te,
                "n_total": len(sub), "region": region}

    def feature_group_summary(self) -> pd.DataFrame:
        """Return feature group counts table for notebook display."""
        groups = {"Continuous": self.CONTINUOUS, "Binary": self.BINARY,
                  "Ordinal":    self.ORDINAL,     "Nominal": self.NOMINAL}
        return pd.DataFrame([
            {"group": g, "n_defined": len(cols),
             "sample_features": ", ".join(cols[:5]) + ("…" if len(cols) > 5 else "")}
            for g, cols in groups.items()
        ])

    def _get_output_names(self, cont, bin_, ord_, nom) -> list:
        names = cont + bin_ + ord_
        try:
            ohe   = self.pipeline_.named_transformers_["nom"].named_steps["encode"]
            names += ohe.get_feature_names_out(nom).tolist()
        except Exception:
            names += nom
        return names

    def save(self, path: str = None):
        """Persist fitted pipeline + label encoder to models/ directory."""
        from pathlib import Path
        dest = Path(path) if path else MODEL_DIR / "preprocessor.pkl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.pipeline_, self.label_encoder_, self.feature_names_), dest)
        print(f"Preprocessor saved → {dest}")

    def load(self, path: str = None):
        """Load a previously saved pipeline."""
        from pathlib import Path
        src = Path(path) if path else MODEL_DIR / "preprocessor.pkl"
        if not src.exists():
            raise FileNotFoundError(src)
        self.pipeline_, self.label_encoder_, self.feature_names_ = joblib.load(src)
        self._fitted = True
