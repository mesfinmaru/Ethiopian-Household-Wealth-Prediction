"""
data_preprocesor.py — CRISP-DM Phase 3: Data Preparation
sklearn ColumnTransformer pipeline: encoding + scaling + splitting.

Feature treatment:
  Numeric continuous → StandardScaler (after median impute)
  Ordinal (settlement, housing codes) → OrdinalEncoder
  Nominal (region, wave) → OneHotEncoder
  Binary → passthrough (already 0/1)
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")

from config import (
    ALL_FEATURES, CV_FOLDS, MODEL_DIR, RANDOM_STATE,
    TARGET, TEST_SIZE, VAL_SIZE,
)


class DataPreprocessor:
    """
    Build a scikit-learn preprocessing pipeline and train/val/test splits.

    Usage
    -----
        dp = DataPreprocessor()
        splits = dp.fit(df)
        # splits["X_train"], splits["X_val"], splits["X_test"]
        # splits["y_train"], splits["y_val"], splits["y_test"]
        # splits["feature_names"]

        X_new = dp.transform(new_df)
    """

    # ── Feature column groups ─────────────────────────────────────────────
    CONTINUOUS = [
        "hh_size", "adulteq", "head_age", "head_age_sq",
        "hh_n_workers", "hh_avg_weeks_worked",
        "enterprise_asset_count", "rooms", "housing_score",
        "n_shocks", "asset_count", "assets_per_member",
        "log_hh_size", "dependency_ratio",
    ]
    BINARY = [
        "head_sex", "is_female_headed", "head_literate",
        "hh_any_wage_earner", "has_nonfarm_enterprise",
        "has_electricity", "owns_phone", "owns_tv", "owns_fridge",
        "experienced_drought", "experienced_illness",
        "experienced_death", "experienced_crop_loss",
        "post_covid", "is_tigray_conflict",
        "has_full_housing", "has_enterprise_data",
    ]
    ORDINAL_HOUSING = ["roof", "wall", "floor", "water", "toilet", "fuel"]
    ORDINAL_SETTLEMENT = ["settlement"]
    NOMINAL = ["region", "wave"]

    def __init__(self) -> None:
        self.pipeline_       = None
        self.label_encoder_  = LabelEncoder()
        self.feature_names_  : list[str] = []
        self._fitted         = False

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        test_size: float  = TEST_SIZE,
        val_size: float   = VAL_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> dict:
        """
        Fit the preprocessing pipeline and build train/val/test splits.

        Parameters
        ----------
        df           : cleaned DataFrame from DataCleaner.fit_transform()
        test_size    : fraction held out as final test set
        val_size     : fraction of remaining data used as validation set
        random_state : reproducibility seed

        Returns
        -------
        dict with keys:
          X_train, X_val, X_test : processed numpy arrays
          y_train, y_val, y_test : int arrays (cons_quint 1–5)
          feature_names          : list of output feature names
          label_classes          : array [1,2,3,4,5]
        """
        df = df.dropna(subset=[TARGET]).copy()

        X_raw = self._select_features(df)
        y_raw = df[TARGET].values.astype(int)
        self.label_encoder_.fit(y_raw)

        # ── build pipeline ────────────────────────────────────────────────
        cont     = [c for c in self.CONTINUOUS       if c in X_raw.columns]
        binary   = [c for c in self.BINARY           if c in X_raw.columns]
        ord_h    = [c for c in self.ORDINAL_HOUSING  if c in X_raw.columns]
        ord_s    = [c for c in self.ORDINAL_SETTLEMENT if c in X_raw.columns]
        nominal  = [c for c in self.NOMINAL          if c in X_raw.columns]

        self.feature_names_ = cont + binary + ord_h + ord_s + nominal

        cont_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ])
        binary_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ])
        ord_h_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1)),
        ])
        ord_s_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder()),
        ])
        nominal_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False,
                                     drop="first")),
        ])

        self.pipeline_ = ColumnTransformer([
            ("cont",    cont_pipe,    cont),
            ("binary",  binary_pipe,  binary),
            ("ord_h",   ord_h_pipe,   ord_h),
            ("ord_s",   ord_s_pipe,   ord_s),
            ("nominal", nominal_pipe, nominal),
        ], remainder="drop")

        # ── split: train/val/test ─────────────────────────────────────────
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_raw, y_raw,
            test_size=test_size, stratify=y_raw, random_state=random_state,
        )
        val_adj = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adj, stratify=y_temp, random_state=random_state,
        )

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
            "feature_names": self._get_output_names(cont, binary, ord_h, ord_s, nominal),
            "label_classes": self.label_encoder_.classes_,
            "n_features":    X_train_p.shape[1],
            "split_sizes":   {
                "train": len(y_train),
                "val":   len(y_val),
                "test":  len(y_test),
            },
        }

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted pipeline."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.pipeline_.transform(self._select_features(df))

    def fit_region(
        self,
        df: pd.DataFrame,
        region: str,
        test_size: float  = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> dict | None:
        """
        Build a train/test split for one region only.
        Returns None if region has too few samples.
        """
        sub = df[df["region"].astype(str) == region].copy()
        if len(sub) < 50:
            return None

        if not self._fitted:
            self.fit(df)   # fit on full data first

        X_raw = self._select_features(sub)
        y     = sub[TARGET].values.astype(int)
        strat = y if pd.Series(y).value_counts().min() >= 2 else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y, test_size=test_size,
            stratify=strat, random_state=random_state,
        )
        return {
            "X_train":  self.pipeline_.transform(X_tr),
            "X_test":   self.pipeline_.transform(X_te),
            "y_train":  y_tr,
            "y_test":   y_te,
            "n_total":  len(sub),
            "region":   region,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.feature_names_ if c in df.columns]
        missing = [c for c in self.feature_names_ if c not in df.columns]
        result = df[cols].copy()
        for c in missing:
            result[c] = np.nan
        return result[self.feature_names_]

    def _get_output_names(self, cont, binary, ord_h, ord_s, nominal) -> list[str]:
        names = cont + binary + ord_h + ord_s
        try:
            ohe_names = (
                self.pipeline_
                .named_transformers_["nominal"]
                .named_steps["encode"]
                .get_feature_names_out(nominal)
                .tolist()
            )
        except Exception:
            ohe_names = nominal
        return names + ohe_names

    def save(self, path: Path = MODEL_DIR / "preprocessor.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            (self.pipeline_, self.label_encoder_, self.feature_names_),
            path,
        )

    def load(self, path: Path = MODEL_DIR / "preprocessor.pkl") -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        self.pipeline_, self.label_encoder_, self.feature_names_ = joblib.load(p)
        self._fitted = True