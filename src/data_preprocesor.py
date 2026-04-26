"""
data_preprocesor.py
Ethiopian Household Wealth Prediction - Multi-output prediction: wealth quintile, settlement type, region, zone
Regional comparison and national ranking.

"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)


# ════════════════════════════════════════════════════════════════════════════
# DataPreprocessor 
# ════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:
    """
    Prepares the harmonised multi-wave Ethiopian ESS dataset for modelling.

    Targets
    -------
    cons_quint  : 1-5 wealth quintile (primary)
    settlement  : 0=urban, 1=rural, 2=small town, 3=large town
    region      : one of 11 Ethiopian regions
    zone_label  : region + zone code (finer geography)

    Usage
    -----
        dp = DataPreprocessor()
        df = dp.load_data("data/processed/all_waves_clean.csv")
        dp.describe_data(df)
        splits = dp.prepare(df)
        # splits["overall"] → (X_train, X_test, y_train, y_test)
        # splits["TIGRAY"]  → per-region split
    """

    # ── Feature groups 
    NUMERIC = [
        "hh_size", "adulteq",
        "food_cons_ann", "nonfood_cons_ann", "educ_cons_ann",
        "total_cons_ann", "nom_totcons_aeq",
        "food_share", "nonfood_share", "educ_share",
        "cons_per_cap", "cons_per_adulteq",
        "log_totcons", "log_cons_cap", "log_hh_size",
        "has_educ_spend",
    ]
    OPTIONAL_NUMERIC = ["fafh_cons_ann", "utilities_cons_ann", "spat_totcons_aeq"]
    ORDINAL = ["settlement"]   # 0,1,2,3 — ordered
    NOMINAL = ["wave"]         # one-hot: wave is nominal, not ordinal for ML

    TARGETS = ["cons_quint", "settlement", "region", "zone_label"]
    PRIMARY_TARGET = "cons_quint"

    def __init__(self):
        self.pipeline_       = None   # sklearn ColumnTransformer
        self.label_encoders_ = {}     # one LabelEncoder per categorical target
        self.feature_names_  = None

    # ── load_data  
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the cleaned CSV produced by data_cleaning.py."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        df = pd.read_csv(filepath)
        # Restore category dtypes
        for col in ["region", "zone_label"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        print(f"Loaded {len(df):,} rows × {df.shape[1]} cols from {filepath}")
        return df

    # ── describe_data
    def describe_data(self, df: pd.DataFrame) -> None:
        """Quick EDA summary."""
        print(f"\n{'='*55}")
        print(f"  Dataset overview")
        print(f"{'='*55}")
        print(f"  Shape         : {df.shape}")
        print(f"  Waves         : {sorted(df['wave'].unique())}")
        print(f"  Regions       : {df['region'].nunique()} unique")
        print(f"  Zones         : {df['zone_label'].nunique()} unique")
        print(f"\n  cons_quint (target):")
        ct = df["cons_quint"].value_counts().sort_index()
        for q, n in ct.items():
            bar = "█" * (n // 200)
            print(f"    Q{q}: {n:5,}  {bar}")
        print(f"\n  settlement:")
        labels = {0: "urban", 1: "rural", 2: "small town", 3: "large town"}
        st = df["settlement"].value_counts().sort_index()
        for s, n in st.items():
            print(f"    {s} ({labels.get(s,'?')}): {n:,}")
        print(f"\n  Region distribution:")
        rd = df.groupby("region")["cons_quint"].agg(["count", "mean"]).round(2)
        rd.columns = ["n_households", "mean_quintile"]
        rd = rd.sort_values("mean_quintile", ascending=False)
        print(rd.to_string())
        print(f"\n  Missing values:")
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        print(miss.to_string() if not miss.empty else "    None")
        print(f"{'='*55}\n")

    # ── Internal: build feature matrix ───────────────────────────────────
    def _build_feature_matrix(self, df: pd.DataFrame, use_optional: bool = True):
        num = [c for c in self.NUMERIC if c in df.columns]
        if use_optional:
            num += [c for c in self.OPTIONAL_NUMERIC if c in df.columns]
        ord_feats = [c for c in self.ORDINAL  if c in df.columns]
        nom_feats = [c for c in self.NOMINAL  if c in df.columns]

        self.feature_names_ = num + ord_feats + nom_feats

        numeric_pipe  = Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        ordinal_pipe  = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(categories="auto")),
        ])
        nominal_pipe  = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.pipeline_ = ColumnTransformer([
            ("num", numeric_pipe, num),
            ("ord", ordinal_pipe, ord_feats),
            ("nom", nominal_pipe, nom_feats),
        ])
        return df[num + ord_feats + nom_feats]

    # ── encode categorical targets ────────────────────────────────────────
    def _encode_targets(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        encoded = {}
        for t in self.TARGETS:
            if t not in df.columns:
                continue
            if df[t].dtype.name in ("category", "object"):
                le = LabelEncoder()
                encoded[t] = le.fit_transform(df[t].astype(str))
                self.label_encoders_[t] = le
            else:
                encoded[t] = df[t].values.astype(int)
        return encoded

    # ── prepare (main entry point) ────────────────────────────────────────
    def prepare(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        use_optional: bool = True,
    ) -> dict:
        """
        Build train/test splits for:
          - 'overall'   : full dataset, multi-output (all 4 targets)
          - one key per region: single-region, primary target only

        Returns
        -------
        dict with keys 'overall' and each region name.
        Each value is a dict:
          {
            'X_train', 'X_test',
            'y_train', 'y_test',        ← primary target (cons_quint)
            'Y_train', 'Y_test',        ← multi-output (all targets, overall only)
          }
        """
        df = df.dropna(subset=[self.PRIMARY_TARGET]).copy()

        # ── Overall split ──────────────────────────────────────────────
        X_raw = self._build_feature_matrix(df, use_optional)
        targets = self._encode_targets(df)

        X_tr_raw, X_te_raw, idx_tr, idx_te = train_test_split(
            X_raw, df.index,
            test_size=test_size,
            stratify=targets[self.PRIMARY_TARGET],
            random_state=random_state,
        )

        X_train = self.pipeline_.fit_transform(X_tr_raw)
        X_test  = self.pipeline_.transform(X_te_raw)

        y_primary = targets[self.PRIMARY_TARGET]
        y_tr = y_primary[df.index.get_indexer(idx_tr)]
        y_te = y_primary[df.index.get_indexer(idx_te)]

        # Multi-output matrix (all 4 targets stacked)
        Y_all = np.column_stack([targets[t] for t in self.TARGETS if t in targets])
        Y_tr  = Y_all[df.index.get_indexer(idx_tr)]
        Y_te  = Y_all[df.index.get_indexer(idx_te)]

        splits = {
            "overall": {
                "X_train": X_train, "X_test": X_test,
                "y_train": y_tr,    "y_test": y_te,
                "Y_train": Y_tr,    "Y_test": Y_te,
                "target_names": [t for t in self.TARGETS if t in targets],
            }
        }

        print(f"Overall — train: {len(y_tr):,}  test: {len(y_te):,} "
              f"| features: {X_train.shape[1]}")

        # ── Per-region splits ──────────────────────────────────────────
        for region in sorted(df["region"].astype(str).unique()):
            mask = df["region"].astype(str) == region
            sub  = df[mask].reset_index(drop=True)
            if len(sub) < 50:
                continue  # too small to split reliably

            X_sub = self._build_feature_matrix(sub, use_optional)
            sub_targets = sub[self.PRIMARY_TARGET].values.astype(int)

            # Check we have enough per class for stratified split
            min_class = pd.Series(sub_targets).value_counts().min()
            strat = sub_targets if min_class >= 2 else None

            X_sub_tr, X_sub_te, y_sub_tr, y_sub_te = train_test_split(
                X_sub, sub_targets,
                test_size=test_size,
                stratify=strat,
                random_state=random_state,
            )
            X_sub_tr = self.pipeline_.transform(X_sub_tr)
            X_sub_te = self.pipeline_.transform(X_sub_te)

            splits[region] = {
                "X_train": X_sub_tr, "X_test": X_sub_te,
                "y_train": y_sub_tr, "y_test": y_sub_te,
                "n_total": len(sub),
            }
            print(f"  {region:<22}: {len(sub_targets):,} households "
                  f"(train {len(y_sub_tr)} / test {len(y_sub_te)})")

        return splits

    def save_pipeline(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump((self.pipeline_, self.label_encoders_, self.feature_names_), path)
        print(f"Pipeline saved → {path}")

    def load_pipeline(self, path: str = "models/preprocessor.pkl"):
        self.pipeline_, self.label_encoders_, self.feature_names_ = joblib.load(path)
        print(f"Pipeline loaded from {path}")

