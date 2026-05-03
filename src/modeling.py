"""
modeling.py — WealthPredictor
Multi-class classification: cons_quint 1 (poorest) → 5 (wealthiest).
Uses legitimate proxy features only — NEVER consumption aggregates.

Models: RandomForest, XGBoost, LightGBM, LogisticRegression.
Modes:  overall Ethiopia | per-region | regional ranking | pairwise comparison.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from config import CV_FOLDS, MIN_REGION_N, MODEL_DIR, RANDOM_STATE, TARGET, TEST_SIZE


class WealthPredictor:
    """
    Train and evaluate wealth quintile classifiers.

    Usage
    -----
        wp      = WealthPredictor()
        results = wp.train_evaluate(X_train, y_train, X_test, y_test)
        reg_df  = wp.train_per_region(df, feature_cols)
        ranking = wp.regional_ranking(reg_df)
        comp    = wp.compare_regions("TIGRAY", "OROMIA")
        imp     = wp.feature_importance(feature_names)
        wp.save()
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.rs              = random_state
        self.models_         : dict = {}
        self.results_        : pd.DataFrame | None = None
        self.best_model_     = None
        self.best_name_      : str = ""
        self.region_models_  : dict = {}

    # ── Model registry ────────────────────────────────────────────────────

    def _build_models(self) -> dict:
        m = {
            "Logistic Regression": LogisticRegression(
                max_iter=1_000, solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
                random_state=self.rs,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.rs, n_jobs=-1,
            ),
        }
        if _HAS_XGB:
            m["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=self.rs, verbosity=0,
            )
        if _HAS_LGB:
            m["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced",
                random_state=self.rs, verbose=-1,
            )
        return m

    # ════════════════════════════════════════════════════════════════════
    # 1. Train & evaluate overall model
    # ════════════════════════════════════════════════════════════════════

    def train_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
        cv_folds: int = CV_FOLDS,
    ) -> pd.DataFrame:
        """
        Train all models, evaluate on test set, cross-validate on train set.

        Returns
        -------
        pd.DataFrame sorted by weighted_f1 descending, columns:
          model, accuracy, weighted_f1, macro_f1, cv_acc_mean, cv_acc_std
        """
        rows = []
        cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.rs)

        for name, model in self._build_models().items():
            model.fit(X_train, y_train)
            y_pred    = model.predict(X_test)
            acc       = accuracy_score(y_test, y_pred)
            w_f1      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            m_f1      = f1_score(y_test, y_pred, average="macro",    zero_division=0)
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv, scoring="f1_weighted", n_jobs=-1)
            self.models_[name] = model
            rows.append({
                "model":       name,
                "accuracy":    round(acc,              4),
                "weighted_f1": round(w_f1,             4),
                "macro_f1":    round(m_f1,             4),
                "cv_f1_mean":  round(cv_scores.mean(), 4),
                "cv_f1_std":   round(cv_scores.std(),  4),
            })

        self.results_ = (
            pd.DataFrame(rows)
            .sort_values("weighted_f1", ascending=False)
            .reset_index(drop=True)
        )
        self.best_name_  = self.results_.iloc[0]["model"]
        self.best_model_ = self.models_[self.best_name_]
        return self.results_

    def classification_report_df(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """sklearn classification report as a DataFrame."""
        y_pred = self.best_model_.predict(X_test)
        rep = classification_report(
            y_test, y_pred,
            target_names=[f"Q{i} ({'poorest' if i==1 else 'wealthiest' if i==5 else ''})" for i in range(1,6)],
            output_dict=True, zero_division=0,
        )
        return pd.DataFrame(rep).T.round(4)

    def confusion_matrix_df(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """Confusion matrix as labelled DataFrame."""
        y_pred  = self.best_model_.predict(X_test)
        labels  = [f"Q{i}" for i in range(1, 6)]
        cm      = confusion_matrix(y_test, y_pred)
        return pd.DataFrame(cm, index=labels, columns=labels)

    # ════════════════════════════════════════════════════════════════════
    # 2. Hyperparameter tuning
    # ════════════════════════════════════════════════════════════════════

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "XGBoost",
        cv_folds: int = 3,
    ) -> pd.DataFrame:
        """
        GridSearchCV for specified model.
        Returns best params + CV results as a DataFrame.
        """
        grids = {
            "XGBoost":    {"n_estimators":[200,400], "max_depth":[4,6,8],
                           "learning_rate":[0.03,0.05,0.1]},
            "LightGBM":   {"n_estimators":[200,400], "max_depth":[4,6],
                           "learning_rate":[0.05,0.1]},
            "Random Forest": {"n_estimators":[200,400], "max_depth":[8,12,16],
                              "min_samples_leaf":[3,5,10]},
        }
        base = self._build_models()
        if model_name not in base:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Available: {list(base.keys())}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.rs)
        gs = GridSearchCV(base[model_name], grids.get(model_name, {}),
                          cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=0,
                          return_train_score=True)
        gs.fit(X_train, y_train)
        self.models_[model_name] = gs.best_estimator_
        if model_name == self.best_name_:
            self.best_model_ = gs.best_estimator_

        results_df = pd.DataFrame(gs.cv_results_)[
            ["params","mean_test_score","std_test_score","rank_test_score"]
        ].sort_values("rank_test_score")
        results_df.attrs["best_params"]    = gs.best_params_
        results_df.attrs["best_cv_score"]  = round(gs.best_score_, 4)
        return results_df

    # ════════════════════════════════════════════════════════════════════
    # 3. Per-region models
    # ════════════════════════════════════════════════════════════════════

    def train_per_region(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        test_size: float = TEST_SIZE,
    ) -> pd.DataFrame:
        """
        Train one LightGBM (or RF fallback) per region on cons_quint.

        Returns
        -------
        pd.DataFrame — one row per region, sorted by mean_pred_quintile desc.
        """
        rows = []
        for region in sorted(df["region"].astype(str).unique()):
            sub = df[df["region"].astype(str) == region].copy()
            if len(sub) < MIN_REGION_N:
                continue

            avail = [c for c in feature_cols if c in sub.columns]
            X = sub[avail].fillna(sub[avail].median(numeric_only=True))
            y = sub[TARGET].values.astype(int)
            strat = y if pd.Series(y).value_counts().min() >= 2 else None

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, stratify=strat, random_state=self.rs,
            )
            clf = (
                lgb.LGBMClassifier(n_estimators=200, max_depth=6,
                                   class_weight="balanced",
                                   random_state=self.rs, verbose=-1)
                if _HAS_LGB else
                RandomForestClassifier(n_estimators=200, max_depth=10,
                                       class_weight="balanced",
                                       random_state=self.rs, n_jobs=-1)
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            acc  = accuracy_score(y_te, y_pred)
            w_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            mpq  = float(y_pred.mean())

            self.region_models_[region] = {
                "model": clf, "accuracy": acc, "weighted_f1": w_f1,
                "mean_pred_q": mpq, "n_total": len(sub),
                "y_pred": y_pred, "y_test": y_te,
            }
            rows.append({
                "region":             region,
                "n_households":       len(sub),
                "accuracy":           round(acc,  4),
                "weighted_f1":        round(w_f1, 4),
                "mean_pred_quintile": round(mpq,  3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("mean_pred_quintile", ascending=False)
            .reset_index(drop=True)
        )

    # ════════════════════════════════════════════════════════════════════
    # 4. Regional ranking & comparison
    # ════════════════════════════════════════════════════════════════════

    def regional_ranking(self, per_region_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wealth_rank + quintile concentration statistics.
        Saves to data/processed/regional_wealth_ranking.csv.

        Returns DataFrame sorted by wealth_rank (1 = wealthiest).
        """
        df = per_region_df.copy()
        df["wealth_rank"] = (
            df["mean_pred_quintile"].rank(ascending=False).astype(int)
        )
        pct_rows = [
            {"region": r,
             "pct_q1_poorest":    round(100*(m["y_pred"]==1).mean(), 1),
             "pct_q5_wealthiest": round(100*(m["y_pred"]==5).mean(), 1)}
            for r, m in self.region_models_.items()
        ]
        df = df.merge(pd.DataFrame(pct_rows), on="region", how="left")
        df = df.sort_values("wealth_rank").reset_index(drop=True)

        from config import RANKING_CSV
        RANKING_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RANKING_CSV, index=False)
        return df

    def compare_regions(self, region_a: str, region_b: str) -> pd.DataFrame:
        """
        Head-to-head comparison between two regions.

        Returns
        -------
        pd.DataFrame — one row per region with wealth stats + wealthier flag.
        """
        rows = []
        for r in [region_a, region_b]:
            if r not in self.region_models_:
                raise KeyError(f"No model for '{r}'. Run train_per_region() first.")
            m = self.region_models_[r]
            y = m["y_pred"]
            rows.append({
                "region":             r,
                "n_households":       m["n_total"],
                "mean_pred_quintile": round(m["mean_pred_q"], 3),
                "pct_q1_poorest":     round(100*(y==1).mean(), 1),
                "pct_q5_wealthiest":  round(100*(y==5).mean(), 1),
                "Q1": int((y==1).sum()), "Q2": int((y==2).sum()),
                "Q3": int((y==3).sum()), "Q4": int((y==4).sum()),
                "Q5": int((y==5).sum()),
                "accuracy":    round(m["accuracy"],    4),
                "weighted_f1": round(m["weighted_f1"], 4),
            })
        result = pd.DataFrame(rows)
        result["wealthier"] = (
            result["mean_pred_quintile"] == result["mean_pred_quintile"].max()
        ).map({True: "← wealthier", False: ""})
        return result

    def compare_all_pairs(self) -> pd.DataFrame:
        """All pairwise regional comparisons sorted by wealth gap (delta)."""
        regions = list(self.region_models_.keys())
        rows = []
        for i, ra in enumerate(regions):
            for rb in regions[i+1:]:
                qa = self.region_models_[ra]["mean_pred_q"]
                qb = self.region_models_[rb]["mean_pred_q"]
                rows.append({"region_a": ra, "region_b": rb,
                             "mean_q_a": round(qa,3), "mean_q_b": round(qb,3),
                             "wealth_gap": round(abs(qa-qb),3),
                             "wealthier": ra if qa>=qb else rb})
        return (pd.DataFrame(rows)
                .sort_values("wealth_gap", ascending=False)
                .reset_index(drop=True))

    # ════════════════════════════════════════════════════════════════════
    # 5. Feature importance
    # ════════════════════════════════════════════════════════════════════

    def feature_importance(
        self,
        feature_names: list[str],
        model_name: str | None = None,
        region: str | None = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Return feature importances as a DataFrame (for plotting in notebook)."""
        if region:
            clf, label = self.region_models_[region]["model"], f"region={region}"
        else:
            name = model_name or self.best_name_
            clf, label = self.models_.get(name, self.best_model_), name

        if clf is None:
            raise RuntimeError("Call train_evaluate() first.")

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).mean(axis=0)
        else:
            raise AttributeError(f"'{label}' does not expose feature importances.")

        n = len(imp)
        return (
            pd.DataFrame({"feature": feature_names[:n], "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ════════════════════════════════════════════════════════════════════
    # 6. Persistence
    # ════════════════════════════════════════════════════════════════════

    def save(self, out_dir: Path = MODEL_DIR) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.best_model_:
            joblib.dump(self.best_model_, out_dir / "best_model.pkl")
        for region, info in self.region_models_.items():
            safe = region.replace(" ", "_")
            joblib.dump(info["model"], out_dir / f"model_{safe}.pkl")

    def load(self, out_dir: Path = MODEL_DIR) -> None:
        best = out_dir / "best_model.pkl"
        if best.exists():
            self.best_model_ = joblib.load(best)
        for pkl in sorted(out_dir.glob("model_*.pkl")):
            region = pkl.stem.replace("model_", "").replace("_", " ")
            self.region_models_[region] = {"model": joblib.load(pkl)}