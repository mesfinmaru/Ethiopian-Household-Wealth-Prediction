"""
modeling.py
═══════════════════════════════════════════════════════════════════════════════
WealthPredictor + ModelEvaluator

WealthPredictor: multi-class classification for cons_quint (1–5)
  Trains: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes,
          Gradient Boosting, XGBoost, LightGBM (if installed)
  Modes:  overall Ethiopia | per-region | regional ranking | pairwise comparison

ModelEvaluator: comprehensive evaluation toolkit.

TARGET: cons_quint (1=poorest, 5=wealthiest) — 5-class classification.
INPUT : legitimate proxy features only. NEVER consumption aggregates.

═══════════════════════════════════════════════════════════════════════════════
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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

from config import MIN_REGION_N, MODEL_DIR, RANDOM_STATE, TARGET


# ════════════════════════════════════════════════════════════════════════════════
# ModelEvaluator — class-reference interface (Chapter 4)
# ════════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Comprehensive evaluation toolkit for classification and regression.

    Mirrors the class-reference ModelEvaluator from Chapter 4 but extended
    for 5-class wealth quintile output (multi-class metrics, per-class ROC).

    Usage
    -----
        evaluator = ModelEvaluator()
        metrics   = evaluator.evaluate_classification(model, X_test, y_test)
        evaluator.plot_confusion_matrix(y_test, y_pred)
        evaluator.plot_roc_curves(model, X_test, y_test)
        cv_df     = evaluator.cross_validate(model, X_train, y_train)
        lc_df     = evaluator.learning_curve_analysis(model, X_train, y_train)
    """

    def __init__(self):
        self.classification_metrics = {}
        self.cv_results_            = {}

    # ── Classification evaluation ─────────────────────────────────────────────

    def evaluate_classification(self, model, X_test, y_test,
                                 y_pred=None) -> dict:
        """
        Comprehensive classification evaluation with all metrics.
        Prints results and returns metrics dict.
        """
        if y_pred is None:
            y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

        self.classification_metrics = {
            "accuracy":    round(acc,  4),
            "precision":   round(prec, 4),
            "recall":      round(rec,  4),
            "weighted_f1": round(f1_w, 4),
            "macro_f1":    round(f1_m, 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        print("── Classification Evaluation ───────────────────────")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}  (weighted)")
        print(f"  Recall    : {rec:.4f}  (weighted)")
        print(f"  F1 Score  : {f1_w:.4f}  (weighted)")
        print(f"  Macro F1  : {f1_m:.4f}")
        print("\nDetailed Classification Report:")
        q_names = [f"Q{i}" for i in range(1, 6)]
        print(classification_report(y_test, y_pred,
                                     target_names=q_names, zero_division=0))
        return self.classification_metrics

    def cross_validate(self, model, X_train, y_train,
                       cv_folds: int = 5, scoring: str = "f1_weighted") -> pd.DataFrame:
        """
        Stratified k-fold cross-validation.
        Returns DataFrame with fold scores and mean ± std.
        Chapter 4 reference: cross_val_score, StratifiedKFold.
        """
        skf     = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=RANDOM_STATE)
        scores  = cross_val_score(model, X_train, y_train,
                                   cv=skf, scoring=scoring, n_jobs=-1)
        rows    = [{"fold": i+1, scoring: round(s, 4)} for i, s in enumerate(scores)]
        rows.append({"fold": "MEAN", scoring: round(scores.mean(), 4)})
        rows.append({"fold": "STD",  scoring: round(scores.std(),  4)})
        result  = pd.DataFrame(rows)
        print(f"\nCV ({cv_folds}-fold) {scoring}: "
              f"{scores.mean():.4f} ± {scores.std():.4f}")
        self.cv_results_[scoring] = result
        return result

    def learning_curve_analysis(self, model, X_train, y_train,
                                 cv_folds: int = 5) -> pd.DataFrame:
        """
        Learning curve analysis: detects overfitting/underfitting.
        Returns DataFrame with train_size, train_score, val_score.
        Chapter 4 reference: learning_curve from sklearn.
        """
        sizes = np.linspace(0.1, 1.0, 10)
        skf   = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                 random_state=RANDOM_STATE)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=sizes, cv=skf,
            scoring="f1_weighted", n_jobs=-1,
        )
        return pd.DataFrame({
            "train_size":  train_sizes,
            "train_score": train_scores.mean(axis=1).round(4),
            "train_std":   train_scores.std(axis=1).round(4),
            "val_score":   val_scores.mean(axis=1).round(4),
            "val_std":     val_scores.std(axis=1).round(4),
        })

    def classification_report_df(self, y_test, y_pred) -> pd.DataFrame:
        """Classification report as a labelled DataFrame."""
        q_names = [f"Q{i} ({'poorest' if i==1 else 'wealthiest' if i==5 else ''})"
                   for i in range(1, 6)]
        rep = classification_report(y_test, y_pred,
                                    target_names=q_names,
                                    output_dict=True, zero_division=0)
        return pd.DataFrame(rep).T.round(4)

    def confusion_matrix_df(self, y_test, y_pred,
                             labels: list = None) -> pd.DataFrame:
        """Confusion matrix as a labelled DataFrame."""
        lbl = labels or [f"Q{i}" for i in range(1, 6)]
        return pd.DataFrame(confusion_matrix(y_test, y_pred),
                             index=lbl, columns=lbl)


# ════════════════════════════════════════════════════════════════════════════════
# WealthPredictor — full modelling pipeline for Ethiopian wealth quintiles
# ════════════════════════════════════════════════════════════════════════════════

class WealthPredictor:
    """
    Train and evaluate wealth quintile classifiers for Ethiopia.

    Combines the class-reference ClassificationPipeline interface with
    ESS-specific features: per-region models, regional wealth ranking,
    and pairwise regional wealth comparison.

    Usage
    -----
        wp      = WealthPredictor()
        results = wp.train_evaluate(X_train, y_train, X_test, y_test)
        reg_df  = wp.train_per_region(df, feature_cols)
        ranking = wp.regional_ranking(reg_df)
        comp    = wp.compare_regions("TIGRAY","OROMIA")
        imp_df  = wp.feature_importance(feature_names)
        wp.save()
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.rs             = random_state
        self.models_        = {}
        self.results_       = None
        self.best_model_    = None
        self.best_name_     = ""
        self.region_models_ = {}
        self.evaluator      = ModelEvaluator()
        self.label_encoder_ = None

    # ── Model registry (Chapter 4 reference: all algorithms) ─────────────────

    def _build_models(self) -> dict:
        """Build the full model registry with appropriate hyper-parameters."""
        m = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=self.rs),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=5,
                class_weight="balanced", random_state=self.rs),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", random_state=self.rs, n_jobs=-1),
            "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=self.rs),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=200, learning_rate=0.1, random_state=self.rs),
        }
        if _HAS_XGB:
            m["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=self.rs, verbosity=0)
        if _HAS_LGB:
            m["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=self.rs, verbose=-1)
        return m

    # ── 1. Overall training & evaluation ─────────────────────────────────────

    def train_evaluate(self, X_train, y_train, X_test, y_test,
                       cv_folds: int = 5) -> pd.DataFrame:
        """
        Train all classifiers. Evaluate on test set + stratified CV on train.

        Metrics: accuracy, weighted F1, macro F1, CV F1 mean ± std.
        Returns DataFrame sorted by weighted_f1 descending.
        train_all_models, cross_validate_models.
        """
        cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                       random_state=self.rs)
        # Encode labels to 0..n_classes-1 to satisfy learners like XGBoost
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_train)
        y_tr_enc = self.label_encoder_.transform(y_train)
        y_te_enc = self.label_encoder_.transform(y_test)
        rows = []
        for name, model in self._build_models().items():
            model.fit(X_train, y_tr_enc)
            y_pred_enc = model.predict(X_test)
            # decode predictions back to original label space for reporting
            try:
                y_pred = self.label_encoder_.inverse_transform(y_pred_enc)
            except Exception:
                y_pred = y_pred_enc
            acc       = accuracy_score(y_test, y_pred)
            w_f1      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            m_f1      = f1_score(y_test, y_pred, average="macro",    zero_division=0)
            prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec       = recall_score(y_test, y_pred, average="weighted",    zero_division=0)
            cv_scores = cross_val_score(model, X_train, y_tr_enc,
                                        cv=cv, scoring="f1_weighted", n_jobs=-1)
            self.models_[name] = model
            rows.append({
                "model":       name,
                "accuracy":    round(acc,              4),
                "precision":   round(prec,             4),
                "recall":      round(rec,              4),
                "weighted_f1": round(w_f1,             4),
                "macro_f1":    round(m_f1,             4),
                "cv_f1_mean":  round(cv_scores.mean(), 4),
                "cv_f1_std":   round(cv_scores.std(),  4),
            })

        self.results_ = (pd.DataFrame(rows)
                         .sort_values("weighted_f1", ascending=False)
                         .reset_index(drop=True))
        self.best_name_  = self.results_.iloc[0]["model"]
        self.best_model_ = self.models_[self.best_name_]
        print(f"\n✓ Best model: {self.best_name_} "
              f"(weighted F1 = {self.results_.iloc[0]['weighted_f1']:.4f})")
        return self.results_

    def classification_report_df(self, X_test, y_test) -> pd.DataFrame:
        """Full classification report for the best model as a DataFrame."""
        y_pred_enc = self.best_model_.predict(X_test)
        if self.label_encoder_ is not None:
            try:
                y_pred = self.label_encoder_.inverse_transform(y_pred_enc)
            except Exception:
                y_pred = y_pred_enc
        else:
            y_pred = y_pred_enc
        return self.evaluator.classification_report_df(y_test, y_pred)

    def confusion_matrix_df(self, X_test, y_test) -> pd.DataFrame:
        """Labelled confusion matrix for the best model."""
        y_pred_enc = self.best_model_.predict(X_test)
        if self.label_encoder_ is not None:
            try:
                y_pred = self.label_encoder_.inverse_transform(y_pred_enc)
            except Exception:
                y_pred = y_pred_enc
        else:
            y_pred = y_pred_enc
        return self.evaluator.confusion_matrix_df(y_test, y_pred)

    # ── 2. Hyperparameter tuning ──────────────────────────────────────────────

    def tune(self, X_train, y_train,
             model_name: str = "XGBoost", cv_folds: int = 3) -> pd.DataFrame:
        """
        GridSearchCV for the specified model.
        Returns CV results DataFrame; best params stored in .attrs.
        GridSearchCV hyperparameter optimisation.
        """
        grids = {
            "XGBoost":        {"n_estimators":[200,400],"max_depth":[4,6,8],
                               "learning_rate":[0.03,0.05,0.1]},
            "LightGBM":       {"n_estimators":[200,400],"max_depth":[4,6],
                               "learning_rate":[0.05,0.1]},
            "Random Forest":  {"n_estimators":[200,400],"max_depth":[8,12,16],
                               "min_samples_leaf":[3,5,10]},
            "Gradient Boosting": {"n_estimators":[100,200],"max_depth":[3,5],
                                  "learning_rate":[0.05,0.1]},
            "Decision Tree":  {"max_depth":[6,10,15],"min_samples_leaf":[3,5,10]},
        }
        base = self._build_models()
        if model_name not in base:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(base)}")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.rs)
        gs = GridSearchCV(base[model_name], grids.get(model_name, {}),
                          cv=cv, scoring="f1_weighted", n_jobs=-1,
                          verbose=0, return_train_score=True)
        # encode labels to satisfy learners like XGBoost which expect 0..n-1
        le = LabelEncoder()
        le.fit(y_train)
        y_tr_enc = le.transform(y_train)
        gs.fit(X_train, y_tr_enc)
        self.models_[model_name] = gs.best_estimator_
        if model_name == self.best_name_:
            self.best_model_ = gs.best_estimator_

        out = (pd.DataFrame(gs.cv_results_)
               [["params","mean_test_score","std_test_score","rank_test_score"]]
               .sort_values("rank_test_score"))
        out.attrs["best_params"]   = gs.best_params_
        out.attrs["best_cv_score"] = round(gs.best_score_, 4)
        print(f"Best params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")
        return out

    # ── 3. Per-region models ──────────────────────────────────────────────────

    def train_per_region(self, df: pd.DataFrame, feature_cols: list,
                         test_size: float = 0.20) -> pd.DataFrame:
        """
        Train one model per region predicting cons_quint.
        Uses LightGBM if available, else Random Forest.

        Returns DataFrame with per-region accuracy, F1, mean predicted quintile.
        Minimum MIN_REGION_N households required per region.
        """
        rows = []
        for region in sorted(df["region"].astype(str).unique()):
            sub = df[df["region"].astype(str) == region].copy()
            if len(sub) < MIN_REGION_N:
                continue
            avail = [c for c in feature_cols if c in sub.columns]
            X     = sub[avail].fillna(sub[avail].median(numeric_only=True))
            y     = sub[TARGET].values.astype(int)
            strat = y if pd.Series(y).value_counts().min() >= 2 else None
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, stratify=strat, random_state=self.rs)

            clf = (lgb.LGBMClassifier(n_estimators=200, max_depth=6,
                                      class_weight="balanced",
                                      random_state=self.rs, verbose=-1)
                   if _HAS_LGB else
                   RandomForestClassifier(n_estimators=200, max_depth=10,
                                          class_weight="balanced",
                                          random_state=self.rs, n_jobs=-1))
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)
            w_f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            mpq    = float(y_pred.mean())
            self.region_models_[region] = {
                "model":       clf,
                "accuracy":    acc,
                "weighted_f1": w_f1,
                "mean_pred_q": mpq,
                "n_total":     len(sub),
                "y_pred":      y_pred,
                "y_test":      y_te,
            }
            rows.append({
                "region":              region,
                "n_households":        len(sub),
                "accuracy":            round(acc,  4),
                "weighted_f1":         round(w_f1, 4),
                "mean_pred_quintile":  round(mpq,  3),
            })

        return (pd.DataFrame(rows)
                .sort_values("mean_pred_quintile", ascending=False)
                .reset_index(drop=True))

    # ── 4. Regional ranking ───────────────────────────────────────────────────

    def regional_ranking(self, per_region_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wealth_rank + quintile concentration to per-region results.
        Saves ranking to data/processed/regional_wealth_ranking.csv.
        """
        df = per_region_df.copy()
        df["wealth_rank"] = df["mean_pred_quintile"].rank(ascending=False).astype(int)
        pct = pd.DataFrame([
            {"region":             r,
             "pct_q1_poorest":    round(100*(m["y_pred"]==1).mean(), 1),
             "pct_q5_wealthiest": round(100*(m["y_pred"]==5).mean(), 1)}
            for r, m in self.region_models_.items()
        ])
        df = df.merge(pct, on="region", how="left")
        df = df.sort_values("wealth_rank").reset_index(drop=True)
        from config import RANKING_CSV
        RANKING_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RANKING_CSV, index=False)
        return df

    # ── 5. Regional comparison ────────────────────────────────────────────────

    def compare_regions(self, region_a: str, region_b: str) -> pd.DataFrame:
        """Head-to-head wealth comparison of two regions."""
        rows = []
        for r in (region_a, region_b):
            if r not in self.region_models_:
                raise KeyError(f"No model for '{r}'. Run train_per_region() first.")
            m  = self.region_models_[r]
            yp = m["y_pred"]
            rows.append({
                "region":             r,
                "n_households":       m["n_total"],
                "mean_pred_quintile": round(m["mean_pred_q"], 3),
                "pct_q1_poorest":     round(100*(yp==1).mean(), 1),
                "pct_q5_wealthiest":  round(100*(yp==5).mean(), 1),
                "Q1":int((yp==1).sum()),"Q2":int((yp==2).sum()),
                "Q3":int((yp==3).sum()),"Q4":int((yp==4).sum()),
                "Q5":int((yp==5).sum()),
                "accuracy":    round(m["accuracy"],    4),
                "weighted_f1": round(m["weighted_f1"], 4),
            })
        result = pd.DataFrame(rows)
        result["wealthier"] = (
            result["mean_pred_quintile"] == result["mean_pred_quintile"].max()
        ).map({True: "← wealthier", False: ""})
        return result

    def compare_all_pairs(self) -> pd.DataFrame:
        """All pairwise region comparisons sorted by wealth gap (largest first)."""
        regions = list(self.region_models_)
        rows = []
        for i, ra in enumerate(regions):
            for rb in regions[i+1:]:
                qa = self.region_models_[ra]["mean_pred_q"]
                qb = self.region_models_[rb]["mean_pred_q"]
                rows.append({"region_a":ra,"region_b":rb,
                             "mean_q_a":round(qa,3),"mean_q_b":round(qb,3),
                             "wealth_gap":round(abs(qa-qb),3),
                             "wealthier":ra if qa >= qb else rb})
        return (pd.DataFrame(rows)
                .sort_values("wealth_gap", ascending=False)
                .reset_index(drop=True))

    # ── 6. Feature importance ─────────────────────────────────────────────────

    def feature_importance(self, feature_names: list,
                            model_name: str = None,
                            region: str = None,
                            top_n: int = 20) -> pd.DataFrame:
        """
        Feature importances from the best model or a region model.
        Works for tree-based models; falls back to |coef| for LR.
        Chapter 4 reference: feature importance analysis.
        """
        if region:
            clf = self.region_models_[region]["model"]
        else:
            name = model_name or self.best_name_
            clf  = self.models_.get(name, self.best_model_)
        if clf is None:
            raise RuntimeError("Call train_evaluate() first.")

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).mean(axis=0)
        else:
            raise AttributeError("Model does not expose feature importances.")

        n = len(imp)
        return (pd.DataFrame({"feature": feature_names[:n], "importance": imp})
                .sort_values("importance", ascending=False)
                .head(top_n).reset_index(drop=True))

    # ── 7. Persistence ────────────────────────────────────────────────────────

    def save(self, out_dir: str = None):
        """Save best model and all region models to models/ directory."""
        from pathlib import Path
        out = Path(out_dir) if out_dir else MODEL_DIR
        out.mkdir(parents=True, exist_ok=True)
        if self.best_model_:
            joblib.dump(self.best_model_, out / "best_model.pkl")
            print(f"Best model saved → {out / 'best_model.pkl'}")
        for region, info in self.region_models_.items():
            safe = region.replace(" ", "_")
            joblib.dump(info["model"], out / f"model_{safe}.pkl")

    def load(self, out_dir: str = None):
        """Load previously saved models from models/ directory."""
        from pathlib import Path
        out  = Path(out_dir) if out_dir else MODEL_DIR
        best = out / "best_model.pkl"
        if best.exists():
            self.best_model_ = joblib.load(best)
        for pkl in sorted(out.glob("model_*.pkl")):
            region = pkl.stem.replace("model_","").replace("_"," ")
            self.region_models_[region] = {"model": joblib.load(pkl)}