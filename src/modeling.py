"""
Wealth Prediction Models
========================
ML models for household consumption prediction.

MODEL CHOICES:
- Ridge/Lasso: Linear baselines with regularization
- Random Forest: Ensemble, handles non-linearity
- XGBoost/LightGBM: State-of-the-art gradient boosting
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class WealthPredictor:
    """Multi-model wealth prediction with regional capability."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = None
        self.best_model = None
        self.best_name = None
        self.regional_models = {}

        self.base_models = {
            'Ridge': Ridge(alpha=1.0, random_state=random_state),
            'Lasso': Lasso(alpha=0.01, random_state=random_state, max_iter=5000),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15,
                                                    random_state=random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                            random_state=random_state),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                         random_state=random_state, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                           random_state=random_state, verbose=-1),
        }

    def train_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate on test set"""
        results = []
        for name, model in self.base_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                cv_r2 = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
                self.models[name] = model
                results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_R2': cv_r2})
            except Exception as e:
                print(f"  {name}: Error - {e}")

        self.results = pd.DataFrame(results).sort_values('R2', ascending=False)
        if not self.results.empty:
            self.best_name = self.results.iloc[0]['Model']
            self.best_model = self.models[self.best_name]
        return self.results

    def hyperparameter_tune(self, X_train, y_train, model_name='XGBoost', cv=3):
        """Grid search hyperparameter tuning"""
        grids = {
            'XGBoost': {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8], 'learning_rate': [0.01, 0.05, 0.1]},
            'LightGBM': {'n_estimators': [100, 200], 'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1]},
            'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [10, 15, 20], 'min_samples_split': [2, 5, 10]},
        }
        if model_name not in self.base_models:
            return None
        grid = grids.get(model_name, {})
        if not grid:
            return None
        search = GridSearchCV(self.base_models[model_name], grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        self.models[model_name] = search.best_estimator_
        print(f"  Best: {search.best_params_}, R²={search.best_score_:.4f}")
        return search.best_estimator_

    def get_feature_importance(self, model_name=None, top_n=20):
        """Get feature importance"""
        m = self.models.get(model_name) if model_name else self.best_model
        if m is None:
            return None
        if hasattr(m, 'feature_importances_'):
            imp = m.feature_importances_
        elif hasattr(m, 'coef_'):
            imp = np.abs(m.coef_).flatten()
        else:
            return None
        return pd.Series(imp).sort_values(ascending=False).head(top_n)

    def predict_by_region(self, df, target='log_total_consumption', region_col=None, min_samples=30):
        """Train per-region models"""
        if region_col is None:
            for c in df.columns:
                if 'region' in c.lower() and 2 < df[c].nunique() < 20:
                    region_col = c
                    break
        if region_col is None:
            print("No region column found")
            return None

        print(f"\nREGIONAL MODELS ({region_col}):")
        results = []
        for region in sorted(df[region_col].dropna().unique()):
            rdf = df[df[region_col] == region].copy()
            if len(rdf) < min_samples:
                continue
            X = rdf.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
            y = rdf[target]
            X = X.fillna(X.median())
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=self.random_state, verbose=-1)
            m.fit(X_tr, y_tr)
            yp = m.predict(X_te)
            r2 = r2_score(y_te, yp)
            self.regional_models[region] = m
            results.append({'Region': region, 'N': len(rdf), 'R2': r2})
            print(f"  {region}: R²={r2:.3f} (n={len(rdf)})")
        return pd.DataFrame(results)