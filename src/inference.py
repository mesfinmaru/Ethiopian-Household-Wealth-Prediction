"""
Prediction API for Dashboard Integration
=========================================
Production-ready interface for making predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class WealthPredictorAPI:
    """Prediction interface for trained model."""

    def __init__(self, model_path='../models/'):
        self.model_path = model_path
        self.model = self._load('best_model.pkl')
        self.scaler = self._load('scaler.pkl')
        self.feature_names = self._load('feature_names.pkl')

    def _load(self, fname):
        fp = os.path.join(self.model_path, fname)
        return joblib.load(fp) if os.path.exists(fp) else None

    def predict_single(self, hh_size=4, head_age=35, head_gender=1,
                       education_years=6, literacy_rate=0.5,
                       rooms=2, has_electricity=1, has_water=1,
                       asset_owned=None, region='Oromia', post_covid=1):
        """Predict consumption for one household"""
        if asset_owned is None:
            asset_owned = ['mobile', 'radio', 'bed']

        features = {
            'hh_size': hh_size, 'head_age': head_age, 'head_gender': head_gender,
            'head_elderly': int(head_age >= 60), 'head_young': int(head_age <= 25),
            'edu_ratio': literacy_rate, 'rooms': rooms,
            'has_electricity': has_electricity, 'has_water': has_water,
            'asset_count': len(asset_owned), 'wave': 5 if post_covid else 4,
            'post_covid': post_covid
        }

        df = pd.DataFrame([features])
        for col in (self.feature_names or []):
            if col not in df.columns:
                df[col] = 0
        if self.feature_names:
            df = df[self.feature_names]

        if self.scaler:
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        if self.model is None:
            return {'error': 'No model loaded'}

        log_pred = float(self.model.predict(df)[0])
        consumption = float(np.expm1(log_pred))
        per_capita = consumption / max(hh_size, 1)

        if per_capita < 8000:
            cat = 'Low'
        elif per_capita < 25000:
            cat = 'Medium'
        else:
            cat = 'High'

        return {
            'log_prediction': log_pred,
            'consumption_etb': consumption,
            'per_capita_etb': per_capita,
            'monthly_per_capita': per_capita / 12,
            'wealth_category': cat,
        }

    def what_if(self, base_params, vary_param, vary_values):
        results = []
        for val in vary_values:
            p = base_params.copy()
            p[vary_param] = val
            pred = self.predict_single(**p)
            pred[vary_param] = val
            results.append(pred)
        return pd.DataFrame(results)