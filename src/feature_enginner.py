"""
Feature Engineering Module
==========================
Creates derived features for wealth prediction.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature creation for household wealth data."""

    def __init__(self):
        self.created = []

    def log_transforms(self, df):
        """Log transform skewed features (skew > 1)"""
        df = df.copy()
        for c in df.select_dtypes(include=[np.number]).columns:
            if df[c].min() >= 0 and abs(df[c].skew()) > 1:
                df[f'{c}_log'] = np.log1p(df[c])
                self.created.append(f'{c}_log')
        return df

    def interactions(self, df, top_n=8):
        """Pairwise interactions for top features"""
        df = df.copy()
        num = df.select_dtypes(include=[np.number])
        top = num.columns[:top_n]
        for i in range(len(top)):
            for j in range(i+1, len(top)):
                name = f'{top[i]}_x_{top[j]}'
                df[name] = df[top[i]] * df[top[j]]
                self.created.append(name)
        return df

    def ratios(self, df):
        df = df.copy()
        if 'total_consumption' in df.columns and 'hh_size' in df.columns:
            df['cons_per_capita'] = df['total_consumption'] / df['hh_size'].clip(lower=1)
            self.created.append('cons_per_capita')
        if 'asset_count' in df.columns and 'hh_size' in df.columns:
            df['assets_per_capita'] = df['asset_count'] / df['hh_size'].clip(lower=1)
            self.created.append('assets_per_capita')
        return df

    def engineer_all(self, df):
        df = self.log_transforms(df)
        df = self.interactions(df)
        df = self.ratios(df)
        print(f"  Engineered {len(self.created)} features")
        return df