"""
Data Cleaning Module
====================
Missing value imputation, outlier treatment, duplicate removal.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Data cleaning for household survey data."""

    def __init__(self):
        self.imputer = None
        self.cleaning_log = []

    def detect_missing(self, df):
        """Missing values report"""
        m = pd.DataFrame({
            'column': df.columns,
            'missing': df.isnull().sum().values,
            'pct': (df.isnull().sum() / len(df) * 100).values
        })
        return m[m['missing'] > 0].sort_values('pct', ascending=False)

    def handle_missing(self, df, strategy='simple', threshold=0.6):
        """
        Handle missing values.
        strategy: 'simple' (median/mode), 'knn', 'iterative'
        threshold: drop columns with missing % above this
        """
        df = df.copy()
        report = self.detect_missing(df)

        # Drop excessively sparse columns
        drop = report[report['pct'] > threshold * 100]['column'].tolist()
        if drop:
            df = df.drop(columns=drop)
            self.cleaning_log.append(f"Dropped {len(drop)} sparse columns")

        num = df.select_dtypes(include=[np.number]).columns
        cat = df.select_dtypes(include=['object', 'category']).columns

        if strategy == 'simple':
            if len(num) > 0:
                imp = SimpleImputer(strategy='median')
                df[num] = imp.fit_transform(df[num])
                self.imputer = imp
            if len(cat) > 0:
                imp_c = SimpleImputer(strategy='most_frequent')
                df[cat] = imp_c.fit_transform(df[cat])
        elif strategy == 'knn':
            if len(num) > 0:
                imp = KNNImputer(n_neighbors=5)
                df[num] = imp.fit_transform(df[num])
                self.imputer = imp
            if len(cat) > 0:
                imp_c = SimpleImputer(strategy='most_frequent')
                df[cat] = imp_c.fit_transform(df[cat])
        elif strategy == 'iterative':
            if len(num) > 0:
                imp = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                    max_iter=10, random_state=42
                )
                df[num] = imp.fit_transform(df[num])
                self.imputer = imp
            if len(cat) > 0:
                imp_c = SimpleImputer(strategy='most_frequent')
                df[cat] = imp_c.fit_transform(df[cat])

        self.cleaning_log.append(f"Imputed: {strategy}")
        return df

    def handle_outliers(self, df, method='iqr', strategy='cap', threshold=3.0):
        """Cap or remove outliers"""
        df = df.copy()
        num = df.select_dtypes(include=[np.number]).columns

        for col in num:
            if method == 'iqr':
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - threshold * IQR, Q3 + threshold * IQR
            else:
                m, s = df[col].mean(), df[col].std()
                lo, hi = m - threshold * s, m + threshold * s

            if strategy == 'cap':
                df[col] = df[col].clip(lo, hi)
            elif strategy == 'remove':
                df = df[(df[col] >= lo) & (df[col] <= hi)]

        return df.reset_index(drop=True)

    def remove_duplicates(self, df, subset=None):
        n = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        self.cleaning_log.append(f"Removed {n - len(df)} duplicates")
        return df

    def remove_low_variance(self, df):
        from sklearn.feature_selection import VarianceThreshold
        num = df.select_dtypes(include=[np.number])
        sel = VarianceThreshold(threshold=0.001)
        sel.fit(num)
        kept = num.columns[sel.get_support()].tolist()
        dropped = [c for c in num.columns if c not in kept]
        if dropped:
            df = df.drop(columns=dropped)
            self.cleaning_log.append(f"Removed {len(dropped)} low-variance features")
        return df

    def get_report(self):
        return "\n".join([f"{i+1}. {log}" for i, log in enumerate(self.cleaning_log)])