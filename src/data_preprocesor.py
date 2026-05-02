"""
Data Preprocessing Module
=========================
Scaling, encoding, train/val/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocessing pipeline."""

    def __init__(self):
        self.scaler = None
        self.encoders = {}

    def separate(self, df, target):
        return df.drop(columns=[target]), df[target]

    def encode_categorical(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        return X

    def scale(self, X, method='standard'):
        X = X.copy()
        num = X.select_dtypes(include=[np.number]).columns
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        if len(num) > 0:
            X[num] = self.scaler.fit_transform(X[num])
        return X

    def split(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        val_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adj, random_state=random_state
        )
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }