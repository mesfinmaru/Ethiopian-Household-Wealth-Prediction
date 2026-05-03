from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict:
    return {"rmse": mean_squared_error(y_true, y_pred, squared=False), "r2": r2_score(y_true, y_pred)}


def evaluate_classification(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
