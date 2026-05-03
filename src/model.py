from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline


@dataclass
class TrainedModels:
    regressor: Pipeline
    classifier: Pipeline
    all_regressors: Dict[str, Pipeline]
    all_classifiers: Dict[str, Pipeline]


def _candidate_models(random_state: int = 42):
    reg = {
        "random_forest_reg": RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
        "gradient_boosting_reg": GradientBoostingRegressor(random_state=random_state),
    }
    cls = {
        "random_forest_cls": RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=-1, class_weight="balanced"),
        "gradient_boosting_cls": GradientBoostingClassifier(random_state=random_state),
    }
    return reg, cls


def train_models(X_train, y_reg_train, y_cls_train, preprocessor, random_state: int = 42) -> TrainedModels:
    reg_candidates, cls_candidates = _candidate_models(random_state=random_state)

    trained_regs: Dict[str, Pipeline] = {}
    for name, est in reg_candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", est)])
        pipe.fit(X_train, y_reg_train)
        trained_regs[name] = pipe

    trained_cls: Dict[str, Pipeline] = {}
    for name, est in cls_candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", est)])
        pipe.fit(X_train, y_cls_train)
        trained_cls[name] = pipe

    return TrainedModels(
        regressor=trained_regs["random_forest_reg"],
        classifier=trained_cls["random_forest_cls"],
        all_regressors=trained_regs,
        all_classifiers=trained_cls,
    )


def save_all_models(models: TrainedModels, model_dir: str | Path = "models") -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.all_regressors.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    for name, model in models.all_classifiers.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    joblib.dump(models.regressor, model_dir / "best_regressor.joblib")
    joblib.dump(models.classifier, model_dir / "best_classifier.joblib")
