from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from src.data_loader import LSMSDataLoader
from src.data_merger import combine_waves
from src.data_cleaner import DataCleaner
from src.feature_engineering import add_household_features, build_targets
from src.preprocessing import make_preprocessor
from src.model import save_all_models, train_models
from src.evaluation import evaluate_classification, evaluate_regression


def run_pipeline(wave_map: dict[int, str], output_dir: str = "outputs", model_dir: str = "models") -> dict:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    loader = LSMSDataLoader(data_root=".")
    waves = loader.load_multi_wave(wave_map)
    df = combine_waves(waves)

    cleaner = DataCleaner()
    df = cleaner.clean(df, sparse_threshold=0.75, numeric_strategy="median", categorical_strategy="most_frequent")

    df = add_household_features(df)
    df = build_targets(df)

    split_year = int(max(df["year"].unique()))
    train_df = df[df["year"] < split_year].copy()
    test_df = df[df["year"] == split_year].copy()

    target_cols = ["consumption_per_capita", "poor"]
    preprocessor = make_preprocessor(train_df, target_cols=target_cols)
    models = train_models(train_df.drop(columns=target_cols), train_df["consumption_per_capita"], train_df["poor"], preprocessor)

    X_test = test_df.drop(columns=target_cols, errors="ignore")
    reg_pred = models.regressor.predict(X_test)
    cls_pred = models.classifier.predict(X_test)

    reg_metrics = evaluate_regression(test_df["consumption_per_capita"], reg_pred)
    cls_metrics = evaluate_classification(test_df["poor"], cls_pred)

    X_all = df.drop(columns=target_cols, errors="ignore")
    df["pred_consumption_per_capita"] = models.regressor.predict(X_all)
    df["pred_poor"] = models.classifier.predict(X_all)

    region_avg = (
        df.groupby("region", dropna=False)["pred_consumption_per_capita"].mean().reset_index(name="avg_pred_consumption")
        if "region" in df.columns else pd.DataFrame()
    )
    zone_group = [c for c in ["region", "zone"] if c in df.columns] or ["year"]
    pov_region_zone = df.groupby(zone_group, dropna=False)["pred_poor"].mean().reset_index(name="poverty_rate")
    national = pd.DataFrame({
        "national_avg_pred_consumption": [df["pred_consumption_per_capita"].mean()],
        "national_pred_poverty_rate": [df["pred_poor"].mean()],
    })

    df.to_csv(outdir / "predictions.csv", index=False)
    region_avg.to_csv(outdir / "avg_consumption_by_region.csv", index=False)
    pov_region_zone.to_csv(outdir / "poverty_rate_region_zone.csv", index=False)
    national.to_csv(outdir / "national_summary.csv", index=False)

    save_all_models(models, model_dir=model_dir)
    joblib.dump({"cleaning_summary": cleaner.summary, "test_year": split_year}, Path(model_dir) / "training_metadata.joblib")

    return {
        "regression": reg_metrics,
        "classification": cls_metrics,
        "test_year": split_year,
        "rows": len(df),
        "cleaning_summary": cleaner.summary.__dict__,
        "model_dir": model_dir,
    }
