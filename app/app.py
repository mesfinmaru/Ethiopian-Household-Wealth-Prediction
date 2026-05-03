"""Streamlit app for training and inference with saved models in `models/` folder."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath('..'))
from src.pipeline import run_pipeline

st.set_page_config(page_title="ESS Welfare Pipeline", layout="wide")
st.title("Ethiopian Household Welfare - Training & Prediction App")

st.markdown("This app can run the multi-wave pipeline and uses models saved in the `models/` directory.")

with st.expander("1) Run training pipeline"):
    st.write("Set local wave folders (2011, 2013, 2015, 2018, 2021).")
    base = st.text_input("Base data directory", value="data")
    if st.button("Run Pipeline"):
        wave_map = {2011: f"{base}/2011", 2013: f"{base}/2013", 2015: f"{base}/2015", 2018: f"{base}/2018", 2021: f"{base}/2021"}
        try:
            metrics = run_pipeline(wave_map=wave_map, output_dir="outputs", model_dir="models")
            st.success("Pipeline completed")
            st.json(metrics)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

with st.expander("2) Inspect saved models"):
    model_dir = Path("../models") if Path("../models").exists() else Path("models")
    files = sorted([p.name for p in model_dir.glob("*.joblib")]) if model_dir.exists() else []
    st.write("Model folder:", str(model_dir))
    st.write(files if files else "No model files found yet.")

with st.expander("3) Batch prediction preview"):
    pred_file = st.text_input("Predictions CSV", value="outputs/predictions.csv")
    if st.button("Load predictions"):
        try:
            df = pd.read_csv(pred_file)
            st.dataframe(df.head(20))
        except Exception as exc:
            st.error(f"Could not load predictions: {exc}")
