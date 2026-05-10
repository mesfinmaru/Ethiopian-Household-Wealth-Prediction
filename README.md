# Ethiopian Household Wealth Prediction

Streamlit application and machine learning pipeline for predicting Ethiopian household wealth quintiles from five waves of World Bank LSMS-ISA / ESS survey data.

## Overview

This project builds a leakage-free classification system that predicts `cons_quint` on a 1 to 5 scale, where 1 represents the poorest 20 percent of households and 5 represents the wealthiest 20 percent. The workflow combines survey decoding, cleaning, feature engineering, supervised learning, regional ranking, and an interactive Streamlit dashboard.

The application is designed for academic analysis and policy-oriented exploration rather than operational targeting or allocation decisions.

## What The App Does

The Streamlit interface in `app/app.py` provides the full workflow in one place:

- Home page with project overview, CRISP-DM pipeline, dataset build action, and summary metrics
- Data Explorer for raw previews, summary statistics, missing-value inspection, and regional comparison
- EDA for distributions, bivariate analysis, correlation views, temporal trends, and shock exposure
- Preprocessing audit with cleaning logs, imputation strategy, feature engineering, and preprocessing groups
- Modelling workspace for training and comparing classifiers, plus per-region models
- Regional Wealth Map for ranking regions and comparing two regions side by side
- Predict Household form for single-household quintile prediction
- About page with methods, ethics, and project references

## Core Pipeline

The codebase is organized around a survey-to-model pipeline:

1. `src/sav_reader.py` decodes the Wave 2 SPSS `.sav` files in pure Python and normalizes truncated variable names.
2. `src/data_loader.py` loads each wave’s survey modules, merges them, and keeps the pipeline leakage-free.
3. `src/missing_value_handler.py` applies survey-aware imputation strategies, including wave-aware donor filling and group imputation.
4. `src/data_cleaner.py` wraps the missing-value pipeline with outlier capping, coverage flags, and variance filtering.
5. `src/feature_enginner.py` creates interpretable domain features such as housing quality, asset wealth, labour intensity, vulnerability, and geography-based signals.
6. `src/data_preprocesor.py` builds the sklearn `ColumnTransformer` pipeline and stratified train/validation/test splits.
7. `src/modeling.py` trains and evaluates multiple classifiers, supports tuning, per-region modelling, regional ranking, and model persistence.

## Models And Methods

The supervised learning module evaluates multiple classification models for the five-class wealth task, including:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Naive Bayes
- Gradient Boosting
- AdaBoost
- XGBoost, when installed
- LightGBM, when installed

The modelling layer also supports:

- Stratified cross-validation
- Weighted and macro F1 scoring
- Confusion matrix inspection
- Feature importance extraction
- Per-region wealth prediction and ranking
- Saving and loading trained models with `joblib`

## Data Sources

The repository uses survey data from the World Bank LSMS-ISA / Ethiopian Socioeconomic Survey waves:

- W1: 2011–12
- W2: 2013–14
- W3: 2015–16
- W4: 2018–19
- W5: 2021–22

Raw files are stored under `data/raw/`, and the processed combined dataset is written to `data/processed/all_waves_clean.csv`.

## Project Structure

```text
app/
    app.py
data/
    raw/
    processed/
models/
notebooks/
reports/
src/
    config.py
    data_cleaner.py
    data_loader.py
    data_preprocesor.py
    feature_enginner.py
    missing_value_handler.py
    modeling.py
    sav_reader.py
requirements.txt
README.md
```

## Setup

Run the following from the project root (`DSA Project`).

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in your current user scope:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Confirm raw survey data exists

Make sure `data/raw/` contains all five survey-wave folders and section files before building/training.

## Run

### Run the Streamlit app

Use the venv Python explicitly (works even if `streamlit` is not on PATH):

```powershell
.venv\Scripts\python.exe -m streamlit run app/app.py
```

Then open:

- `http://localhost:8501`

If the combined dataset is missing, open the Home page and click **Build Dataset** to create `data/processed/all_waves_clean.csv`.

### Run tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s test -p "test_*.py" -v
```

## Typical Workflow

1. Load or build the combined dataset.
2. Inspect missingness and coverage in the Data Explorer.
3. Review cleaning and feature engineering steps in the Preprocessing page.
4. Train and compare models in the Modelling page.
5. Explore regional wealth patterns and compare regions.
6. Use the prediction form for a single-household estimate.

## Notes

- The project avoids leakage by excluding consumption aggregate columns from features.
- Wave 2 uses a custom SPSS decoder to handle `.sav` files without external parsing dependencies.
- Saved artifacts are written to `models/` and processed outputs to `data/processed/`.
- Some source filenames retain historical typos from the original project structure, such as `data_preprocesor.py` and `feature_enginner.py`.

## Dependencies

The project is built with Python and the scientific stack, including Streamlit, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Joblib, XGBoost, LightGBM, and pyreadstat.

## License And Attribution

This project uses World Bank LSMS-ISA / ESS survey data. Please follow the relevant data access and citation requirements from the source provider.

## Last Updated

2026-05-09
