"""
data_cleaner.py
Ethiopian Household Wealth Prediction - Data Understanding & Cleaning
Handles multi-wave harmonization for ESS consumption aggregate data.

Waves:
  W1: 2011-12  (cons_agg_2011-12_w1.csv)
  W2: 2013-14  (cons_agg_2013-14_w2.sav)  — requires pyreadstat
  W3: 2015-16  (cons_agg_2015-16_w3.csv)
  W4: 2018-19  (cons_agg_2018-19_w4.csv)
  W5: 2021-22  (cons_agg_2021-22_w5.csv)

Target: cons_quint (1=poorest … 5=wealthiest)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Region code → label mapping (saq01 in W1/W2/W3 is numeric)
REGION_MAP = {
    1: "TIGRAY",
    2: "AFAR",
    3: "AMHARA",
    4: "OROMIA",
    5: "SOMALI",
    6: "BENISHANGUL GUMUZ",
    7: "SNNP",
    12: "GAMBELA",
    13: "HARAR",
    14: "ADDIS ABABA",
    15: "DIRE DAWA",
}


# ── Core feature columns that exist (or can be derived) in every wave
CORE_COLS = [
    "household_id",
    "ea_id",
    "wave",
    "region",        # harmonised from saq01
    "rural",         # 1=rural, 0=urban
    "hh_size",
    "adulteq",
    "food_cons_ann",
    "nonfood_cons_ann",
    "educ_cons_ann",
    "total_cons_ann",
    "nom_totcons_aeq",
    "cons_quint",    # TARGET
]

# Extra cols available only in W4 & W5 - kept as optional enrichment
EXTRA_COLS = ["fafh_cons_ann", "utilities_cons_ann", "spat_totcons_aeq"]


# ════════════════════════════════════════════════════════════════════════════
# Wave loaders
# ════════════════════════════════════════════════════════════════════════════

def _harmonise_region(df: pd.DataFrame, col: str = "saq01") -> pd.Series:
    """Convert saq01 to a consistent string region label."""
    col_vals = df[col]
    if pd.api.types.is_numeric_dtype(col_vals):
        return col_vals.map(REGION_MAP)
    return col_vals.str.strip().str.upper()


def _harmonise_rural(df: pd.DataFrame, col: str) -> pd.Series:
    """Return rural as integer: 1=rural, 0=urban."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return (s.str.strip().str.upper() == "RURAL").astype(int)


def load_wave1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["household_id"]    = df["household_id"].astype(str)
    out["ea_id"]           = df["ea_id"].astype(str)
    out["wave"]            = 1
    out["region"]          = _harmonise_region(df, "saq01")
    out["rural"]           = _harmonise_rural(df, "rural")
    out["hh_size"]         = df["hh_size"]
    out["adulteq"]         = df["adulteq"]
    out["food_cons_ann"]   = df["food_cons_ann"]
    out["nonfood_cons_ann"]= df["nonfood_cons_ann"]
    out["educ_cons_ann"]   = df["educ_cons_ann"]
    out["total_cons_ann"]  = df["total_cons_ann"]
    out["nom_totcons_aeq"] = df["nom_totcons_aeq"]
    out["cons_quint"]      = df["cons_quint"]
    # Optional extras — fill with NaN for consistency
    for c in EXTRA_COLS:
        out[c] = np.nan
    return out


def load_wave2(path: str) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
    except ImportError:
        raise ImportError(
        )
    out = pd.DataFrame()
    out["household_id"]    = df["household_id"].astype(str)
    out["ea_id"]           = df["ea_id"].astype(str)
    out["wave"]            = 2
    out["region"]          = _harmonise_region(df, "saq01")
    out["rural"]           = _harmonise_rural(df, "rural")
    out["hh_size"]         = df["hh_size"]
    out["adulteq"]         = df["adulteq"]
    out["food_cons_ann"]   = df["food_cons_ann"]
    out["nonfood_cons_ann"]= df["nonfood_cons_ann"]
    out["educ_cons_ann"]   = df["educ_cons_ann"]
    out["total_cons_ann"]  = df["total_cons_ann"]
    out["nom_totcons_aeq"] = df["nom_totcons_aeq"]
    out["cons_quint"]      = df["cons_quint"]
    for c in EXTRA_COLS:
        out[c] = np.nan
    return out


def load_wave3(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["household_id"]    = df["household_id"].astype(str)
    out["ea_id"]           = df["ea_id"].astype(str)
    out["wave"]            = 3
    out["region"]          = _harmonise_region(df, "saq01")
    out["rural"]           = _harmonise_rural(df, "rural")
    out["hh_size"]         = df["hh_size"]
    out["adulteq"]         = df["adulteq"]
    out["food_cons_ann"]   = df["food_cons_ann"]
    out["nonfood_cons_ann"]= df["nonfood_cons_ann"]
    out["educ_cons_ann"]   = df["educ_cons_ann"]
    out["total_cons_ann"]  = df["total_cons_ann"]
    out["nom_totcons_aeq"] = df["nom_totcons_aeq"]
    out["cons_quint"]      = df["cons_quint"]
    for c in EXTRA_COLS:
        out[c] = np.nan
    return out


def load_wave4(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["household_id"]    = df["household_id"].astype(str)
    out["ea_id"]           = df["ea_id"].astype(str)
    out["wave"]            = 4
    out["region"]          = _harmonise_region(df, "saq01")
    out["rural"]           = _harmonise_rural(df, "saq14")
    out["hh_size"]         = df["hh_size"]
    out["adulteq"]         = df["adulteq"]
    out["food_cons_ann"]   = df["food_cons_ann"]
    out["nonfood_cons_ann"]= df["nonfood_cons_ann"]
    out["educ_cons_ann"]   = df["educ_cons_ann"]
    out["total_cons_ann"]  = df["total_cons_ann"]
    out["nom_totcons_aeq"] = df["nom_totcons_aeq"]
    out["cons_quint"]      = df["cons_quint"]
    out["fafh_cons_ann"]   = df.get("fafh_cons_ann", np.nan)
    out["utilities_cons_ann"] = df.get("utilities_cons_ann", np.nan)
    out["spat_totcons_aeq"]= df.get("spat_totcons_aeq", np.nan)
    return out


def load_wave5(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["household_id"]    = df["household_id"].astype(str)
    out["ea_id"]           = df["ea_id"].astype(str)
    out["wave"]            = 5
    out["region"]          = _harmonise_region(df, "saq01")
    out["rural"]           = _harmonise_rural(df, "saq14")
    out["hh_size"]         = df["hh_size"]
    out["adulteq"]         = df["adulteq"]
    out["food_cons_ann"]   = df["food_cons_ann"]
    out["nonfood_cons_ann"]= df["nonfood_cons_ann"]
    out["educ_cons_ann"]   = df["educ_cons_ann"]
    out["total_cons_ann"]  = df["total_cons_ann"]
    out["nom_totcons_aeq"] = df["nom_totcons_aeq"]
    out["cons_quint"]      = df["cons_quint"]
    out["fafh_cons_ann"]   = df.get("fafh_cons_ann", np.nan)
    out["utilities_cons_ann"] = df.get("utilities_cons_ann", np.nan)
    out["spat_totcons_aeq"]= df.get("spat_totcons_aeq", np.nan)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Cleaning helpers
# ════════════════════════════════════════════════════════════════════════════

def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the target (cons_quint) is missing."""
    before = len(df)
    df = df.dropna(subset=["cons_quint"]).copy()
    print(f"  [target] Dropped {before - len(df)} rows with missing cons_quint")
    return df


def impute_consumption_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing consumption values using wave-level medians.
    Missing food/nonfood/educ but known total_cons_ann → redistribute.
    Otherwise impute with wave-region median.
    """
    cons_cols = ["food_cons_ann", "nonfood_cons_ann", "educ_cons_ann",
                 "total_cons_ann", "nom_totcons_aeq"]

    for col in cons_cols:
        if df[col].isnull().any():
            # Compute wave × region median
            medians = df.groupby(["wave", "region"])[col].transform("median")
            filled = df[col].fillna(medians)
            # Fallback: wave-level median if region median still NaN
            wave_med = df.groupby("wave")[col].transform("median")
            df[col] = filled.fillna(wave_med)
            print(f"  [impute] {col}: imputed {df[col].isnull().sum()} remaining NaNs")

    return df


def remove_outliers(df: pd.DataFrame, col: str = "nom_totcons_aeq",
                    lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Remove extreme outliers per wave using percentile clipping.
    Default: keep 1st-99th percentile within each wave.
    """
    before = len(df)
    bounds = df.groupby("wave")[col].quantile([lower_pct, upper_pct]).unstack()
    mask = df.apply(
        lambda row: bounds.loc[row["wave"], lower_pct] <= row[col] <= bounds.loc[row["wave"], upper_pct],
        axis=1
    )
    df = df[mask].copy()
    print(f"  [outliers] Removed {before - len(df)} rows outside {lower_pct}-{upper_pct} per wave")
    return df


def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce correct dtypes."""
    df["cons_quint"] = df["cons_quint"].astype(int)
    df["wave"]       = df["wave"].astype(int)
    df["rural"]      = df["rural"].astype(int)
    df["hh_size"]    = df["hh_size"].astype(int)
    df["region"]     = df["region"].astype("category")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_cleaning(
    data_dir: str = "data/raw",
    output_path: str = "data/processed/all_waves_clean.csv",
    include_w2: bool = False,
) -> pd.DataFrame:
    """
    Load, harmonise, and clean all waves.

    Parameters
    ----------
    data_dir     : folder containing the raw data files
    output_path  : where to save the cleaned CSV

    Returns
    -------
    Cleaned, harmonised DataFrame
    """
    loaders = [
        (os.path.join(data_dir, "cons_agg_2011-12_w1.csv"), load_wave1),
        (os.path.join(data_dir, "cons_agg_2015-16_w3.csv"), load_wave3),
        (os.path.join(data_dir, "cons_agg_2018-19_w4.csv"), load_wave4),
        (os.path.join(data_dir, "cons_agg_2021-22_w5.csv"), load_wave5),
    ]
    if include_w2:
        loaders.insert(1, (os.path.join(data_dir, "cons_agg_2013-14_w2.sav"), load_wave2))

    frames = []
    for path, loader in loaders:
        if not os.path.exists(path):
            print(f"  [skip] File not found: {path}")
            continue
        print(f"\nLoading {os.path.basename(path)} ...")
        wave_df = loader(path)
        print(f"  Loaded {len(wave_df):,} rows, {wave_df.shape[1]} cols")
        frames.append(wave_df)

    print("\n── Concatenating waves ──")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined shape: {df.shape}")

    print("\n── Cleaning ──")
    df = drop_missing_target(df)
    df = impute_consumption_features(df)
    df = remove_outliers(df)
    df = validate_types(df)

    # Derived features useful for modelling
    df["food_share"]    = df["food_cons_ann"] / df["total_cons_ann"].replace(0, np.nan)
    df["cons_per_cap"]  = df["total_cons_ann"] / df["hh_size"].replace(0, np.nan)
    df["log_totcons"]   = np.log1p(df["nom_totcons_aeq"])

    print(f"\n── Final dataset: {df.shape[0]:,} rows × {df.shape[1]} cols ──")
    print(f"  cons_quint distribution:\n{df['cons_quint'].value_counts().sort_index().to_string()}")
    print(f"  Waves present: {sorted(df['wave'].unique())}")
    print(f"  Missing values remaining:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# Run directly
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = run_cleaning(
        data_dir="data/raw",
        output_path="data/processed/all_waves_clean.csv",
        include_w2=True,  
    )