"""
data_loader.py — EthiopianSurveyLoader
Loads and merges all ESS/LSMS survey sections into one leakage-free dataset.

LEAKAGE RULE — from cons_agg, ONLY these columns are kept:
  hh_size, adulteq, region, settlement, zone_id, survey_weight, cons_quint
  All *_cons_ann and *_cons_aeq columns are BANNED (they define cons_quint).

One row per household after merging:
  geography  → from cons_agg
  head chars → from sect1 (filtered to relationship==1)
  labour     → from sect3 (aggregated to household level)
  enterprise → from sect7 (W1/W2/W3 only)
  housing    → from sect9 (W1/W2/W3) or sect8 (W4/W5)
  shocks     → from sect8 (W1/W2/W3) or sect9 (W4/W5), pivoted to wide
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    DATA_RAW, DATA_PROC, ENTERPRISE_COLS, HOUSING_COLS, LABOUR_COLS,
    REGION_MAP, ROSTER_COLS, SHOCK_COLS, SHOCK_PATTERNS,
    WAVE_DIRS, WAVE_FILES, WEIGHT_COL, ELEC_FROM_LIGHTING_WAVES,
    CLEANED_CSV,
)


# ════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ════════════════════════════════════════════════════════════════════════════

def _read(path: Path) -> pd.DataFrame:
    """Read CSV or SPSS (.sav). Returns empty DataFrame if file missing."""
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".sav":
        try:
            import pyreadstat
            df, _ = pyreadstat.read_sav(str(path))
            return df
        except ImportError:
            raise ImportError("pip install pyreadstat  to read Wave 2 .sav files")
    return pd.read_csv(path, low_memory=False)


def _load(wave: int, section: str) -> pd.DataFrame:
    fname = WAVE_FILES[wave].get(section)
    if not fname:
        return pd.DataFrame()
    return _read(DATA_RAW / WAVE_DIRS[wave] / fname)


# ── Coding helpers ────────────────────────────────────────────────────────────

def _to_binary(series: pd.Series, yes_val=1) -> pd.Series:
    """
    Normalise 1/2 integer or '1. YES'/'2. NO' string → float 1.0/0.0.
    Preserves NaN.
    """
    s = series.copy()
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.5 and s.dtype != object:
        # Integer coded: 1=yes, 2=no
        result = (numeric == yes_val).astype(float)
        result[numeric.isna()] = np.nan
        return result
    # String coded: starts with "1" = yes
    str_s  = s.astype(str).str.strip()
    result = str_s.str.startswith("1").astype(float)
    result[s.isna()] = np.nan
    return result


def _is_head(series: pd.Series) -> pd.Series:
    """Return boolean mask for household head (relationship == 1)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return numeric == 1
    return series.astype(str).str.strip().str.startswith("1.")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ════════════════════════════════════════════════════════════════════════════
# Module loaders
# ════════════════════════════════════════════════════════════════════════════

def load_geography(wave: int) -> pd.DataFrame:
    """
    From cons_agg: geography + demographics + target.
    BANNED columns: all *_cons_ann, *_cons_aeq, price_index_hce.
    ALLOWED: hh_size, adulteq, region, settlement, zone_id, cons_quint.
    """
    raw = _load(wave, "cons_agg")
    if raw.empty:
        return raw

    # Region
    if pd.api.types.is_numeric_dtype(raw["saq01"]):
        region = raw["saq01"].map(REGION_MAP)
    else:
        region = raw["saq01"].astype(str).str.strip().str.upper()

    # Settlement
    if "saq14" in raw.columns:
        settlement = (raw["saq14"].str.strip().str.upper() == "RURAL").astype(int)
    else:
        settlement = raw["rural"].astype(int)  # 0/1 in W1, 1/2/3 in W3

    # Zone (geographic sub-unit from ea_id prefix)
    ea = raw["ea_id"].astype(str).str.strip()
    zone_id = ea.apply(lambda s: s[:4] if len(s) <= 11 else s[:5])

    return pd.DataFrame({
        "household_id":  raw["household_id"].astype(str).str.strip(),
        "wave":          int(wave),
        "region":        region,
        "settlement":    settlement,
        "zone_id":       zone_id,
        "hh_size":       _to_numeric(raw["hh_size"]),
        "adulteq":       _to_numeric(raw["adulteq"]),
        "survey_weight": _to_numeric(raw[WEIGHT_COL[wave]]),
        TARGET:          _to_numeric(raw["cons_quint"]),
    })


def load_head_characteristics(wave: int) -> pd.DataFrame:
    """
    From sect1: household head demographics and education.

    head_sex         : 1=male, 0=female
    head_age         : age in years
    head_age_sq      : age squared (captures life-cycle effects)
    head_edu_level   : W1/W3: 1=none,2=read-write,3=primary,4=secondary+
                       W4/W5: binary literacy (1=can read, 0=cannot)
    head_literate    : 1 if any education
    is_female_headed : 1 if head is female
    """
    raw = _load(wave, "roster")
    if raw.empty:
        return raw

    raw["household_id"] = raw["household_id"].astype(str).str.strip()
    cm  = ROSTER_COLS[wave]
    heads = raw[_is_head(raw[cm["rel"]])].drop_duplicates("household_id").copy()

    sex     = _to_binary(heads[cm["sex"]], yes_val=1)   # 1=male, 2=female → 1/0
    age     = _to_numeric(heads[cm["age"]])
    edu_raw = heads[cm["edu"]]

    if wave in (4, 5):
        # s1q07: "1. YES" can read → literacy proxy
        edu_level = _to_binary(edu_raw, yes_val=1)
        literate  = edu_level
    else:
        edu_level = _to_numeric(edu_raw)
        literate  = (edu_level >= 2).astype(float)
        literate[edu_level.isna()] = np.nan

    return pd.DataFrame({
        "household_id":   heads["household_id"].values,
        "head_sex":       sex.values,
        "head_age":       age.values,
        "head_age_sq":    (age ** 2).values,
        "is_female_headed": (sex == 0).astype(float).values,
        "head_edu_level": edu_level.values,
        "head_literate":  literate.values,
    }).reset_index(drop=True)


def load_labour(wave: int) -> pd.DataFrame:
    """
    From sect3: labour market participation (individual → household aggregate).

    hh_any_wage_earner : 1 if ≥1 member worked for pay
    hh_n_workers       : count of wage earners
    hh_avg_weeks_worked: mean weeks worked (W1/W3) or mean earnings (W4/W5)
    """
    raw = _load(wave, "labour")
    if raw.empty:
        return raw

    raw["household_id"] = raw["household_id"].astype(str).str.strip()
    cm = LABOUR_COLS[wave]

    raw["_worked"] = _to_binary(raw[cm["worked"]], yes_val=1)

    weeks_col = cm["weeks"]
    if weeks_col and weeks_col in raw.columns:
        raw["_weeks"] = _to_numeric(raw[weeks_col])
    else:
        raw["_weeks"] = np.nan

    return (
        raw.groupby("household_id")
        .agg(
            hh_any_wage_earner=("_worked", "max"),
            hh_n_workers=("_worked", "sum"),
            hh_avg_weeks_worked=("_weeks", "mean"),
        )
        .reset_index()
    )


def load_enterprise(wave: int) -> pd.DataFrame:
    """
    From sect7 (W1/W2/W3 only): non-farm enterprise ownership.

    has_nonfarm_enterprise : 1 if household owns any non-farm business
    enterprise_asset_count : number of enterprise asset types owned (0–8)

    Returns empty DataFrame for W4/W5.
    """
    if wave in (4, 5) or WAVE_FILES[wave].get("enterprise") is None:
        return pd.DataFrame()

    raw = _load(wave, "enterprise")
    if raw.empty:
        return raw

    raw["household_id"] = raw["household_id"].astype(str).str.strip()
    cm = ENTERPRISE_COLS[wave]

    has_ent = _to_binary(raw[cm["has_ent"]], yes_val=1)

    asset_cols = [c for c in cm["assets"] if c in raw.columns]
    ent_assets = (
        raw[asset_cols].apply(_to_numeric).fillna(0).gt(0).sum(axis=1)
        if asset_cols else pd.Series(0, index=raw.index)
    )

    return pd.DataFrame({
        "household_id":          raw["household_id"].values,
        "has_nonfarm_enterprise": has_ent.values,
        "enterprise_asset_count":ent_assets.values,
    }).drop_duplicates("household_id").reset_index(drop=True)


def load_housing(wave: int) -> pd.DataFrame:
    """
    From sect9 (W1/W2/W3) or sect8 (W4/W5): dwelling quality + asset ownership.

    Ordinal features (kept as raw codes for ordinal encoding downstream):
      roof, wall, floor, water, toilet, fuel
    Binary:
      has_electricity, owns_phone, owns_tv, owns_fridge
    Composite:
      housing_score = normalised 0–1 index from available components
      asset_count   = sum of owns_phone + owns_tv + owns_fridge

    ESS coding note — lower code generally = better quality:
      roof:  1=iron sheets, 4=grass/leaves
      floor: 1=earth/mud, 5=tiles
      water: 1=piped into dwelling, 6+=surface water
      toilet:1=flush, 5=open field
    """
    raw = _load(wave, "housing")
    if raw.empty:
        return raw

    raw["household_id"] = raw["household_id"].astype(str).str.strip()
    cm = HOUSING_COLS[wave]

    def _col(key):
        c = cm.get(key)
        return _to_numeric(raw[c]) if c and c in raw.columns else pd.Series(np.nan, index=raw.index)

    out = pd.DataFrame({"household_id": raw["household_id"].values})
    out["rooms"]  = _col("rooms")
    out["roof"]   = _col("roof")
    out["wall"]   = _col("wall")
    out["floor"]  = _col("floor")
    out["water"]  = _col("water")
    out["toilet"] = _col("toilet")
    out["fuel"]   = _col("fuel")

    # Electricity
    if cm.get("electricity") and cm["electricity"] in raw.columns:
        out["has_electricity"] = _to_binary(raw[cm["electricity"]], yes_val=1).values
    elif cm.get("lighting") and cm["lighting"] in raw.columns and wave in ELEC_FROM_LIGHTING_WAVES:
        out["has_electricity"] = (_to_numeric(raw[cm["lighting"]]) == 1).astype(float).values
    else:
        out["has_electricity"] = np.nan

    # Asset ownership (W1/W2/W3 only — cols are None for W4/W5)
    for asset in ("owns_phone", "owns_tv", "owns_fridge"):
        c = cm.get(asset)
        if c and c in raw.columns:
            out[asset] = _to_binary(raw[c], yes_val=1).values
        else:
            out[asset] = np.nan

    out["asset_count"] = out[["owns_phone","owns_tv","owns_fridge"]].fillna(0).sum(axis=1)

    # Housing score: normalise each ordinal component (invert so higher=better),
    # then average with electricity and rooms
    score_parts = []

    for col, invert in [("roof",True),("wall",True),("floor",False),
                        ("water",True),("toilet",True),("fuel",True)]:
        vals = out[col]
        valid = vals.dropna()
        if len(valid) > 10:
            mn, mx = valid.min(), valid.max()
            if mx > mn:
                norm = (vals - mn) / (mx - mn)
                score_parts.append(1 - norm if invert else norm)

    if out["has_electricity"].notna().any():
        score_parts.append(out["has_electricity"])
    if out["rooms"].notna().any():
        score_parts.append((out["rooms"].clip(1, 10) / 10).clip(0, 1))
    if out["asset_count"].notna().any():
        score_parts.append((out["asset_count"] / 3).clip(0, 1))

    if score_parts:
        out["housing_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    else:
        out["housing_score"] = np.nan

    return out.drop_duplicates("household_id").reset_index(drop=True)


def load_shocks(wave: int) -> pd.DataFrame:
    """
    From sect8 (W1/W2/W3) or sect9 (W4/W5): shock exposure.
    Long format → pivoted to one row per household.

    n_shocks               : total shocks experienced (0–18/20)
    experienced_drought    : 1 if affected by drought
    experienced_illness    : 1 if affected by member illness
    experienced_death      : 1 if death of main income earner
    experienced_crop_loss  : 1 if crop failure or livestock disease
    """
    raw = _load(wave, "shocks")
    if raw.empty:
        return raw

    raw["household_id"] = raw["household_id"].astype(str).str.strip()
    cm = SHOCK_COLS[wave]

    raw["_affected"] = _to_binary(raw[cm["affected"]], yes_val=1)
    raw["_code"]     = raw[cm["code"]].astype(str).str.strip().str.upper()

    for shock, patterns in SHOCK_PATTERNS.items():
        pattern = "|".join(patterns)
        raw[f"_is_{shock}"] = raw["_code"].str.contains(pattern, na=False)

    def _agg(g):
        return pd.Series({
            "n_shocks":             g["_affected"].sum(),
            "experienced_drought":  int((g["_affected"] & g["_is_drought"]).any()),
            "experienced_illness":  int((g["_affected"] & g["_is_illness"]).any()),
            "experienced_death":    int((g["_affected"] & g["_is_death"]).any()),
            "experienced_crop_loss":int((g["_affected"] & g["_is_crop_loss"]).any()),
        })

    return raw.groupby("household_id").apply(_agg).reset_index()


# ════════════════════════════════════════════════════════════════════════════
# Wave assembler
# ════════════════════════════════════════════════════════════════════════════

TARGET = "cons_quint"


def build_wave(wave: int) -> pd.DataFrame:
    """
    Merge all sections for one wave into a single household-level DataFrame.
    All joins are LEFT on household_id; missing modules contribute NaN columns.

    Returns ~35 leakage-free features + cons_quint target.
    """
    base = load_geography(wave)
    if base.empty:
        return base

    for loader_fn in [
        load_head_characteristics,
        load_labour,
        load_enterprise,
        load_housing,
        load_shocks,
    ]:
        try:
            other = loader_fn(wave)
            if not other.empty and "household_id" in other.columns:
                # Avoid duplicate columns (keep base version)
                dup = [c for c in other.columns
                       if c in base.columns and c != "household_id"]
                base = base.merge(
                    other.drop(columns=dup), on="household_id", how="left"
                )
        except Exception:
            pass   # module unavailable — shows as NaN in notebook

    # Derived features
    base["log_hh_size"] = np.log1p(base["hh_size"])
    base["dependency_ratio"] = (
        (base["hh_size"] - base["adulteq"]).clip(lower=0)
        / base["hh_size"].replace(0, np.nan)
    ).fillna(0)
    base["assets_per_member"] = (
        base.get("asset_count", pd.Series(0, index=base.index))
        / base["hh_size"].replace(0, np.nan)
    )
    base["post_covid"]         = int(wave == 5)
    base["is_tigray_conflict"] = (
        (base["region"].astype(str) == "TIGRAY") & (wave == 5)
    ).astype(int)

    return base


def build_all_waves(
    waves: list | None = None,
    include_w2: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build and stack all wave datasets into one clean DataFrame.

    Parameters
    ----------
    waves      : wave numbers to include (default [1,3,4,5])
    include_w2 : set True after: pip install pyreadstat
    save       : if True, write to data/processed/all_waves_clean.csv

    Returns
    -------
    pd.DataFrame — one row per household, all waves stacked
    """
    if waves is None:
        waves = [1, 2, 3, 4, 5] if include_w2 else [1, 3, 4, 5]

    frames = [build_wave(w) for w in sorted(waves)]
    frames = [f for f in frames if not f.empty]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["region"] = combined["region"].astype("category")

    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        combined.to_csv(CLEANED_CSV, index=False)

    return combined


# ── Notebook utilities ────────────────────────────────────────────────────────

def wave_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table per wave for notebook display."""
    rows = []
    for w in sorted(df["wave"].unique()):
        sub = df[df["wave"] == w]
        rows.append({
            "wave":          int(w),
            "period":        {1:"2011-12",3:"2015-16",4:"2018-19",5:"2021-22"}.get(w,""),
            "n_households":  len(sub),
            "pct_missing":   round(sub.isnull().mean().mean() * 100, 1),
            "mean_quintile": round(sub[TARGET].mean(), 2),
        })
    return pd.DataFrame(rows)


def feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """% non-null per column — for notebook display."""
    pct = (df.notna().mean() * 100).round(1).reset_index()
    pct.columns = ["feature", "pct_non_null"]
    return pct.sort_values("pct_non_null", ascending=False).reset_index(drop=True)