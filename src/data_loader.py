"""
data_loader.py
═══════════════════════════════════════════════════════════════════════════════
Ethiopian ESS/LSMS Survey Loader — 5 waves × 6 modules per wave

Leakage-free design
───────────────────
BANNED  (define cons_quint → data leakage if used as features):
  food_cons_ann, nonfood_cons_ann, educ_cons_ann, fafh_cons_ann,
  utilities_cons_ann, total_cons_ann, nom_totcons_aeq, spat_totcons_aeq

ALLOWED from cons_agg:
  hh_size, adulteq, region, settlement, zone_id, survey_weight

TARGET  : cons_quint (kept separate, never used as a feature)

Wave 2 always included
───────────────────────
W2 (2013-14) uses SPSS .sav files decoded by sav_reader.read_sav().
W2_COL_RENAME is applied inside read_sav(), so all section loaders see
consistent column names regardless of wave format.

W2 structural gaps handled AFTER all waves are merged:
  - hh_s3q21_a (weeks) → hh_s3q21 in W2 (truncation fixed by W2_COL_RENAME)
  - hh_s9q02_a (rooms) → hh_s9q02 in W2 (truncation fixed by W2_COL_RENAME)
  - hh_s1q04_a (age)   → hh_s1q04 in W2 (handled in ROSTER_COLS[2])
═══════════════════════════════════════════════════════════════════════════════
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    ALL_WAVES, DATA_RAW, DATA_PROC, ELEC_FROM_LIGHTING,
    ENTERPRISE_COLS, HOUSING_COLS, LABOUR_COLS,
    REGION_MAP, ROSTER_COLS, SHOCK_COLS, SHOCK_PATTERNS,
    WAVE_DIRS, WAVE_FILES, WAVE_YEARS, WAVE_CONTEXT,
    WEIGHT_COL, CLEANED_CSV, TARGET, SETTLEMENT_LABELS,
)


# ════════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ════════════════════════════════════════════════════════════════════════════════

def read_file(path: Path) -> pd.DataFrame:
    """
    Read a CSV or SPSS (.sav) survey file into a DataFrame.

    For .sav files, calls sav_reader.read_sav() which:
      1. Decodes bytecode-compressed binary → raw DataFrame
      2. Applies W2_COL_RENAME (truncated names → standard names)

    Returns an empty DataFrame if the file does not exist.
    """
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".sav":
        from sav_reader import read_sav
        return read_sav(path)
    return pd.read_csv(path, low_memory=False)


def load_section(wave: int, section: str) -> pd.DataFrame:
    """Load one section file for a given wave. Returns empty DF if unavailable."""
    fname = WAVE_FILES[wave].get(section)
    if not fname:
        return pd.DataFrame()
    return read_file(DATA_RAW / WAVE_DIRS[wave] / fname)


# ════════════════════════════════════════════════════════════════════════════════
# Encoding helpers
# ════════════════════════════════════════════════════════════════════════════════

def to_binary(series: pd.Series, yes_val: int = 1) -> pd.Series:
    """
    Normalise diverse binary encodings to float 0.0 / 1.0 (NaN preserved).

    ESS uses:
      W1/W2/W3 : integers 1=yes, 2=no
      W4/W5    : strings "1. YES" / "2. NO"
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5 and series.dtype != object:
        result = (numeric == yes_val).astype(float)
        result[numeric.isna()] = np.nan
        return result
    result = series.astype(str).str.strip().str.startswith("1").astype(float)
    result[series.isna()] = np.nan
    return result


def is_head(series: pd.Series) -> pd.Series:
    """Return boolean mask for household head (relationship_to_head == 1)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return numeric == 1
    return series.astype(str).str.strip().str.startswith("1.")


def to_num(series: pd.Series) -> pd.Series:
    """Safe numeric coercion — non-numeric values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def get_hh_id(raw: pd.DataFrame) -> pd.Series:
    """
    Extract the primary household identifier for any wave.
    W2: 'household_id' after W2_COL_RENAME (original: 'househ_a').
    """
    for col in ("household_id", "househ_a", "househol"):
        if col in raw.columns:
            return raw[col].astype(str).str.strip()
    return pd.Series("", index=raw.index)


# ════════════════════════════════════════════════════════════════════════════════
# Section loaders — one function per survey module
# ════════════════════════════════════════════════════════════════════════════════

def load_geography(wave: int) -> pd.DataFrame:
    """
    Load cons_agg: geography, demographics, and target (cons_quint).

    Included (leakage-free):
      household_id, wave, region, settlement, zone_id,
      hh_size, adulteq, survey_weight, cons_quint (target)

    Banned (define cons_quint — data leakage):
      *_cons_ann, *_cons_aeq columns

    zone_id is derived from the first 4–5 digits of ea_id,
    providing a geography proxy finer than region but coarser than EA.
    """
    raw = load_section(wave, "cons_agg")
    if raw.empty:
        return raw

    # Region: integer code (W1/W2/W3) or string already (W4/W5)
    if pd.api.types.is_numeric_dtype(raw["saq01"]):
        region = raw["saq01"].map(REGION_MAP)
    else:
        region = raw["saq01"].astype(str).str.strip().str.upper()

    # Settlement: binary (rural=1/urban=0) or string-based (W4/W5)
    if "saq14" in raw.columns:
        settlement = (raw["saq14"].str.strip().str.upper() == "RURAL").astype(int)
    else:
        settlement = to_num(raw.get("rural", pd.Series(1, index=raw.index))).fillna(1).astype(int)

    # Zone ID from enumeration-area ID prefix
    ea       = raw["ea_id"].astype(str).str.strip()
    zone_id  = ea.apply(lambda s: s[:4] if len(s) <= 11 else s[:5]).astype(str).str.strip()
    zone_id  = zone_id.replace("", np.nan)
    zone_id  = zone_id.fillna(get_hh_id(raw).astype(str).str.strip().str[:4])

    out = pd.DataFrame({
        "household_id":  get_hh_id(raw),
        "wave":          int(wave),
        "region":        region,
        "settlement":    settlement,
        "zone_id":       zone_id,
        "hh_size":       to_num(raw["hh_size"]),
        "adulteq":       to_num(raw["adulteq"]),
        "survey_weight": to_num(raw[WEIGHT_COL[wave]]),
        TARGET:          to_num(raw["cons_quint"]),
    })

    # Preserve W2 panel link column for cross-wave imputation
    if wave == 2 and "household_id_w1" in raw.columns:
        out["household_id_w1"] = raw["household_id_w1"].astype(str).str.strip()

    return out


def load_head_characteristics(wave: int) -> pd.DataFrame:
    """
    From sect1 (roster): household head demographics.

    Filters to relationship == 1 (head) then extracts:
      head_sex         : 1=male, 0=female
      head_age         : age in years
      head_age_sq      : age² (non-linear life-cycle wealth effects)
      head_edu_level   : 1–4 (W1/W2/W3) or literacy binary (W4/W5)
      head_literate    : 1 if any literacy / education
      is_female_headed : 1 if head is female

    W2 note: age column is hh_s1q04 (8-char truncation of hh_s1q04_a).
    This is already accounted for in ROSTER_COLS[2] in config.py.
    """
    raw = load_section(wave, "roster")
    if raw.empty:
        return raw

    raw["_id"] = get_hh_id(raw)
    cm    = ROSTER_COLS[wave]
    heads = raw[is_head(raw[cm["rel"]])].drop_duplicates("_id").copy()
    if heads.empty:
        return pd.DataFrame()

    sex     = to_binary(heads[cm["sex"]], yes_val=1)
    age     = to_num(heads[cm["age"]])
    edu_raw = heads[cm["edu"]] if cm["edu"] in heads.columns else pd.Series(np.nan, index=heads.index)

    # W4/W5 have only a literacy proxy (can read/write yes/no)
    if wave in (4, 5):
        edu_level = to_binary(edu_raw, yes_val=1)
        literate  = edu_level
    else:
        edu_level = to_num(edu_raw)
        literate  = (edu_level >= 2).astype(float)
        literate[edu_level.isna()] = np.nan

    return pd.DataFrame({
        "household_id":     heads["_id"].values,
        "head_sex":         sex.values,
        "head_age":         age.values,
        "head_age_sq":      (age ** 2).values,
        "is_female_headed": (sex == 0).astype(float).values,
        "head_edu_level":   edu_level.values,
        "head_literate":    literate.values,
    }).reset_index(drop=True)


def load_labour(wave: int) -> pd.DataFrame:
    """
    From sect3: aggregate individual labour participation to household level.

    Features:
      hh_any_wage_earner  : 1 if ≥ 1 member worked for pay
      hh_n_workers        : count of wage earners
      hh_avg_weeks_worked : mean weeks worked (W1/W2/W3) or
                            mean monthly earnings proxy in ETB (W4/W5)

    W2 note: weeks column is hh_s3q21 (truncated from hh_s3q21_a).
    Already accounted for in LABOUR_COLS[2] in config.py.
    """
    raw = load_section(wave, "labour")
    if raw.empty:
        return raw

    raw["_id"] = get_hh_id(raw)
    cm = LABOUR_COLS[wave]

    raw["_worked"] = to_binary(raw[cm["worked"]], yes_val=1)
    raw["_weeks"]  = (
        to_num(raw[cm["weeks"]])
        if cm["weeks"] and cm["weeks"] in raw.columns
        else pd.Series(np.nan, index=raw.index)
    )

    return (
        raw.groupby("_id")
        .agg(
            hh_any_wage_earner  =("_worked", "max"),
            hh_n_workers        =("_worked", "sum"),
            hh_avg_weeks_worked =("_weeks",  "mean"),
        )
        .reset_index()
        .rename(columns={"_id": "household_id"})
    )


def load_enterprise(wave: int) -> pd.DataFrame:
    """
    From sect7 (W1/W2/W3 only): non-farm enterprise ownership.

    W4/W5 have no enterprise section → returns empty DataFrame.

    Features:
      has_nonfarm_enterprise  : 1 if household owns any non-farm business
      enterprise_asset_count  : count of business asset types owned (0–8)
    """
    if wave in (4, 5) or not WAVE_FILES[wave].get("enterprise"):
        return pd.DataFrame()

    raw = load_section(wave, "enterprise")
    if raw.empty or wave not in ENTERPRISE_COLS:
        return pd.DataFrame()

    raw["_id"] = get_hh_id(raw)
    cm = ENTERPRISE_COLS[wave]

    has_ent    = to_binary(raw[cm["has_ent"]], yes_val=1)
    asset_cols = [c for c in cm["assets"] if c in raw.columns]
    ent_assets = (
        raw[asset_cols].apply(to_num).fillna(0).gt(0).sum(axis=1)
        if asset_cols else pd.Series(0, index=raw.index)
    )

    return pd.DataFrame({
        "household_id":           raw["_id"].values,
        "has_nonfarm_enterprise": has_ent.values,
        "enterprise_asset_count": ent_assets.values,
    }).drop_duplicates("household_id").reset_index(drop=True)


def load_housing(wave: int) -> pd.DataFrame:
    """
    From sect9 (W1/W2/W3) or sect8 (W4/W5): dwelling quality + asset ownership.

    Ordinal features (ESS: lower code = better quality):
      roof, wall, floor, water, toilet, fuel
    Binary: has_electricity, owns_phone, owns_tv, owns_fridge
    Composite:
      housing_score : normalised 0–1 quality index (higher = better dwelling)
      asset_count   : count of owned assets (phone + TV + fridge)

    Note: W4/W5 sect8 has fewer columns (no roof/wall/water/fuel questions).
    These features are NaN for W4/W5 rows — handled downstream in imputer.
    """
    raw = load_section(wave, "housing")
    if raw.empty:
        return raw

    raw["_id"] = get_hh_id(raw)
    cm = HOUSING_COLS[wave]

    def _col(key: str) -> pd.Series:
        c = cm.get(key)
        return to_num(raw[c]) if c and c in raw.columns else pd.Series(np.nan, index=raw.index)

    out = pd.DataFrame({"household_id": raw["_id"].values})
    for feat in ("rooms", "roof", "wall", "floor", "water", "toilet", "fuel"):
        out[feat] = _col(feat).values

    # Electricity: direct (W4/W5) or from lighting code (W1/W2/W3)
    if cm.get("electricity") and cm["electricity"] in raw.columns:
        out["has_electricity"] = to_binary(raw[cm["electricity"]], yes_val=1).values
    elif cm.get("lighting") and cm["lighting"] in raw.columns and wave in ELEC_FROM_LIGHTING:
        out["has_electricity"] = (_col("lighting") == 1).astype(float).values
    else:
        out["has_electricity"] = np.nan

    # Consumer assets (W1/W2/W3 only; W4/W5 sect8 lacks these questions)
    for asset in ("owns_phone", "owns_tv", "owns_fridge"):
        c = cm.get(asset)
        out[asset] = (
            to_binary(raw[c], yes_val=1).values
            if c and c in raw.columns else np.nan
        )
    out["asset_count"] = out[["owns_phone", "owns_tv", "owns_fridge"]].fillna(0).sum(axis=1)

    # Housing quality score: normalise ordinal components, then average
    # Lower ESS code = better for most materials; invert = True for those
    score_parts = []
    for col, invert in [("roof",True),("wall",True),("floor",False),
                        ("water",True),("toilet",True),("fuel",True)]:
        vals = out[col]
        if vals.notna().sum() > 10:
            mn, mx = vals.min(), vals.max()
            if mx > mn:
                norm = (vals - mn) / (mx - mn)
                score_parts.append(1 - norm if invert else norm)
    if out["has_electricity"].notna().any():
        score_parts.append(out["has_electricity"])
    if out["rooms"].notna().any():
        score_parts.append((out["rooms"].clip(1, 10) / 10).clip(0, 1))
    if out["asset_count"].notna().any():
        score_parts.append((out["asset_count"] / 3).clip(0, 1))

    out["housing_score"] = (
        pd.concat(score_parts, axis=1).mean(axis=1)
        if score_parts else np.nan
    )

    return out.drop_duplicates("household_id").reset_index(drop=True)


def load_shocks(wave: int) -> pd.DataFrame:
    """
    From sect8 (W1/W2/W3) or sect9 (W4/W5): household shock exposure.
    Pivots long format (N shock types × HH) → one row per household.

    Features:
      n_shocks               : total distinct shocks experienced
      experienced_drought    : 1 if drought/rainfall failure
      experienced_illness    : 1 if member illness/disease
      experienced_death      : 1 if death of main income earner
      experienced_crop_loss  : 1 if crop failure or livestock disease
    """
    raw = load_section(wave, "shocks")
    if raw.empty:
        return raw

    raw["_id"] = get_hh_id(raw)
    cm = SHOCK_COLS[wave]

    # Convert to bool safely: NaN -> False to avoid bitwise errors in aggregation.
    raw["_affected"] = to_binary(raw[cm["affected"]], yes_val=1).fillna(0).astype(bool)
    raw["_code"] = raw[cm["code"]].astype(str).str.strip().str.upper()

    for shock, patterns in SHOCK_PATTERNS.items():
        raw[f"_is_{shock}"] = raw["_code"].str.contains("|".join(patterns), na=False)

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_shocks":              int(g["_affected"].sum()),
            "experienced_drought":   int((g["_affected"] & g["_is_drought"]).any()),
            "experienced_illness":   int((g["_affected"] & g["_is_illness"]).any()),
            "experienced_death":     int((g["_affected"] & g["_is_death"]).any()),
            "experienced_crop_loss": int((g["_affected"] & g["_is_crop_loss"]).any()),
        })

    return (
        raw.groupby("_id").apply(_agg).reset_index()
        .rename(columns={"_id": "household_id"})
    )


# ════════════════════════════════════════════════════════════════════════════════
# Wave assembler
# ════════════════════════════════════════════════════════════════════════════════

def build_wave(wave: int, verbose: bool = True) -> pd.DataFrame:
    """
    Build a complete household-level dataset for one wave by merging all modules.

    Merge order: geography → head → labour → enterprise → housing → shocks
    All joins are LEFT on household_id; missing modules contribute NaN columns
    so the downstream MissingValueHandler can fill them appropriately.

    Derived context features added here:
      log_hh_size, dependency_ratio, assets_per_member
      post_covid, is_tigray_conflict, has_full_housing, has_enterprise_data

    Returns ~38 leakage-free features + cons_quint target.
    """
    if verbose:
        print(f"\n── Wave {wave} ({WAVE_YEARS.get(wave,'')}) ─────────────────")

    base = load_geography(wave)
    if base.empty:
        if verbose:
            print(f"  [W{wave}] cons_agg not found — skipping wave")
        return base

    loaders = [
        ("head",       load_head_characteristics),
        ("labour",     load_labour),
        ("enterprise", load_enterprise),
        ("housing",    load_housing),
        ("shocks",     load_shocks),
    ]
    for name, fn in loaders:
        try:
            other = fn(wave)
            if not other.empty and "household_id" in other.columns:
                dup = [c for c in other.columns
                       if c in base.columns and c != "household_id"]
                base = base.merge(other.drop(columns=dup),
                                  on="household_id", how="left")
                if verbose:
                    print(f"  [W{wave}] {name:12s} merged  ({len(other)} rows)")
        except Exception as exc:
            if verbose:
                print(f"  [W{wave}] {name:12s} skipped ({exc})")

    # Derived features
    base["log_hh_size"]       = np.log1p(base["hh_size"])
    base["dependency_ratio"]  = (
        (base["hh_size"] - base["adulteq"]).clip(lower=0)
        / base["hh_size"].replace(0, np.nan)
    ).fillna(0)
    base["assets_per_member"] = (
        base.get("asset_count", pd.Series(0, index=base.index))
        / base["hh_size"].replace(0, np.nan)
    ).fillna(0)
    base["post_covid"]         = int(wave == 5)
    base["is_tigray_conflict"] = (
        (base["region"].astype(str) == "TIGRAY") & (wave == 5)
    ).astype(int)
    base["has_full_housing"]   = int(wave in (1, 2, 3))
    base["has_enterprise_data"]= int(wave in (1, 2, 3))

    if verbose:
        pct = base.isnull().mean().mean() * 100
        print(f"  [W{wave}] ✓ {len(base):,} households | "
              f"mean missing: {pct:.1f}% | cols: {len(base.columns)}")
    return base


def build_all_waves(
    waves: list = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build and stack all survey waves into one combined DataFrame.

    Wave 2 is ALWAYS included — it is not optional.

    Parameters
    ----------
    waves   : subset of [1,2,3,4,5]. None = all 5. W2 forced if omitted.
    save    : write combined DataFrame to data/processed/all_waves_clean.csv
    verbose : print loading progress per wave

    Returns
    -------
    pd.DataFrame — one row per household, all selected waves concatenated.
    Ready for MissingValueHandler → FeatureEngineer → DataPreprocessor.
    """
    selected = sorted(set((waves or ALL_WAVES) + [2]))   # W2 always present

    if verbose:
        print("═" * 60)
        print(f"Ethiopian Household Wealth — Waves: {selected}")
        print("═" * 60)

    frames = [build_wave(w, verbose=verbose) for w in selected]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise RuntimeError("No data loaded. Check DATA_RAW paths in config.py.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["region"] = combined["region"].astype("category")

    if verbose:
        print(f"\n✓ Combined: {len(combined):,} households | "
              f"{combined['wave'].nunique()} waves | "
              f"{len(combined.columns)} columns")

    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        combined.to_csv(CLEANED_CSV, index=False)
        if verbose:
            print(f"  Saved → {CLEANED_CSV}")

    return combined


# ════════════════════════════════════════════════════════════════════════════════
# WaveExplorer — for individual-wave and cross-wave notebook exploration
# ════════════════════════════════════════════════════════════════════════════════

class WaveExplorer:
    """
    Interactive wave-level explorer for notebooks.

    Supports:
      - Loading a single wave for per-wave inspection
      - Cross-wave comparison of any feature
      - W2 gap report (shows which columns are missing in W2 and why)
      - Regional wealth summaries

    Usage
    -----
    # Explore a single wave on the fly
    explorer = WaveExplorer(wave=2)
    explorer.summary()
    explorer.w2_gap_report()

    # Explore the combined dataset
    df = build_all_waves()
    explorer = WaveExplorer(df=df)
    explorer.compare_waves("housing_score")
    explorer.wealth_by_region()
    """

    def __init__(self, df: pd.DataFrame = None, wave: int = None):
        if df is not None:
            self.df   = df
            self.wave = None
        elif wave is not None:
            if wave not in range(1, 6):
                raise ValueError(f"wave must be 1–5, got {wave}")
            print(f"Loading wave {wave} ({WAVE_YEARS.get(wave, '')}) …")
            self.df   = build_wave(wave, verbose=True)
            self.wave = wave
        else:
            raise ValueError("Provide df= (combined) or wave= (single wave).")

    def summary(self) -> pd.DataFrame:
        """Per-wave summary: households, rural %, missing %, mean quintile."""
        rows = []
        for w in sorted(self.df["wave"].unique()):
            sub = self.df[self.df["wave"] == w]
            rows.append({
                "wave":          int(w),
                "context":       WAVE_CONTEXT.get(w, ""),
                "n_households":  len(sub),
                "pct_rural":     round(sub["settlement"].mean() * 100, 1),
                "pct_missing":   round(sub.isnull().mean().mean() * 100, 2),
                "mean_quintile": round(sub[TARGET].mean(), 2),
            })
        result = pd.DataFrame(rows).set_index("wave")
        print(result.to_string())
        return result

    def missing_report(self, wave: int = None, threshold: float = 0.0) -> pd.DataFrame:
        """Missing value report per feature, optionally filtered to one wave."""
        sub  = self.df if wave is None else self.df[self.df["wave"] == wave]
        miss = sub.isnull().sum()
        pct  = (miss / len(sub) * 100).round(2)
        out  = pd.DataFrame({"n_missing": miss, "pct_missing": pct})
        return out[out["pct_missing"] > threshold].sort_values("pct_missing", ascending=False)

    def w2_gap_report(self) -> pd.DataFrame:
        """
        Compare null rates of structurally-sparse W2 columns vs other waves.
        Explains root cause (SPSS 8-char truncation vs genuinely absent).
        """
        gap_causes = {
            "head_age":           "hh_s1q04_a (10ch) → hh_s1q04 in W2 SAV [ROSTER_COLS fix]",
            "head_age_sq":        "Derived from head_age",
            "is_female_headed":   "Derived from head_sex",
            "head_literate":      "Derived from head_edu_level",
            "rooms":              "hh_s9q02_a (10ch) → hh_s9q02 in W2 SAV [W2_COL_RENAME fix]",
            "hh_avg_weeks_worked":"hh_s3q21_a (10ch) → hh_s3q21 in W2 SAV [LABOUR_COLS fix]",
            "zone_id":            "Derived from ea_id — always available",
        }
        w2     = self.df[self.df["wave"] == 2]
        non_w2 = self.df[self.df["wave"] != 2]
        rows   = []
        for col, cause in gap_causes.items():
            if col not in self.df.columns:
                continue
            w2_pct  = w2[col].isna().mean() * 100
            oth_pct = non_w2[col].isna().mean() * 100
            rows.append({
                "feature":       col,
                "W2_null%":      round(w2_pct, 1),
                "other_null%":   round(oth_pct, 1),
                "gap_pp":        round(w2_pct - oth_pct, 1),
                "root_cause":    cause,
            })
        result = pd.DataFrame(rows).sort_values("gap_pp", ascending=False)
        print("\n── W2 Gap Report ──────────────────────────────────────────")
        print(result.to_string(index=False))
        return result

    def compare_waves(self, feature: str, stat: str = "mean") -> pd.DataFrame:
        """Compare one feature across waves by a summary statistic."""
        if feature not in self.df.columns:
            raise ValueError(f"'{feature}' not in DataFrame.")
        rows = []
        for w in sorted(self.df["wave"].unique()):
            sub = self.df[self.df["wave"] == w][feature]
            val = sub.isna().mean() * 100 if stat == "pct_missing" else getattr(sub, stat)()
            rows.append({"wave": int(w), "context": WAVE_CONTEXT.get(w, ""), stat: round(val, 3)})
        result = pd.DataFrame(rows).set_index("wave")
        print(f"\n── {feature} ({stat}) by wave ──")
        print(result.to_string())
        return result

    def wealth_by_region(self, wave: int = None) -> pd.DataFrame:
        """Mean quintile per region, optionally for one wave only."""
        sub = self.df if wave is None else self.df[self.df["wave"] == wave]
        result = (
            sub.groupby("region", observed=True)[TARGET]
            .agg(mean_quintile="mean", n_households="count")
            .round(3).sort_values("mean_quintile", ascending=False)
        )
        print(f"\n── Wealth by region ({'all waves' if wave is None else f'Wave {wave}'}) ──")
        print(result.to_string())
        return result

    def get_wave(self, wave: int) -> pd.DataFrame:
        """Return all rows for a specific wave."""
        out = self.df[self.df["wave"] == wave].copy()
        print(f"Wave {wave}: {len(out):,} households | "
              f"missing: {out.isnull().mean().mean()*100:.1f}%")
        return out

    def feature_coverage(self, wave: int = None) -> pd.DataFrame:
        """% non-null per feature for a given wave (or all waves)."""
        sub = self.df if wave is None else self.df[self.df["wave"] == wave]
        pct = (sub.notna().mean() * 100).round(1).reset_index()
        pct.columns = ["feature", "pct_non_null"]
        return pct.sort_values("pct_non_null").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
# Notebook utility functions
# ════════════════════════════════════════════════════════════════════════════════

def explore_wave(wave: int) -> dict:
    """
    Load all raw sections for a single wave and return as a dict.
    Useful for notebook inspection of raw questionnaire columns before merging.

    Returns
    -------
    dict keys: 'cons_agg', 'roster', 'labour', 'enterprise',
               'housing', 'shocks', 'merged'
    """
    result = {sec: load_section(wave, sec)
              for sec in ("cons_agg","roster","labour","enterprise","housing","shocks")}
    result["merged"] = build_wave(wave, verbose=False)
    return result


def wave_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-wave summary DataFrame for notebook display."""
    rows = []
    for w in sorted(df["wave"].unique()):
        sub = df[df["wave"] == w]
        rows.append({
            "wave":          int(w),
            "period":        WAVE_YEARS.get(w, ""),
            "context":       WAVE_CONTEXT.get(w, ""),
            "n_households":  len(sub),
            "pct_missing":   round(sub.isnull().mean().mean() * 100, 1),
            "mean_quintile": round(sub[TARGET].mean(), 2),
        })
    return pd.DataFrame(rows)


def feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """% non-null per feature, sorted descending."""
    pct = (df.notna().mean() * 100).round(1).reset_index()
    pct.columns = ["feature", "pct_non_null"]
    return pct.sort_values("pct_non_null", ascending=False).reset_index(drop=True)


def coverage_by_wave(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature coverage (% non-null) broken down per wave.
    Reveals which features are structurally absent in which waves.
    """
    result = {}
    for w in sorted(df["wave"].unique()):
        sub = df[df["wave"] == w]
        result[f"W{w}_{WAVE_YEARS.get(w,'')}"] = (sub.notna().mean() * 100).round(1)
    return pd.DataFrame(result).T
