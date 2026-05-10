"""
config.py
═══════════════════════════════════════════════════════════════════════════════
Ethiopian Household Wealth Prediction — Project Configuration
InSy3056 Data Science Application, Debre Berhan University

All constants, directory paths, and survey column mappings are defined here.
All other modules import from config — do not hardcode paths or column names.

CRISP-DM Reference: Phase 1 (Business Understanding) defines the project
scope and target variable. All constants here reflect that design.

Wave section layout (verified from actual ESS files):
  W1/W2/W3 : sect7=enterprise | sect8=shocks    | sect9=housing+assets
  W4/W5     : no enterprise   | sect8=housing   | sect9=shocks

W2-specific column truncations (SPSS stores max 8-char variable names):
  hh_s1q04_a → hh_s1q04  (head age)
  hh_s9q02_a → hh_s9q02  (rooms)
  hh_s3q21_a → hh_s3q21  (weeks worked)
  househ_a   → household_id (W2 primary ID)
  househol   → household_id_w1 (W1 panel link)

TARGET: cons_quint (1=poorest → 5=wealthiest)
LEAKAGE RULE: Never use *_cons_ann or *_cons_aeq columns as features —
              these quantities define cons_quint.
═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path

# ── Directory layout ───────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CLEANED_CSV = DATA_PROC / "all_waves_clean.csv"
RANKING_CSV = DATA_PROC / "regional_wealth_ranking.csv"

# ── Wave metadata ──────────────────────────────────────────────────────────────
WAVE_DIRS = {
    1: "ETH_2011_ERSS_v02_M_CSV",
    2: "ETH_2013_ESS_v03_M_SPSS",   # SPSS .sav — W2 is MANDATORY
    3: "ETH_2015_ESS_v03_M_CSV",
    4: "ETH_2018_ESS_v04_M_CSV",
    5: "ETH_2021_ESPS-W5_v02_M_CSV",
}
WAVE_YEARS = {1:"2011-12", 2:"2013-14", 3:"2015-16", 4:"2018-19", 5:"2021-22"}
WAVE_CONTEXT = {
    1: "Pre-MDG baseline",
    2: "High economic growth (~10% GDP/year)",
    3: "El Niño drought year",
    4: "Pre-COVID baseline",
    5: "Post-COVID + Tigray conflict",
}
ALL_WAVES = [1, 2, 3, 4, 5]

# ── File names per wave ────────────────────────────────────────────────────────
WAVE_FILES = {
    1: {"cons_agg":"cons_agg_w1.csv",  "roster":"sect1_hh_w1.csv",
        "labour":"sect3_hh_w1.csv",    "enterprise":"sect7_hh_w1.csv",
        "housing":"sect9_hh_w1.csv",   "shocks":"sect8_hh_w1.csv"},
    2: {"cons_agg":"cons_agg_w2.sav",  "roster":"sect1_hh_w2.sav",
        "labour":"sect3_hh_w2.sav",    "enterprise":"sect7_hh_w2.sav",
        "housing":"sect9_hh_w2.sav",   "shocks":"sect8_hh_w2.sav"},
    3: {"cons_agg":"cons_agg_w3.csv",  "roster":"sect1_hh_w3.csv",
        "labour":"sect3_hh_w3.csv",    "enterprise":"sect7_hh_w3.csv",
        "housing":"sect9_hh_w3.csv",   "shocks":"sect8_hh_w3.csv"},
    4: {"cons_agg":"cons_agg_w4.csv",  "roster":"sect1_hh_w4.csv",
        "labour":"sect3_hh_w4.csv",    "enterprise":None,
        "housing":"sect8_hh_w4.csv",   "shocks":"sect9_hh_w4.csv"},
    5: {"cons_agg":"cons_agg_w5.csv",  "roster":"sect1_hh_w5.csv",
        "labour":"sect3_hh_w5.csv",    "enterprise":None,
        "housing":"sect8_hh_w5.csv",   "shocks":"sect9_hh_w5.csv"},
}

# ── W2 SPSS column name truncation map ────────────────────────────────────────
# Applied automatically in sav_reader.read_sav() before any processing.
# Keys = truncated 8-char SPSS names; Values = standard W1/W3 names.
W2_COL_RENAME = {
    # cons_agg IDs and weights
    "househol":  "household_id_w1",   # W1 panel link (blank for new W2 HHs)
    "househ_a":  "household_id",      # W2 primary ID (always present)
    "individu":  "individual_id_w1",
    "indivi_a":  "individual_id",
    "pw2":       "pw",
    # cons_agg consumption aggregates (banned as features — leakage)
    "nom_totc":  "nom_totcons_aeq",
    "cons_qui":  "cons_quint",
    "food_con":  "food_cons_ann",
    "nonfood":   "nonfood_cons_ann",
    "educ_con":  "educ_cons_ann",
    "total_co":  "total_cons_ann",
    "price_in":  "price_index_hce",
    # housing: rooms question (10-char name → 8-char in W2)
    "hh_s9q02":  "hh_s9q02_a",
    # enterprise asset sub-questions (hh_s7q02_a..h → hh_s7q_a..h in W2)
    "hh_s7q_a":  "hh_s7q02_a",
    "hh_s7q_b":  "hh_s7q02_b",
    "hh_s7q_c":  "hh_s7q02_c",
    "hh_s7q_d":  "hh_s7q02_d",
    "hh_s7q_e":  "hh_s7q02_e",
    "hh_s7q_f":  "hh_s7q02_f",
    "hh_s7q_g":  "hh_s7q02_g",
}

# ── Region mapping (integer codes → string labels) ────────────────────────────
# W1/W2/W3: saq01 is integer | W4/W5: saq01 is already a string
REGION_MAP = {
    1:"TIGRAY",  2:"AFAR",    3:"AMHARA",        4:"OROMIA",  5:"SOMALI",
    6:"BENISHANGUL GUMUZ",    7:"SNNP",           12:"GAMBELA",
    13:"HARAR",  14:"ADDIS ABABA",                15:"DIRE DAWA",
}
SETTLEMENT_LABELS = {0:"Urban", 1:"Rural", 2:"Small town", 3:"Large town"}

# ── Survey weight column per wave ──────────────────────────────────────────────
WEIGHT_COL = {1:"pw", 2:"pw", 3:"pw_w3", 4:"pw_w4", 5:"pw_w5"}

# ── Roster (sect1) column mappings per wave ────────────────────────────────────
# relationship == 1 → household head in all waves
# W1/W2/W3: int codes (1=M,2=F for sex; 1=none..4=secondary+ for edu)
# W4/W5:    string codes ("1. Head", "1. MALE", "1. YES"/can-read)
# W2 note:  age column is hh_s1q04 (truncated from hh_s1q04_a in W1/W3)
ROSTER_COLS = {
    1: {"rel":"hh_s1q02", "sex":"hh_s1q03", "age":"hh_s1q04_a", "edu":"hh_s1q07"},
    2: {"rel":"hh_s1q02", "sex":"hh_s1q03", "age":"hh_s1q04",   "edu":"hh_s1q07"},
    3: {"rel":"hh_s1q02", "sex":"hh_s1q03", "age":"hh_s1q04a",  "edu":"hh_s1q07"},
    4: {"rel":"s1q01",    "sex":"s1q02",    "age":"s1q03a",      "edu":"s1q07"},
    5: {"rel":"s1q01",    "sex":"s1q02",    "age":"s1q03a",      "edu":"s1q07"},
}

# ── Labour (sect3) column mappings per wave ────────────────────────────────────
# worked: 1=yes/2=no (W1-W3) | "1. YES"/"2. NO" (W4/W5)
# weeks:  W1/W3=hh_s3q21_a | W2=hh_s3q21 (truncated) | W4/W5=s3q18 (earnings)
LABOUR_COLS = {
    1: {"worked":"hh_s3q18", "weeks":"hh_s3q21_a"},
    2: {"worked":"hh_s3q18", "weeks":"hh_s3q21"},
    3: {"worked":"hh_s3q18", "weeks":"hh_s3q21_a"},
    4: {"worked":"s3q14",    "weeks":"s3q18"},
    5: {"worked":"s3q14",    "weeks":"s3q18"},
}

# ── Enterprise (sect7) column mappings — W1/W2/W3 only ────────────────────────
ENTERPRISE_COLS = {
    1: {"has_ent":"hh_s7q01",
        "assets":["hh_s7q02_a","hh_s7q02_b","hh_s7q02_c","hh_s7q02_d",
                  "hh_s7q02_e","hh_s7q02_f","hh_s7q02_g","hh_s7q02_h"]},
    2: {"has_ent":"hh_s7q01",
        "assets":["hh_s7q02_a","hh_s7q02_b","hh_s7q02_c","hh_s7q02_d",
                  "hh_s7q02_e","hh_s7q02_f","hh_s7q02_g"]},
    3: {"has_ent":"hh_s7q01",
        "assets":["hh_s7q02_a","hh_s7q02_b","hh_s7q02_c","hh_s7q02_d",
                  "hh_s7q02_e","hh_s7q02_f","hh_s7q02_g","hh_s7q02_h"]},
}

# ── Housing column mappings per wave ───────────────────────────────────────────
# W1/W2/W3: from sect9 | W4/W5: from sect8
# ESS quality coding: lower ordinal code = better quality (1=best)
# Electricity: W1/W2/W3 inferred from lighting (code 1=electric grid)
HOUSING_COLS = {
    1: {"rooms":"hh_s9q02_a", "roof":"hh_s9q04",  "wall":"hh_s9q05",
        "floor":"hh_s9q06",   "water":"hh_s9q07",  "toilet":"hh_s9q08",
        "fuel":"hh_s9q10",    "lighting":"hh_s9q11",
        "owns_phone":"hh_s9q15", "owns_tv":"hh_s9q16", "owns_fridge":"hh_s9q17"},
    2: {"rooms":"hh_s9q02_a", "roof":"hh_s9q04",  "wall":"hh_s9q05",
        "floor":"hh_s9q06",   "water":"hh_s9q07",  "toilet":"hh_s9q08",
        "fuel":"hh_s9q10",    "lighting":"hh_s9q11",
        "owns_phone":"hh_s9q15", "owns_tv":"hh_s9q16", "owns_fridge":"hh_s9q17"},
    3: {"rooms":"hh_s9q02_a", "roof":"hh_s9q04",  "wall":"hh_s9q05",
        "floor":"hh_s9q06",   "water":"hh_s9q07",  "toilet":"hh_s9q10",
        "fuel":"hh_s9q13",    "lighting":None,
        "owns_phone":None, "owns_tv":"hh_s9q16_a", "owns_fridge":"hh_s9q17"},
    4: {"rooms":"s8q02a", "roof":None, "wall":None,
        "floor":"s8q03a", "water":None, "toilet":"s8q04",
        "fuel":None, "lighting":None, "electricity":"s8q06",
        "owns_phone":None, "owns_tv":None, "owns_fridge":None},
    5: {"rooms":"s8q02a", "roof":None, "wall":None,
        "floor":"s8q03a", "water":None, "toilet":"s8q04",
        "fuel":None, "lighting":None, "electricity":"s8q06",
        "owns_phone":None, "owns_tv":None, "owns_fridge":None},
}
ELEC_FROM_LIGHTING = {1, 2, 3}   # lighting==1 → electric grid in these waves

# ── Shock column mappings per wave ─────────────────────────────────────────────
# W1/W2/W3: integer code + int (1=yes,2=no) affected flag
# W4/W5:    string code + string ("1. YES"/"2. NO") affected flag
SHOCK_COLS = {
    1: {"code":"hh_s8q00", "affected":"hh_s8q01", "fmt":"int"},
    2: {"code":"hh_s8q00", "affected":"hh_s8q01", "fmt":"int"},
    3: {"code":"hh_s8q00", "affected":"hh_s8q01", "fmt":"int"},
    4: {"code":"shock_type","affected":"s9q01",    "fmt":"str"},
    5: {"code":"shock_type","affected":"s9q01",    "fmt":"str"},
}
SHOCK_PATTERNS = {
    "drought":   ["106", "DROUGHT"],
    "illness":   ["104", "ILLNESS"],
    "death":     ["101", "DEATH"],
    "crop_loss": ["108", "109", "CROP", "HARVEST", "LIVESTOCK DISEASE"],
}

# ── Target & feature groups ────────────────────────────────────────────────────
TARGET       = "cons_quint"   # 1=poorest quintile → 5=wealthiest quintile

GEO_COLS     = ["region", "settlement", "wave", "zone_id"]
DEMO_COLS    = ["hh_size", "adulteq"]
HEAD_COLS    = ["head_sex", "head_age", "head_age_sq",
                "head_edu_level", "is_female_headed", "head_literate"]
LABOUR_FEAT  = ["hh_any_wage_earner", "hh_n_workers", "hh_avg_weeks_worked"]
ENTERPRISE_F = ["has_nonfarm_enterprise", "enterprise_asset_count"]
HOUSING_FEAT = ["rooms", "roof", "wall", "floor", "water",
                "toilet", "fuel", "has_electricity", "housing_score"]
ASSET_FEAT   = ["owns_phone", "owns_tv", "owns_fridge", "asset_count"]
SHOCK_FEAT   = ["n_shocks", "experienced_drought", "experienced_illness",
                "experienced_death", "experienced_crop_loss"]
DERIVED_FEAT = ["log_hh_size", "dependency_ratio", "assets_per_member",
                "post_covid", "is_tigray_conflict",
                "has_full_housing", "has_enterprise_data"]

ALL_FEATURES = (GEO_COLS + DEMO_COLS + HEAD_COLS + LABOUR_FEAT
                + ENTERPRISE_F + HOUSING_FEAT + ASSET_FEAT
                + SHOCK_FEAT + DERIVED_FEAT)

# ── Model hyper-parameters ─────────────────────────────────────────────────────
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
RANDOM_STATE = 42
CV_FOLDS     = 5
MIN_REGION_N = 50   # minimum households for per-region modelling