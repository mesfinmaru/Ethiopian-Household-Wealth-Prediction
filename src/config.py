"""
config.py — Ethiopian Household Wealth Prediction
All column names verified from actual ESS wave files.

Section layout differs between waves:
  W1 / W3:  sect8 = shocks (long, 18 types)   | sect9 = housing + assets
  W4 / W5:  sect8 = housing (wide, 1 row/HH)  | sect9 = shocks (long, 20 types)
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR= ROOT / "reports"

CLEANED_CSV = DATA_PROC / "all_waves_clean.csv"
RANKING_CSV = DATA_PROC / "regional_wealth_ranking.csv"

# ── Wave metadata ─────────────────────────────────────────────────────────────
WAVE_DIRS = {
    1: "ETH_2011_ERSS_v02_M_CSV",
    2: "ETH_2013_ESS_v03_M_SPSS",
    3: "ETH_2015_ESS_v03_M_CSV",
    4: "ETH_2018_ESS_v04_M_CSV",
    5: "ETH_2021_ESPS-W5_v02_M_CSV",
}
WAVE_YEARS   = {1:"2011-12", 2:"2013-14", 3:"2015-16", 4:"2018-19", 5:"2021-22"}
WAVE_CONTEXT = {
    1: "Pre-MDG baseline",
    2: "High growth period",
    3: "El Niño drought",
    4: "Pre-COVID baseline",
    5: "Post-COVID + Tigray conflict",
}

# ── File names per wave ───────────────────────────────────────────────────────
WAVE_FILES = {
    1: {"cons_agg":  "cons_agg_w1.csv",
        "roster":    "sect1_hh_w1.csv",
        "labour":    "sect3_hh_w1.csv",
        "enterprise":"sect7_hh_w1.csv",
        "housing":   "sect9_hh_w1.csv",   # W1/W3: sect9 = housing+assets
        "shocks":    "sect8_hh_w1.csv"},  # W1/W3: sect8 = shocks
    2: {"cons_agg":  "cons_agg_w2.sav",
        "roster":    "sect1_hh_w2.sav",
        "labour":    "sect3_hh_w2.sav",
        "enterprise":"sect7_hh_w2.sav",
        "housing":   "sect9_hh_w2.sav",
        "shocks":    "sect8_hh_w2.sav"},
    3: {"cons_agg":  "cons_agg_w3.csv",
        "roster":    "sect1_hh_w3.csv",
        "labour":    "sect3_hh_w3.csv",
        "enterprise":"sect7_hh_w3.csv",
        "housing":   "sect9_hh_w3.csv",
        "shocks":    "sect8_hh_w3.csv"},
    4: {"cons_agg":  "cons_agg_w4.csv",
        "roster":    "sect1_hh_w4.csv",
        "labour":    "sect3_hh_w4.csv",
        "enterprise": None,               # W4/W5 have no enterprise section
        "housing":   "sect8_hh_w4.csv",  # W4/W5: sect8 = housing
        "shocks":    "sect9_hh_w4.csv"}, # W4/W5: sect9 = shocks
    5: {"cons_agg":  "cons_agg_w5.csv",
        "roster":    "sect1_hh_w5.csv",
        "labour":    "sect3_hh_w5.csv",
        "enterprise": None,
        "housing":   "sect8_hh_w5.csv",
        "shocks":    "sect9_hh_w5.csv"},
}

# ── Survey weight columns ─────────────────────────────────────────────────────
WEIGHT_COL = {1:"pw", 2:"pw", 3:"pw_w3", 4:"pw_w4", 5:"pw_w5"}

# ── Region mapping ────────────────────────────────────────────────────────────
# W1/W2/W3: saq01 is int | W4/W5: saq01 is string already
REGION_MAP = {
    1:"TIGRAY", 2:"AFAR", 3:"AMHARA", 4:"OROMIA", 5:"SOMALI",
    6:"BENISHANGUL GUMUZ", 7:"SNNP", 12:"GAMBELA",
    13:"HARAR", 14:"ADDIS ABABA", 15:"DIRE DAWA",
}
SETTLEMENT_LABELS = {0:"Urban", 1:"Rural", 2:"Small town", 3:"Large town"}

# ── Roster (sect1) column mappings ───────────────────────────────────────────
# Relationship==1 → household head in all waves
# W1/W3: relationship=hh_s1q02(int), sex=hh_s1q03(1=M,2=F), age=hh_s1q04_a
#        edu_level=hh_s1q07 (1=none,2=read/write only,3=primary,4=secondary+)
# W4/W5: relationship=s1q01(str "1. Head"), sex=s1q02(str), age=s1q03a(int)
#        s1q07 = can read/write (1. YES / 2. NO) — literacy proxy only
ROSTER_COLS = {
    1: {"rel":"hh_s1q02", "sex":"hh_s1q03", "age":"hh_s1q04_a", "edu":"hh_s1q07"},
    2: {"rel":"hh_s1q02", "sex":"hh_s1q03", "age":"hh_s1q04_a", "edu":"hh_s1q07"},
    3: {"rel":"hh_s1q02", "sex":"hh_s1q03", "age":"hh_s1q04a",  "edu":"hh_s1q07"},
    4: {"rel":"s1q01",    "sex":"s1q02",    "age":"s1q03a",     "edu":"s1q07"},
    5: {"rel":"s1q01",    "sex":"s1q02",    "age":"s1q03a",     "edu":"s1q07"},
}

# ── Labour (sect3) column mappings ───────────────────────────────────────────
# worked_for_pay: W1/W3 → int (1=yes,2=no) | W4/W5 → str ("1. YES","2. NO")
# weeks: W1/W3 → hh_s3q21_a (float, weeks worked)
#        W4/W5 → s3q18 contains EARNINGS (ETB), not weeks — used as wage proxy
LABOUR_COLS = {
    1: {"worked":"hh_s3q18", "weeks":"hh_s3q21_a", "is_wage_col":True},
    2: {"worked":"hh_s3q18", "weeks":"hh_s3q21_a", "is_wage_col":True},
    3: {"worked":"hh_s3q18", "weeks":"hh_s3q21_a", "is_wage_col":True},
    4: {"worked":"s3q14",    "weeks":"s3q18",       "is_wage_col":False}, # s3q18=earnings ETB
    5: {"worked":"s3q14",    "weeks":"s3q18",       "is_wage_col":False},
}

# ── Enterprise (sect7) column mappings — W1/W2/W3 only ──────────────────────
# hh_s7q01: 1=yes owns enterprise, 2=no | hh_s7q02_a-h: asset quantities
ENTERPRISE_COLS = {
    1: {"has_ent":"hh_s7q01",
        "assets":["hh_s7q02_a","hh_s7q02_b","hh_s7q02_c","hh_s7q02_d",
                  "hh_s7q02_e","hh_s7q02_f","hh_s7q02_g","hh_s7q02_h"]},
    2: {"has_ent":"hh_s7q01",
        "assets":["hh_s7q02_a","hh_s7q02_b","hh_s7q02_c","hh_s7q02_d",
                  "hh_s7q02_e","hh_s7q02_f","hh_s7q02_g","hh_s7q02_h"]},
    3: {"has_ent":"hh_s7q01",
        "assets":["hh_s7q02_a","hh_s7q02_b","hh_s7q02_c","hh_s7q02_d",
                  "hh_s7q02_e","hh_s7q02_f","hh_s7q02_g","hh_s7q02_h"]},
}

# ── Housing column mappings ───────────────────────────────────────────────────
# W1/W3: from sect9 | W4/W5: from sect8
# owns_phone/tv/fridge: 1=yes, 2=no in W1/W3 | not in W4/W5 sect8
HOUSING_COLS = {
    1: {"rooms":"hh_s9q02_a", "roof":"hh_s9q04",   "wall":"hh_s9q05",
        "floor":"hh_s9q06",   "water":"hh_s9q07",  "toilet":"hh_s9q08",
        "fuel":"hh_s9q10",    "lighting":"hh_s9q11","owns_phone":"hh_s9q15",
        "owns_tv":"hh_s9q16", "owns_fridge":"hh_s9q17"},
    2: {"rooms":"hh_s9q02_a", "roof":"hh_s9q04",   "wall":"hh_s9q05",
        "floor":"hh_s9q06",   "water":"hh_s9q07",  "toilet":"hh_s9q08",
        "fuel":"hh_s9q10",    "lighting":"hh_s9q11","owns_phone":"hh_s9q15",
        "owns_tv":"hh_s9q16", "owns_fridge":"hh_s9q17"},
    # W3: column shifts confirmed from inspection
    3: {"rooms":"hh_s9q02_a", "roof":"hh_s9q04",   "wall":"hh_s9q05",
        "floor":"hh_s9q06",   "water":"hh_s9q07",  "toilet":"hh_s9q10",
        "fuel":"hh_s9q13",    "lighting":None,      "owns_phone":None,
        "owns_tv":"hh_s9q16_a","owns_fridge":"hh_s9q17"},
    # W4/W5: sect8 has floor type & toilet presence but NOT roof/wall/water
    4: {"rooms":"s8q02a",    "roof":None,           "wall":None,
        "floor":"s8q03a",   "water":None,           "toilet":"s8q04",
        "fuel":None,         "lighting":None,        "owns_phone":None,
        "owns_tv":None,     "owns_fridge":None,     "electricity":"s8q06"},
    5: {"rooms":"s8q02a",    "roof":None,           "wall":None,
        "floor":"s8q03a",   "water":None,           "toilet":"s8q04",
        "fuel":None,         "lighting":None,        "owns_phone":None,
        "owns_tv":None,     "owns_fridge":None,     "electricity":"s8q06"},
}
# In W1/W3: lighting==1 → household has electricity (electric grid = code 1)
ELEC_FROM_LIGHTING_WAVES = {1, 2, 3}

# ── Shock column mappings ─────────────────────────────────────────────────────
# W1/W3: shock_code=int(101-118), affected=int(1=yes,2=no)
# W4/W5: shock_code=str("1. DEATH..."), affected=str("1. YES"/"2. NO")
SHOCK_COLS = {
    1: {"code":"hh_s8q00", "affected":"hh_s8q01", "fmt":"int"},
    2: {"code":"hh_s8q00", "affected":"hh_s8q01", "fmt":"int"},
    3: {"code":"hh_s8q00", "affected":"hh_s8q01", "fmt":"int"},
    4: {"code":"shock_type","affected":"s9q01",    "fmt":"str"},
    5: {"code":"shock_type","affected":"s9q01",    "fmt":"str"},
}
# Shock code patterns for drought, illness, death, crop loss
SHOCK_PATTERNS = {
    "drought":    ["106", "DROUGHT"],
    "illness":    ["104", "ILLNESS"],
    "death":      ["101", "DEATH OF HH MEMBER (MAIN)"],
    "crop_loss":  ["108", "109", "CROP", "HARVEST", "LIVESTOCK DISEASE"],
}

# ── Target & feature groups ───────────────────────────────────────────────────
TARGET      = "cons_quint"   # 1=poorest → 5=wealthiest (classification)

GEO_COLS     = ["region", "settlement", "wave", "zone_id"]
DEMO_COLS    = ["hh_size", "adulteq"]
HEAD_COLS    = ["head_sex", "head_age", "head_age_sq",
                "head_edu_level", "is_female_headed", "head_literate"]
LABOUR_FEAT  = ["hh_any_wage_earner", "hh_n_workers", "hh_avg_weeks_worked"]
ENTERPRISE_F = ["has_nonfarm_enterprise", "enterprise_asset_count"]
HOUSING_FEAT = ["rooms", "roof", "wall", "floor", "water", "toilet",
                "fuel", "has_electricity", "housing_score"]
ASSET_FEAT   = ["owns_phone", "owns_tv", "owns_fridge", "asset_count"]
SHOCK_FEAT   = ["n_shocks","experienced_drought","experienced_illness",
                "experienced_death","experienced_crop_loss"]
DERIVED_FEAT = ["log_hh_size","dependency_ratio","assets_per_member",
                "post_covid","is_tigray_conflict"]

ALL_FEATURES = (GEO_COLS + DEMO_COLS + HEAD_COLS + LABOUR_FEAT
                + ENTERPRISE_F + HOUSING_FEAT + ASSET_FEAT
                + SHOCK_FEAT + DERIVED_FEAT)

# Model settings
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
RANDOM_STATE = 42
CV_FOLDS     = 5
MIN_REGION_N = 50