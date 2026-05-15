"""
app.py
═══════════════════════════════════════════════════════════════════════════════
Ethiopian Household Wealth Prediction — Streamlit Web Application

Run:
    cd dsa_project/
    streamlit run app/app.py

Pages:
    Home           — project overview, dataset statistics, CRISP-DM pipeline
    Data Explorer  — per-wave and cross-wave interactive exploration
    EDA            — visualisations: distributions, correlations, trends
    Preprocessing  — missing value analysis, cleaning log, feature groups
    Modelling      — train models, compare results, per-region predictions
    Regional Map   — wealth quintile ranking and regional comparison
    Predict        — single-household wealth quintile prediction form
    About          — methods, data sources, ethical considerations
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
import os
import warnings
import time
from html import escape
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).resolve().parent
ROOT     = APP_DIR.parent
SRC_DIR  = ROOT / "src"
DATA_DIR = ROOT / "data"
MODEL_DIR= ROOT / "models"

# Ensure repository `src/` is importable on Streamlit Cloud and other runtimes
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT))

try:
    from src.config import CLEANED_CSV
except Exception:
    from config import CLEANED_CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import streamlit as st
import joblib

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethiopian Wealth Predictor",
    page_icon="ETH",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/mesfinmaru/Ethiopian-Household-Wealth-Prediction.git",
        "Report a bug":"mailto:mesfinmaru121@gmail.com",
        "About":       "Ethiopian Household Wealth Prediction",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME & CSS
# ══════════════════════════════════════════════════════════════════════════════

THEME_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Root variables ── */
:root {
    --green-dark:   #1B4332;
    --green-mid:    #2D6A4F;
    --green-light:  #52B788;
    --gold:         #D4A017;
    --gold-light:   #F6C94E;
    --red-eth:      #C8102E;
    --bg-dark:      #0D1117;
    --bg-card:      #161B22;
    --bg-card2:     #1C2230;
    --text-primary: #E6EDF3;
    --text-muted:   #8B949E;
    --border:       #30363D;
    --q1: #C8102E;  --q2: #F4830A;  --q3: #F6C94E;
    --q4: #52B788;  --q5: #1565C0;
    --sidebar-width: 21rem;
}

/* ── Base ── */
.stApp { background: var(--bg-dark); font-family: 'Sora', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 6rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--green-dark) 0%, #0D1F17 100%);
    border-right: 1px solid var(--green-mid);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    overflow: hidden;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.25rem;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    padding: 0.14rem 0;
    line-height: 1.15;
}
[data-testid="stSidebar"] hr {
    border-color: var(--green-mid);
    opacity: 0.4;
    margin: 0.45rem 0;
}

/* ── Sidebar team card ── */
.team-card {
    background: rgba(82,183,136,0.08);
    border: 1px solid #2D6A4F;
    border-radius: 8px;
    padding: 0.46rem 0.62rem;
    font-size: 0.69rem;
    color: #8B949E;
    text-align: left;
}
.team-title {
    color: #52B788;
    font-weight: 700;
    margin-bottom: 0.35rem;
    text-align: center;
}
.team-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.14rem;
}
.team-row {
    display: grid;
    grid-template-columns: max-content 1fr;
    column-gap: 0.4rem;
    align-items: start;
    line-height: 1.12;
}
.team-id {
    min-width: 4.65rem;
    color: #F6C94E;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.65rem;
}
.team-name {
    color: #E6EDF3;
    text-align: left;
    font-size: 0.67rem;
    white-space: normal;
    overflow-wrap: anywhere;
}
.team-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    margin-top: 0.35rem;
    font-size: 0.72rem;
}
.team-table th,
.team-table td {
    padding: 0.35rem 0.25rem;
    text-align: left;
    vertical-align: top;
    word-break: break-word;
}
.team-table th {
    color: #52B788;
    font-weight: 700;
    border-bottom: 1px solid rgba(45,106,79,0.65);
}
.team-table td {
    color: #E6EDF3;
    border-bottom: 1px solid rgba(45,106,79,0.25);
}
.team-table tr:last-child td {
    border-bottom: 0;
}
.team-no {
    width: 2.1rem;
    color: #8B949E;
}
.team-id-cell {
    width: 6.8rem;
    color: #F6C94E;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, var(--green-dark) 0%, #163D26 40%, #1B3A5C 100%);
    border: 1px solid var(--green-mid);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(82,183,136,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.15;
    background: linear-gradient(135deg, #F6C94E 0%, #52B788 60%, #74C6E8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 1.05rem; color: rgba(230,237,243,0.75); margin: 0;
    font-family: 'Lora', serif; font-style: italic;
}
.hero-badge {
    display: inline-block;
    background: rgba(82,183,136,0.2); border: 1px solid var(--green-light);
    color: var(--green-light); border-radius: 20px;
    padding: 0.2rem 0.9rem; font-size: 0.78rem; font-weight: 600;
    margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase;
}
.eth-flag {
    font-size: 2.8rem; position: absolute; top: 1.5rem; right: 2rem; opacity: 0.85;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 1.4rem;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: var(--green-light); transform: translateY(-2px); }
.metric-card .mc-val {
    font-size: 2rem; font-weight: 800; line-height: 1;
    color: var(--gold-light); display: block; margin-bottom: 0.25rem;
}
.metric-card .mc-lbl {
    font-size: 0.75rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.06em;
}

/* ── Section header ── */
.sec-header {
    font-size: 1.35rem; font-weight: 700; color: var(--text-primary);
    border-left: 4px solid var(--gold); padding-left: 0.8rem;
    margin: 0.8rem 0 1rem;
}
.sec-header .sec-icon { margin-right: 0.4rem; }

/* ── Info / warning / success boxes ── */
.info-box {
    background: rgba(82,183,136,0.08); border: 1px solid rgba(82,183,136,0.35);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0;
    color: #A0D9B4; font-size: 0.9rem; line-height: 1.6;
}
.warn-box {
    background: rgba(244,131,10,0.08); border: 1px solid rgba(244,131,10,0.4);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0;
    color: #FACC82; font-size: 0.9rem;
}

/* ── Quintile pills ── */
.q-pill {
    display: inline-block; border-radius: 20px;
    padding: 0.2rem 0.75rem; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.04em;
}
.q1 { background: rgba(200,16,46,0.2);  color: #FF6B7A; }
.q2 { background: rgba(244,131,10,0.2); color: #FACC82; }
.q3 { background: rgba(246,201,78,0.2); color: #F6C94E; }
.q4 { background: rgba(82,183,136,0.2); color: #52B788; }
.q5 { background: rgba(21,101,192,0.2); color: #74C6E8; }

/* ── Table styling ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame thead th {
    background: var(--bg-card2) !important;
    color: var(--gold-light) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark));
    color: white; border: 1px solid var(--green-light);
    border-radius: 8px; font-weight: 600; font-family: 'Sora', sans-serif;
    padding: 0.5rem 1.4rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--green-light), var(--green-mid));
    border-color: var(--gold); transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(82,183,136,0.3);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card); border-radius: 10px;
    gap: 0.3rem; padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; color: var(--text-muted);
    font-weight: 600; font-size: 0.88rem; padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: var(--green-mid) !important;
    color: white !important;
}

/* ── Selectbox / slider ── */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stRadio label, .stCheckbox label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
}

/* ── Progress / spinner ── */
.stProgress > div > div { background: var(--green-light) !important; }

/* ── Bottom status bar ── */
.status-bar {
    position: fixed;
    left: calc(var(--sidebar-width) + 1rem);
    right: 1rem;
    bottom: 0.8rem;
    z-index: 9999;
    pointer-events: none;
}
.status-bar__panel {
    pointer-events: auto;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 1rem;
    background: rgba(22, 27, 34, 0.92);
    border: 1px solid rgba(48, 54, 61, 0.95);
    border-radius: 999px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
}
.status-bar__dot {
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 50%;
    background: var(--green-light);
    box-shadow: 0 0 0 0 rgba(82, 183, 136, 0.45);
    animation: statusPulse 1.5s infinite;
    flex-shrink: 0;
}
.status-bar__label {
    color: var(--text-primary);
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
}
.status-bar__track {
    flex: 1;
    min-width: 6rem;
    height: 0.42rem;
    background: rgba(48, 54, 61, 0.9);
    border-radius: 999px;
    overflow: hidden;
}
.status-bar__fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--green-light), var(--gold-light));
    transition: width 0.25s ease;
}
.status-bar__fill--active {
    background-size: 220% 100%;
    animation: statusSweep 1.8s linear infinite;
}
.status-bar__meta {
    color: var(--text-muted);
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}
@keyframes statusPulse {
    0% { box-shadow: 0 0 0 0 rgba(82, 183, 136, 0.35); }
    70% { box-shadow: 0 0 0 10px rgba(82, 183, 136, 0); }
    100% { box-shadow: 0 0 0 0 rgba(82, 183, 136, 0); }
}
@keyframes statusSweep {
    0% { background-position: 0% 50%; }
    100% { background-position: 220% 50%; }
}

@media (max-width: 1200px) {
    .status-bar {
        left: 1rem;
    }
}

/* ── Prediction result card ── */
.pred-result {
    background: linear-gradient(135deg, var(--bg-card2), var(--bg-card));
    border: 2px solid var(--gold);
    border-radius: 16px; padding: 2rem 2.5rem; text-align: center;
    margin: 1.5rem 0;
}
.pred-quintile {
    font-size: 5rem; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, var(--gold), var(--gold-light));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.pred-label {
    font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;
    color: var(--text-muted);
}

/* ── CRISP-DM flow ── */
.crisp-step {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.85rem 1rem; margin-bottom: 0.65rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; border-left: 4px solid var(--green-light);
    transition: border-color 0.2s;
}
.crisp-step:hover { border-left-color: var(--gold); }
.crisp-num {
    background: var(--green-mid); color: white;
    border-radius: 50%; width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.9rem; flex-shrink: 0;
}
.crisp-content h4 { margin: 0 0 0.2rem; color: var(--text-primary); font-size: 0.95rem; }
.crisp-content p  { margin: 0; color: var(--text-muted); font-size: 0.82rem; line-height: 1.5; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 2rem 1rem; margin-top: 3rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted); font-size: 0.8rem;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

QUINTILE_LABELS = {
    1: ("Q1 — Poorest 20%",   "q1", "●"),
    2: ("Q2 — Lower Middle",  "q2", "●"),
    3: ("Q3 — Middle",        "q3", "●"),
    4: ("Q4 — Upper Middle",  "q4", "●"),
    5: ("Q5 — Wealthiest 20%","q5", "●"),
}

REGION_LIST = [
    "TIGRAY","AFAR","AMHARA","OROMIA","SOMALI",
    "BENISHANGUL GUMUZ","SNNP","GAMBELA",
    "HARAR","ADDIS ABABA","DIRE DAWA",
]

WAVE_META = {
    1: {"year":"2011–12","label":"W1","context":"Pre-MDG baseline"},
    2: {"year":"2013–14","label":"W2","context":"High growth period (~10% GDP/yr)"},
    3: {"year":"2015–16","label":"W3","context":"El Niño drought year"},
    4: {"year":"2018–19","label":"W4","context":"Pre-COVID baseline"},
    5: {"year":"2021–22","label":"W5","context":"Post-COVID + Tigray conflict"},
}

COLORS5  = ["#C8102E","#F4830A","#F6C94E","#52B788","#1565C0"]
PLT_STYLE = {
    "figure.facecolor":  "#161B22",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#E6EDF3",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#E6EDF3",
    "grid.color":        "#21262D",
    "grid.alpha":        0.6,
}

def apply_plt_style():
    plt.rcParams.update(PLT_STYLE)

def q_pill(q: int) -> str:
    lbl, cls, _ = QUINTILE_LABELS.get(q, ("Unknown","q3",""))
    return f'<span class="q-pill {cls}">{lbl}</span>'

def metric_card(value, label, delta=None):
    delta_html = f'<span style="font-size:.7rem;color:#52B788;">▲ {delta}</span>' if delta else ""
    return (
        f'<div class="metric-card">'
        f'<span class="mc-val">{value}</span>'
        f'<span class="mc-lbl">{label}</span>'
        f'{delta_html}'
        f'</div>'
    )

def sec_header(icon, title):
    st.markdown(
        f'<div class="sec-header"><span class="sec-icon">{icon}</span>{title}</div>',
        unsafe_allow_html=True,
    )

def info_box(text):
    st.markdown(f'<div class="info-box">ℹ️ &nbsp;{text}</div>', unsafe_allow_html=True)

def warn_box(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def set_status(message: str, progress: int = 0, active: bool = True):
    st.session_state["ui_status_message"] = message
    st.session_state["ui_status_progress"] = int(max(0, min(100, progress)))
    st.session_state["ui_status_active"] = bool(active)


def render_status_bar(message: str | None = None,
                      progress: int | None = None,
                      active: bool | None = None):
    if message is not None or progress is not None or active is not None:
        set_status(
            message if message is not None else st.session_state.get("ui_status_message", "Ready"),
            progress if progress is not None else st.session_state.get("ui_status_progress", 100),
            active if active is not None else st.session_state.get("ui_status_active", False),
        )

    msg = escape(str(st.session_state.get("ui_status_message", "Ready")))
    pct = int(max(0, min(100, st.session_state.get("ui_status_progress", 100))))
    is_active = bool(st.session_state.get("ui_status_active", False))
    if not is_active:
        return
    fill_class = "status-bar__fill status-bar__fill--active" if is_active else "status-bar__fill"
    st.markdown(f"""
    <div class="status-bar" role="status" aria-live="polite">
        <div class="status-bar__panel">
            <span class="status-bar__dot"></span>
            <span class="status-bar__label">{msg}</span>
            <div class="status-bar__track">
                <div class="{fill_class}" style="width: {pct}%;"></div>
            </div>
            <span class="status-bar__meta">{pct}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING 
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_cleaned_data() -> pd.DataFrame | None:
    """Load the pre-built cleaned CSV. Returns None if not built yet."""
    candidates = [
        CLEANED_CSV,
        DATA_DIR / "processed" / CLEANED_CSV.name,
        ROOT / "data" / "processed" / CLEANED_CSV.name,
        Path(f"data/processed/{CLEANED_CSV.name}"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            if "region" in df.columns:
                df["region"] = df["region"].astype("category")
            if df.isna().sum().sum() > 0:
                try:
                    from data_cleaner import DataCleaner
                    cleaner = DataCleaner()
                    df = cleaner.fit_transform(df)
                    df.to_csv(p, index=False)
                except Exception:
                    pass
            return df
    return None

@st.cache_data(show_spinner=False)
def try_build_data() -> pd.DataFrame | None:
    """Try to build data via data_loader if CSVs not found."""
    try:
        from data_loader import build_all_waves
        from data_cleaner import DataCleaner

        raw = build_all_waves(save=False, verbose=False)
        cleaned = DataCleaner().fit_transform(raw)
        (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(DATA_DIR / "processed" / CLEANED_CSV.name, index=False)
        return cleaned
    except Exception:
        return None

def load_models() -> dict:
    """Load saved model files from models/ directory.
    Note: Caching removed to allow hot-reload of updated preprocessor/model pickles.
    """
    found = {}
    dirs = [MODEL_DIR, ROOT / "models", Path("models")]
    for d in dirs:
        if d.exists():
            for pkl in sorted(d.glob("*.pkl")):
                try:
                    found[pkl.stem] = joblib.load(pkl)
                except Exception:
                    pass
    return found

def run_pipeline_cached():
    """Run the full 5-wave data pipeline with a progress indicator."""
    progress = st.progress(0, text="Loading Wave 1 (2011-12)…")
    try:
        set_status("Loading Wave 1 (2011-12)…", 5, True)
        render_status_bar()
        from data_loader import build_wave
        from data_cleaner import DataCleaner
        frames = []
        for i, w in enumerate([1, 2, 3, 4, 5], 1):
            progress.progress(i * 18, text=f"Loading Wave {w} ({WAVE_META[w]['year']})…")
            set_status(f"Loading Wave {w} ({WAVE_META[w]['year']})…", i * 18, True)
            render_status_bar()
            f = build_wave(w, verbose=False)
            if not f.empty:
                frames.append(f)
            time.sleep(0.1)
        import pandas as pd_inner
        df = pd_inner.concat(frames, ignore_index=True, sort=False)
        df = DataCleaner().fit_transform(df)
        progress.progress(95, text="Saving processed data…")
        set_status("Saving processed data…", 95, True)
        render_status_bar()
        (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "processed" / CLEANED_CSV.name, index=False)
        progress.progress(100, text="Done!")
        set_status("Dataset build complete", 100, False)
        render_status_bar()
        time.sleep(0.4)
        progress.empty()
        return df
    except Exception as e:
        progress.empty()
        set_status("Dataset build failed", 0, False)
        render_status_bar()
        st.error(f"Pipeline error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 0.45rem 0 0.2rem;'>
            <div style='font-size:2rem;'>ET</div>
            <div style='font-weight:800; font-size:1.05rem;'
                        color:#F6C94E; line-height:1.2; margin-top:0.2rem;'>
                Ethiopian Wealth<br>Predictor
            </div>
        </div>
        <hr/>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=[
                "Home",
                "Data Explorer",
                "EDA",
                "Preprocessing",
                "Modelling",
                "Regional Wealth Map",
                "Predict Household",
                "About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Dataset status
        df = load_cleaned_data()
        if df is not None:
            st.markdown(f"""
            <div style='background:rgba(82,183,136,0.1);border:1px solid #2D6A4F;
                        border-radius:8px;padding:0.55rem 0.7rem;font-size:0.72rem;'>
                <div style='color:#52B788;font-weight:700;margin-bottom:0.18rem;'>
                    Dataset Ready
                </div>
                <div style='color:#8B949E;'>
                    {df.shape[0]:,} households<br>
                    {df['wave'].nunique()} waves · {len(df.columns)} features
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(200,16,46,0.1);border:1px solid #C8102E;
                        border-radius:8px;padding:0.55rem 0.7rem;font-size:0.72rem;'>
                <div style='color:#FF6B7A;font-weight:700;'>Data Not Built</div>
                <div style='color:#8B949E;'>Go to Home → Build Dataset</div>
            </div>""", unsafe_allow_html=True)

        # team card moved to About page to keep sidebar compact

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="eth-flag">ET</div>
        <div class="hero-badge">InSy3056 · Data Science Application</div>
        <div class="hero-title">Ethiopian Household<br>Wealth Prediction</div>
        <div class="hero-sub">
            Predicting welfare quintiles from 5 waves of World Bank Living Standards
            Measurement Study surveys (2011–2022)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Overview metrics
    df = load_cleaned_data()
    if df is not None:
        n_hh   = df.shape[0]
        n_wave = df["wave"].nunique()
        n_reg  = df["region"].nunique()
        n_feat = df.shape[1] - 2
        st.markdown(
            '<div class="metric-row">'
            + metric_card(f"{n_hh:,}", "Total Households")
            + metric_card(str(n_wave), "Survey Waves" "<br>2011-2022")
            + metric_card(str(n_reg), "Ethiopian Regions")
            + metric_card(str(n_feat), "Proxy Features")
            + metric_card("5-class", "Classification Task")
            + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="warn-box">
        The processed dataset has not been built yet. Click <strong>Build Dataset</strong>
        below to load all 5 survey waves from the raw data files.
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.95], gap="small")

    with col_left:
        # Project description
        sec_header("", "Project Overview")
        st.markdown("""
        <div style="color:#8B949E; font-size:0.93rem; line-height:1.8;">
        Ethiopia's Living Standards Measurement Study (<strong style="color:#E6EDF3;">LSMS/ESS</strong>)
        is a longitudinal household panel survey conducted by the World Bank and
        the Ethiopian Central Statistical Agency. It covers five rounds:

        <br><br>
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <tr style="border-bottom:1px solid #30363D;">
            <th style="color:#52B788;padding:0.5rem;text-align:left;">Wave</th>
            <th style="color:#52B788;padding:0.5rem;text-align:left;">Period</th>
            <th style="color:#52B788;padding:0.5rem;text-align:left;">Context</th>
        </tr>
        <tr><td style="padding:0.45rem;">W1</td><td>2011–12</td><td>Pre-MDG baseline</td></tr>
        <tr><td style="padding:0.45rem;">W2</td><td>2013–14</td><td>High economic growth</td></tr>
        <tr><td style="padding:0.45rem;">W3</td><td>2015–16</td><td>El Niño drought</td></tr>
        <tr><td style="padding:0.45rem;">W4</td><td>2018–19</td><td>Pre-COVID baseline</td></tr>
        <tr><td style="padding:0.45rem;">W5</td><td>2021–22</td><td>Post-COVID + Tigray conflict</td></tr>
        </table>

        <br>
        The <strong style="color:#F6C94E;">target variable</strong> is
        <code style="color:#52B788;">cons_quint</code> (consumption-based wealth quintile,
        1 = poorest 20% → 5 = wealthiest 20%). Only <em>proxy</em> features are used —
        no consumption aggregates are ever included as inputs (strict leakage prevention).
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # CRISP-DM pipeline
        sec_header("", "CRISP-DM Pipeline")
        steps = [
            ("1", "Business Understanding",
             "Define poverty prediction as 5-class classification. Identify leakage-free "
             "proxy features (housing, assets, demographics, shocks)."),
            ("2", "Data Understanding",
             "Load 5 LSMS waves. W2 in SPSS .sav format decoded by embedded SAV reader. "
             "Explore coverage gaps, wave design differences, W2 column truncation."),
            ("3", "Data Preparation",
             "7-step missing-value pipeline (MNAR flags, W2 cross-wave donor fill, "
             "group medians, KNN). IQR outlier capping. Feature engineering (7 groups)."),
            ("4", "Modelling",
             "10 classifiers (LR, DT, RF, KNN, NB, SVM, AdaBoost, GBT, XGBoost, LightGBM). "
             "Per-region models + GridSearchCV tuning. Unsupervised: K-Means, PCA, t-SNE."),
            ("5", "Evaluation",
             "Stratified 5-fold CV. Accuracy, weighted F1, macro F1, ROC-AUC. "
             "Learning curves, validation curves. Paired t-test for significance."),
            ("6", "Deployment",
             "This Streamlit application: interactive EDA, preprocessing audit, "
             "model comparison, regional wealth map, and single-HH prediction."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="crisp-step">
                <div class="crisp-num">{num}</div>
                <div class="crisp-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>""", unsafe_allow_html=True)

    # Build dataset button
    sec_header("", "Dataset Build")
    st.markdown("<div style='display:flex;gap:0.8rem;justify-content:center;align-items:center;flex-wrap:wrap;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        if st.button("Build Dataset (All 5 Waves)", use_container_width=True):
            with st.spinner("Building multi-wave dataset…"):
                df = run_pipeline_cached()
            if df is not None:
                st.success(f"Dataset built: {df.shape[0]:,} households × {df.shape[1]} columns")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Build failed — check that raw data files are in data/raw/")
    with col2:
        cleaned_path = DATA_DIR / "processed" / "all_waves_clean.csv"
        if cleaned_path.exists():
            with open(cleaned_path, "rb") as f:
                st.download_button(
                    "Download Cleaned CSV",
                    f, "all_waves_clean.csv", "text/csv",
                    use_container_width=True,
                )
    
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def page_data_explorer():
    sec_header("", "Data Explorer")

    df = load_cleaned_data()
    if df is None:
        warn_box("Dataset not built. Go to Home → Build Dataset first.")
        return

    # Attempt to restore previously saved preprocessor and models into session
    try:
        from src.data_preprocesor import DataPreprocessor
    except Exception:
        from data_preprocesor import DataPreprocessor
    try:
        from src.modeling import WealthPredictor
    except Exception:
        from modeling import WealthPredictor

    # If saved artifacts exist on disk, load them into session_state so refreshes keep trained models
    try:
        if "wp" not in st.session_state:
            dp = DataPreprocessor()
            try:
                dp.load(str(MODEL_DIR / "preprocessor.pkl"))
                st.session_state["preprocessor"] = (dp.pipeline_, dp.label_encoder_, dp.feature_names_)
            except Exception:
                # preprocessor not present or failed to load
                pass

            wp = WealthPredictor()
            try:
                wp.load(str(MODEL_DIR))
                # If a best model was loaded, populate minimal session state entries
                if getattr(wp, "best_model_", None) is not None:
                    st.session_state["wp"] = wp
                    st.session_state["best_name"] = getattr(wp, "best_name_", "saved_model") or "saved_model"
                    # minimal placeholder for model_results so Results tab is accessible
                    import pandas as _pd
                    if "model_results" not in st.session_state:
                        st.session_state["model_results"] = _pd.DataFrame([
                            {"model": st.session_state["best_name"], "accuracy": 0.0}
                        ])
            except Exception:
                pass
    except Exception:
        pass

    # Wave selector
    wave_options = {f"W{w} · {WAVE_META[w]['year']} ({WAVE_META[w]['context']})": w
                    for w in sorted(df["wave"].unique())}
    mode = st.radio("View mode", ["All Waves Combined", "Single Wave"],
                    horizontal=True)

    if mode == "Single Wave":
        chosen = st.selectbox("Select Wave", list(wave_options.keys()))
        view = df[df["wave"] == wave_options[chosen]].copy()
    else:
        view = df.copy()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Households", f"{len(view):,}")
    col_b.metric("Features",   str(view.shape[1]))
    col_c.metric("Missing cells",
                 f"{view.isnull().sum().sum():,}",
                 delta=f"{view.isnull().mean().mean()*100:.1f}% avg")
    col_d.metric("Mean Quintile", f"{view['cons_quint'].mean():.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Raw Preview", "Summary Stats", "Missing Values", "Geographic"]
    )

    with tab1:
        cols = st.multiselect(
            "Columns to display",
            options=view.columns.tolist(),
            default=["household_id","wave","region","settlement",
                     "hh_size","head_age","housing_score","cons_quint"],
        )
        n_rows = st.slider("Rows to show", 10, 200, 50)
        st.dataframe(view[cols].head(n_rows), use_container_width=True, height=420)

    with tab2:
        st.dataframe(view.describe().round(3), use_container_width=True)

    with tab3:
        miss = view.isnull().sum()
        pct  = (miss / len(view) * 100).round(2)
        miss_df = pd.DataFrame({"n_missing": miss, "pct_missing": pct})
        miss_df = miss_df[miss_df["n_missing"] > 0].sort_values("pct_missing", ascending=False)

        if miss_df.empty:
            st.success("No missing values!")
        else:
            apply_plt_style()
            fig, ax = plt.subplots(figsize=(10, max(4, len(miss_df)*0.28)))
            colors = ["#C8102E" if v>50 else "#F4830A" if v>20 else "#F6C94E"
                      for v in miss_df["pct_missing"]]
            ax.barh(miss_df.index, miss_df["pct_missing"], color=colors, edgecolor="none")
            ax.axvline(50, color="#C8102E", ls="--", lw=1.2, label=">50%")
            ax.axvline(20, color="#F4830A", ls="--", lw=1.2, label=">20%")
            ax.set_xlabel("% Missing"); ax.legend(fontsize=8)
            ax.set_title("Missing Value Rate per Feature", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.dataframe(miss_df, use_container_width=True)

    with tab4:
        region_q = (view.groupby("region", observed=True)["cons_quint"]
                       .agg(n="count", mean_q="mean").round(3)
                       .sort_values("mean_q", ascending=False).reset_index())
        apply_plt_style()
        fig, ax = plt.subplots(figsize=(9, 6))
        norm = (region_q["mean_q"] - 1) / 4
        clrs = plt.cm.RdYlGn(norm.values)
        ax.barh(region_q["region"], region_q["mean_q"],
                color=clrs, edgecolor="none")
        ax.axvline(3, color="#8B949E", ls="--", lw=1.2, label="Q3 median")
        ax.set_xlabel("Mean Wealth Quintile (1=poorest, 5=richest)")
        ax.set_title("Mean Wealth Quintile by Region", fontweight="bold")
        ax.legend()
        for i, row in region_q.iterrows():
            ax.text(row["mean_q"] + 0.02, i, f"{row['mean_q']:.2f}",
                    va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EDA
# ══════════════════════════════════════════════════════════════════════════════

def page_eda():
    sec_header("", "Exploratory Data Analysis")

    df = load_cleaned_data()
    if df is None:
        warn_box("Dataset not built. Go to Home → Build Dataset first.")
        return

    # Apply feature engineering if available
    try:
        from feature_enginner import FeatureEngineer
        from data_cleaner import DataCleaner
        df_clean = DataCleaner().fit_transform(df)
        df_eng   = FeatureEngineer().engineer_all(df_clean)
    except Exception:
        df_eng = df.copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Univariate", "Bivariate", "Multivariate",
        "Temporal Trends", "Shocks"
    ])

    with tab1:
        st.markdown("**Univariate: continuous feature distributions by wealth quintile**")
        cont_opts = [c for c in ["hh_size","head_age","rooms","housing_score",
                                  "housing_quality_idx","modern_asset_score",
                                  "dependency_ratio","adults_ratio"]
                     if c in df_eng.columns]
        feat = st.selectbox("Select feature", cont_opts)

        apply_plt_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram by quintile
        for q, c in enumerate(COLORS5, 1):
            data = df_eng.loc[df_eng["cons_quint"]==q, feat].dropna()
            axes[0].hist(data, bins=25, color=c, alpha=0.55, label=f"Q{q}",
                         density=True, edgecolor="none")
        axes[0].set_title(f"Distribution of {feat} by Quintile", fontweight="bold")
        axes[0].set_xlabel(feat); axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=8)

        # Box plot by quintile
        groups = [df_eng.loc[df_eng["cons_quint"]==q, feat].dropna()
                  for q in range(1, 6)]
        bp = axes[1].boxplot(groups, patch_artist=True, notch=False,
                             medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], COLORS5):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        axes[1].set_xticklabels([f"Q{i}" for i in range(1,6)])
        axes[1].set_title(f"{feat} — Box Plot", fontweight="bold")
        kw_stat, kw_p = stats.kruskal(*groups)
        axes[1].set_xlabel(f"Kruskal-Wallis p={kw_p:.4f}")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown("**Bivariate: asset ownership & utility access rate by wealth quintile**")
        asset_cols = [c for c in ["owns_phone","owns_tv","owns_fridge",
                                   "has_electricity","improved_water",
                                   "improved_sanitation","clean_fuel",
                                   "has_nonfarm_enterprise","hh_any_wage_earner"]
                      if c in df_eng.columns]

        pivot = df_eng.groupby("cons_quint")[asset_cols].mean().multiply(100)
        apply_plt_style()
        fig, ax = plt.subplots(figsize=(11, 4))
        x      = np.arange(len(asset_cols))
        width  = 0.15
        for i, (q, c) in enumerate(zip(range(1,6), COLORS5), 0):
            if q in pivot.index:
                ax.bar(x + i*width, pivot.loc[q, asset_cols],
                       width, color=c, label=f"Q{q}", edgecolor="none", alpha=0.85)
        ax.set_xticks(x + width*2)
        ax.set_xticklabels([c.replace("_"," ") for c in asset_cols],
                            rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("% Households")
        ax.set_title("Asset/Utility Access Rate by Wealth Quintile", fontweight="bold")
        ax.legend(title="Quintile")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        st.markdown("**Multivariate: Pearson correlation with `cons_quint` (target)**")
        num_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in
                    ("household_id","survey_weight","wave")]

        corr = df_eng[num_cols].corr()["cons_quint"].drop("cons_quint")
        corr = corr.sort_values(key=abs, ascending=False).head(25)

        apply_plt_style()
        fig, ax = plt.subplots(figsize=(9, 6))
        clrs = ["#52B788" if v > 0 else "#C8102E" for v in corr.values]
        ax.barh(corr.index, corr.values, color=clrs, edgecolor="none")
        ax.axvline(0, color="#30363D", lw=1)
        ax.set_xlabel("Pearson Correlation with cons_quint")
        ax.set_title("Top Feature Correlations with Wealth Quintile", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if st.checkbox("Show full correlation matrix"):
            apply_plt_style()
            sub = [c for c in num_cols if c in df_eng.columns][:20]
            fig2, ax2 = plt.subplots(figsize=(12, 9))
            sns.heatmap(df_eng[sub].corr(), cmap="RdBu_r", center=0,
                        vmin=-1, vmax=1, annot=True, fmt=".2f",
                        annot_kws={"size":6}, linewidths=0.3, ax=ax2)
            ax2.set_title("Pearson Correlation Matrix", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    with tab4:
        trend_opts = [c for c in ["has_electricity","owns_phone","improved_water",
                                   "head_literate","clean_fuel","is_urban",
                                   "housing_score","housing_quality_idx"]
                      if c in df_eng.columns]
        chosen = st.multiselect("Select indicators", trend_opts,
                                default=trend_opts[:4])
        if chosen:
            trend = df_eng.groupby("wave")[chosen].mean()
            apply_plt_style()
            fig, ax = plt.subplots(figsize=(10, 4))
            for col, c in zip(chosen, plt.cm.Set2(np.linspace(0, 1, len(chosen)))):
                ax.plot(trend.index, trend[col] * 100, "o-", color=c, lw=2.5,
                        ms=7, label=col.replace("_"," "))
                ax.fill_between(trend.index, trend[col]*100, alpha=0.07, color=c)
            ax.set_xticks(trend.index)
            ax.set_xticklabels([f"W{w}\n{WAVE_META[w]['year']}" for w in trend.index])
            ax.set_ylabel("% Households (or normalised score ×100)")
            ax.set_title("Temporal Trends — Selected Indicators", fontweight="bold")
            ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab5:
        detailed_shock_cols = [
            c for c in [
                "experienced_drought",
                "experienced_illness",
                "experienced_death",
                "experienced_crop_loss",
            ]
            if c in df_eng.columns
        ]
        proxy_shock_cols = [c for c in ["is_tigray_conflict", "post_covid"] if c in df_eng.columns]

        # Prefer direct household shock variables; fallback to macro/context shocks.
        if detailed_shock_cols:
            shock_cols = detailed_shock_cols
            tab_title_suffix = "Household Shock Exposure"
        elif proxy_shock_cols:
            shock_cols = proxy_shock_cols
            tab_title_suffix = "Shock Proxies Available in Current Dataset"
            st.info(
                "Detailed household shock variables are not present in this processed dataset. "
                "Showing available shock proxies (conflict and COVID context)."
            )
        else:
            st.info(
                "No shock-related columns are available in the current dataset. "
                "Rebuild the dataset with shock fields from source sections if you want full shock analysis."
            )
            return

        by_q = df_eng.groupby("cons_quint")[shock_cols].mean().multiply(100)
        by_reg = df_eng.groupby("region", observed=True)[shock_cols].mean().multiply(100)

        apply_plt_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # By quintile
        for i, (q, c) in enumerate(zip(range(1,6), COLORS5)):
            if q in by_q.index:
                axes[0].bar(
                    np.arange(len(shock_cols)) + i*0.15,
                    by_q.loc[q, shock_cols], 0.15,
                    color=c, label=f"Q{q}", edgecolor="none",
                )
        axes[0].set_xticks(np.arange(len(shock_cols)) + 0.3)
        axes[0].set_xticklabels([c.replace("experienced_","").replace("_"," ")
                                   for c in shock_cols], rotation=20, ha="right")
        axes[0].set_ylabel("% Households Affected")
        axes[0].set_title(f"{tab_title_suffix} by Quintile", fontweight="bold")
        axes[0].legend(fontsize=7)

        # By region for first available shock/proxy column.
        regional_col = None
        for candidate in ["experienced_drought", "is_tigray_conflict", "post_covid"]:
            if candidate in by_reg.columns:
                regional_col = candidate
                break
        if regional_col is not None:
            dr = by_reg[regional_col].sort_values(ascending=False)
            axes[1].barh(
                dr.index,
                dr.values,
                color=plt.cm.OrRd(dr.values / (dr.max() or 1)),
                edgecolor="none",
            )
            axes[1].set_xlabel(f"% HH: {regional_col.replace('_', ' ').title()}")
            axes[1].set_title(
                f"{regional_col.replace('_', ' ').title()} by Region",
                fontweight="bold",
            )
        else:
            axes[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def page_preprocessing():
    sec_header("", "Data Preprocessing Audit")

    df = load_cleaned_data()
    if df is None:
        warn_box("Dataset not built. Go to Home → Build Dataset first.")
        return

    info_box(
        "This page audits the 7-step missing-value pipeline and shows the "
        "ColumnTransformer feature groups used in the sklearn preprocessing pipeline "

    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Cleaning Log", "🔍 Missing Strategy", "⚗️ Feature Engineering", "🏗 Pipeline Groups"
    ])

    with tab1:
        try:
            from data_cleaner import DataCleaner
            with st.spinner("Running cleaning pipeline…"):
                set_status("Running cleaning pipeline…", 35, True)
                render_status_bar()
                cleaner  = DataCleaner()
                df_clean = cleaner.fit_transform(df.copy())
            set_status("Cleaning complete", 100, False)
            render_status_bar()
            st.success(f"✅ Cleaning complete: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
            log = cleaner.report()
            if not log.empty:
                st.dataframe(log, use_container_width=True)

            # Before/after null comparison
            col1, col2 = st.columns(2)
            col1.metric("Nulls Before", f"{df.isnull().sum().sum():,}")
            col2.metric("Nulls After",  f"{df_clean.isnull().sum().sum():,}")
        except Exception as e:
            st.error(f"Could not run cleaner: {e}")
            df_clean = df.copy()

    with tab2:
        try:
            from missing_value_handler import MissingValueHandler
            handler = MissingValueHandler()
            try:
                strategy_df = handler.missing_report(df)
            except Exception:
                # Fallback: compute missing-value summary directly if handler fails
                miss = df.isnull().sum()
                pct  = (miss / len(df) * 100).round(2)
                strategy_df = (
                    pd.DataFrame({"feature": miss.index, "n_missing": miss.values, "pct_missing": pct.values})
                    .loc[lambda d: d["n_missing"] > 0]
                    .sort_values("pct_missing", ascending=False)
                    .reset_index(drop=True)
                )

            if strategy_df is None or strategy_df.empty:
                st.success("✅ No missing values detected.")
            else:
                st.markdown(f"**{len(strategy_df)} features with missing values:**")
                st.dataframe(strategy_df, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"MissingValueHandler unavailable: {e}")

        st.markdown("#### W2 Column Gap Analysis")
        info_box(
            "Wave 2 (SPSS format) truncates variable names to 8 characters. "
            "Gaps below are caused by name truncation — fixed via W2_COL_RENAME "
            "in config.py and ROSTER_COLS[2] / LABOUR_COLS[2]."
        )
        w2_gaps = {
            "head_age":           ("hh_s1q04_a → hh_s1q04", "Rename fix in ROSTER_COLS[2]"),
            "rooms":              ("hh_s9q02_a → hh_s9q02", "Rename fix in W2_COL_RENAME"),
            "hh_avg_weeks_worked":("hh_s3q21_a → hh_s3q21", "Rename fix in LABOUR_COLS[2]"),
            "head_age_sq":        ("Derived from head_age",  "Recomputed after rename fix"),
            "is_female_headed":   ("Derived from head_sex",  "Recomputed post-fill"),
            "zone_id":            ("Derived from ea_id",     "Always available"),
        }
        gap_df = pd.DataFrame(
            [(f, cause, fix) for f, (cause, fix) in w2_gaps.items()
             if f in df.columns],
            columns=["Feature","Root Cause","Fix"]
        )
        w2 = df[df["wave"] == 2]
        if not w2.empty:
            gap_df["W2 null%"]   = gap_df["Feature"].map(
                lambda c: f"{w2[c].isna().mean()*100:.1f}%" if c in w2.columns else "N/A"
            )
            gap_df["Other null%"]= gap_df["Feature"].map(
                lambda c: f"{df[df['wave']!=2][c].isna().mean()*100:.1f}%"
                          if c in df.columns else "N/A"
            )
        st.dataframe(gap_df, use_container_width=True)

    with tab3:
        try:
            from feature_enginner import FeatureEngineer
            from data_cleaner import DataCleaner
            with st.spinner("Running feature engineering…"):
                set_status("Running feature engineering…", 60, True)
                render_status_bar()
                dc = DataCleaner()
                fe = FeatureEngineer()
                df_c = dc.fit_transform(df.copy())
                df_e = fe.engineer_all(df_c)

            set_status("Feature engineering complete", 100, False)
            render_status_bar()

            st.success(f"✅ {len(fe.created_features_)} engineered features added")
            st.dataframe(fe.summary(), use_container_width=True)

            # Distribution of one key feature
            if "housing_quality_idx" in df_e.columns:
                apply_plt_style()
                fig, ax = plt.subplots(figsize=(8, 3))
                for q, c in enumerate(COLORS5, 1):
                    d = df_e.loc[df_e["cons_quint"]==q, "housing_quality_idx"].dropna()
                    ax.hist(d, bins=30, color=c, alpha=0.55, density=True,
                            label=f"Q{q}", edgecolor="none")
                ax.set_title("housing_quality_idx by Quintile", fontweight="bold")
                ax.set_xlabel("Composite Housing Quality Index (0–1)")
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
        except Exception as e:
            st.error(f"Feature engineer unavailable: {e}")

    with tab4:
        try:
            from data_preprocesor import DataPreprocessor
            dp = DataPreprocessor()
            grp = dp.feature_group_summary()
            st.dataframe(grp, use_container_width=True)

            # Show groups visually
            apply_plt_style()
            fig, ax = plt.subplots(figsize=(7, 3))
            grp["n_defined"] = grp["n_defined"].astype(int)
            bars = ax.barh(grp["group"], grp["n_defined"],
                           color=["#52B788","#F6C94E","#74C6E8","#F4830A"],
                           edgecolor="none")
            ax.set_xlabel("Number of features defined")
            ax.set_title("Feature Groups in ColumnTransformer", fontweight="bold")
            for bar, n in zip(bars, grp["n_defined"]):
                ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        str(n), va="center", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception as e:
            st.error(f"DataPreprocessor unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODELLING
# ══════════════════════════════════════════════════════════════════════════════

def page_modelling():
    sec_header("", "Supervised Learning — Model Training & Evaluation")

    df = load_cleaned_data()
    if df is None:
        warn_box("Dataset not built. Go to Home → Build Dataset first.")
        return

    info_box(
        "Train and compare all 10 classifiers. "
        "Results include accuracy, weighted F1, macro F1, and 5-fold CV scores. "
        "Per-region models are trained separately for regional wealth ranking."
    )

    tab1, tab2, tab3 = st.tabs(["Train All Models", "Results & Comparison", "Per-Region Models"])

    with tab1:
        st.markdown("#### Configure Training")
        c1, c2, c3 = st.columns(3)
        test_size  = c1.slider("Test set size (%)", 10, 30, 20) / 100
        val_size   = c2.slider("Validation set size (%)", 10, 25, 15) / 100
        cv_folds   = c3.selectbox("CV folds", [3, 5, 10], index=1)

        if st.button("Train All Models (Classifiers)", use_container_width=True):
            with st.spinner("Running full pipeline and training models…"):
                try:
                    set_status("Training all classifiers…", 45, True)
                    render_status_bar()
                    from data_cleaner      import DataCleaner
                    from feature_enginner  import FeatureEngineer
                    from data_preprocesor import DataPreprocessor
                    from modeling          import WealthPredictor

                    dc = DataCleaner()
                    fe = FeatureEngineer()
                    dp = DataPreprocessor()
                    wp = WealthPredictor()

                    df_c = dc.fit_transform(df.copy())
                    df_e = fe.engineer_all(df_c)
                    splits = dp.fit(df_e, test_size=test_size, val_size=val_size)

                    X_tr, X_te = splits["X_train"], splits["X_test"]
                    y_tr, y_te = splits["y_train"], splits["y_test"]

                    results = wp.train_evaluate(X_tr, y_tr, X_te, y_te,
                                                cv_folds=int(cv_folds))
                    dp.save(str(MODEL_DIR / "preprocessor.pkl"))
                    wp.save(str(MODEL_DIR))
                    # persist training results so metrics reappear after reload
                    try:
                        MODEL_DIR.mkdir(parents=True, exist_ok=True)
                        joblib.dump(results, MODEL_DIR / "model_results.pkl")
                    except Exception:
                        pass

                    set_status("Model training complete", 100, False)
                    render_status_bar()

                    st.session_state["model_results"]   = results
                    st.session_state["best_name"]        = wp.best_name_
                    st.session_state["splits"]           = splits
                    st.session_state["wp"]               = wp
                    st.session_state["df_engineered"]    = df_e
                    st.success(f"✅ Training complete — Best: **{wp.best_name_}**")

                except Exception as e:
                    st.error(f"Training failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab2:
        if "model_results" not in st.session_state:
            info_box("Train models first in the 🏋 Train tab.")
            return

        results = st.session_state["model_results"]
        best    = st.session_state["best_name"]

        st.markdown(f"**Best model: `{best}`**")
        st.dataframe(results.round(4), use_container_width=True)

        # Visualise comparison
        apply_plt_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # F1 bar chart
        sort_idx = results["weighted_f1"].argsort().values
        sorted_r = results.iloc[sort_idx]
        clrs     = ["#F6C94E" if m==best else "#1976D2"
                    for m in sorted_r["model"]]
        axes[0].barh(sorted_r["model"], sorted_r["weighted_f1"],
                     color=clrs, edgecolor="none")
        axes[0].set_xlabel("Weighted F1")
        axes[0].set_title("Model Comparison — Weighted F1", fontweight="bold")
        for i, v in enumerate(sorted_r["weighted_f1"]):
            axes[0].text(v+0.002, i, f"{v:.4f}", va="center", fontsize=8)

        # CV scores box-style as scatter
        axes[1].errorbar(
            range(len(results)), results["cv_f1_mean"],
            yerr=results["cv_f1_std"], fmt="o", ms=8,
            capsize=5, color="#52B788", ecolor="#30363D", lw=2,
        )
        axes[1].set_xticks(range(len(results)))
        axes[1].set_xticklabels(results["model"], rotation=40, ha="right", fontsize=8)
        axes[1].set_ylabel("CV Weighted F1 (mean ± std)")
        axes[1].set_title("5-Fold CV Scores", fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Confusion matrix
        splits = st.session_state.get("splits")
        wp     = st.session_state.get("wp")
        if splits and wp:
            try:
                cm_df = wp.confusion_matrix_df(splits["X_test"], splits["y_test"])
                apply_plt_style()
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlOrRd",
                            linewidths=0.3, ax=ax2)
                ax2.set_title(f"Confusion Matrix — {best}", fontweight="bold")
                ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()
            except Exception:
                pass

    with tab3:
        if "wp" not in st.session_state:
            info_box("Train models first to access per-region results.")
            return

        wp  = st.session_state["wp"]
        dfe = st.session_state["df_engineered"]
        feat_names = st.session_state["splits"]["feature_names"]

        if st.button("Train Per-Region Models", use_container_width=True):
            with st.spinner("Training region-specific models…"):
                try:
                    set_status("Training region-specific models…", 55, True)
                    render_status_bar()
                    reg_df  = wp.train_per_region(dfe, feat_names, test_size=0.20)
                    ranking = wp.regional_ranking(reg_df)
                    st.session_state["ranking"] = ranking
                    set_status("Regional model training complete", 100, False)
                    render_status_bar()
                    st.success(f"✅ {len(reg_df)} region models trained")
                except Exception as e:
                    st.error(f"Region modelling failed: {e}")
                if "ranking" in st.session_state:
                    st.dataframe(st.session_state["ranking"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: REGIONAL WEALTH MAP
# ══════════════════════════════════════════════════════════════════════════════

def page_regional_map():
    sec_header("", "Regional Wealth Map & Comparison")

    df = load_cleaned_data()
    if df is None:
        warn_box("Dataset not built. Go to Home → Build Dataset first.")
        return

    # Try to obtain region ranking from session, saved CSV, or compute if possible
    try:
        from src.config import RANKING_CSV
    except Exception:
        from config import RANKING_CSV

    region_stats = None
    if "ranking" in st.session_state:
        region_stats = st.session_state["ranking"]
    elif RANKING_CSV.exists():
        region_stats = pd.read_csv(RANKING_CSV)
    else:
        # attempt to compute using loaded WealthPredictor if available
        if "wp" in st.session_state and "df_engineered" in st.session_state:
            try:
                wp = st.session_state["wp"]
                dfe = st.session_state["df_engineered"]
                feat_names = st.session_state.get("splits", {}).get("feature_names", [])
                per_region = wp.train_per_region(dfe, feat_names, test_size=0.20)
                region_stats = wp.regional_ranking(per_region)
            except Exception:
                region_stats = None

    if region_stats is None or (hasattr(region_stats, "empty") and region_stats.empty):
        info_box("No regional ranking available. Train per-region models or build dataset.")
        return

    # Normalize column name for plotting logic
    if "mean_pred_quintile" in region_stats.columns and "mean_quintile" not in region_stats.columns:
        region_stats = region_stats.rename(columns={"mean_pred_quintile": "mean_quintile"})

    region_stats = (
        region_stats
        .round(3).reset_index()
        .sort_values("mean_quintile", ascending=False)
        .reset_index(drop=True)
    )
    region_stats.insert(0, "rank", range(1, len(region_stats)+1))

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        apply_plt_style()
        fig, ax = plt.subplots(figsize=(9, 7))
        norm_q  = (region_stats["mean_quintile"] - 1) / 4
        clrs    = plt.cm.RdYlGn(norm_q.values)

        bars = ax.barh(
            region_stats["region"], region_stats["mean_quintile"],
            color=clrs, edgecolor="none", height=0.65,
        )
        ax.axvline(3.0, color="#8B949E", ls="--", lw=1.5, label="Q3 national avg")
        ax.set_xlabel("Mean Predicted Quintile (1=poorest → 5=wealthiest)")
        ax.set_title("Ethiopian Regional Wealth Ranking\n(2011–2022 ESS Panel)",
                     fontweight="bold")
        ax.legend(fontsize=9)

        for bar, row in zip(bars, region_stats.itertuples()):
            ax.text(0.05, bar.get_y() + bar.get_height()/2,
                    f"#{row.rank}", va="center", ha="left",
                    fontsize=8, color="white", fontweight="bold")
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.2f}", va="center", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Regional Wealth Rankings")
        def row_html(r):
            q  = r["mean_quintile"]
            cls = "q5" if q>=4 else "q4" if q>=3.2 else "q3" if q>=2.5 else "q2" if q>=1.8 else "q1"
            return (f"<tr><td style='padding:0.4rem;color:#8B949E;'>{r['rank']}</td>"
                    f"<td style='padding:0.4rem;color:#E6EDF3;'>{r['region']}</td>"
                    f"<td style='padding:0.4rem;'><span class='q-pill {cls}'>{q:.2f}</span></td>"
                    f"<td style='padding:0.4rem;color:#8B949E;'>{int(r['n_households']):,}</td></tr>")

        table_html = """
        <table style='width:100%;border-collapse:collapse;font-size:0.83rem;'>
        <tr style='border-bottom:1px solid #30363D;'>
            <th style='padding:0.4rem;color:#52B788;'>#</th>
            <th style='color:#52B788;'>Region</th>
            <th style='color:#52B788;'>Mean Q</th>
            <th style='color:#52B788;'>HH</th>
        </tr>""" + "".join(row_html(r) for _, r in region_stats.iterrows()) + "</table>"

        st.markdown(table_html, unsafe_allow_html=True)

    # Pairwise comparison
    sec_header("", "Region Pairwise Comparison")
    regions_list = region_stats["region"].tolist()
    c1, c2 = st.columns(2)
    reg_a = c1.selectbox("Region A", regions_list, index=0)
    reg_b = c2.selectbox("Region B", regions_list, index=min(3, len(regions_list)-1))

    if reg_a != reg_b:
        row_a = region_stats[region_stats["region"]==reg_a].iloc[0]
        row_b = region_stats[region_stats["region"]==reg_b].iloc[0]
        gap   = abs(row_a["mean_quintile"] - row_b["mean_quintile"])
        richer = reg_a if row_a["mean_quintile"] >= row_b["mean_quintile"] else reg_b

        comp_data = {
            "Metric":         ["Mean Quintile","% Q1 (Poorest)","% Q5 (Richest)","N Households"],
            reg_a:            [f"{row_a['mean_quintile']:.3f}",
                               f"{row_a['pct_q1']:.1f}%",f"{row_a['pct_q5']:.1f}%",
                               f"{int(row_a['n_households']):,}"],
            reg_b:            [f"{row_b['mean_quintile']:.3f}",
                               f"{row_b['pct_q1']:.1f}%",f"{row_b['pct_q5']:.1f}%",
                               f"{int(row_b['n_households']):,}"],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        st.markdown(
            f'<div class="info-box"><strong>{richer}</strong> is wealthier on average '
            f'(gap = {gap:.3f} quintile points).</div>',
            unsafe_allow_html=True,
        )

    # By-wave trend for selected region
    sec_header("", "Quintile Trend for Selected Region")
    sel_region = st.selectbox("Select region for trend", regions_list, key="trend_region")
    region_trend = (df[df["region"].astype(str)==sel_region]
                     .groupby("wave")["cons_quint"]
                     .mean().reset_index())

    if not region_trend.empty:
        apply_plt_style()
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(region_trend["wave"], region_trend["cons_quint"],
                 "o-", color="#F6C94E", lw=2.5, ms=9)
        ax2.fill_between(region_trend["wave"], region_trend["cons_quint"],
                          alpha=0.15, color="#F6C94E")
        ax2.axhline(3, color="#8B949E", ls="--", lw=1.2, label="Q3 median")
        ax2.set_xticks(region_trend["wave"])
        ax2.set_xticklabels([f"W{w}\n{WAVE_META[w]['year']}"
                              for w in region_trend["wave"]])
        ax2.set_ylabel("Mean Quintile")
        ax2.set_title(f"Wealth Trend — {sel_region}", fontweight="bold")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def page_predict():
    sec_header("", "Single Household Wealth Quintile Prediction")

    info_box(
        "Enter household characteristics below. The model uses only "
        "<strong>proxy features</strong> — never consumption expenditure data — "
        "to predict the expected wealth quintile (1=poorest → 5=richest). "
        "All inputs mirror the actual ESS survey questions."
    )

    # Detect whether the currently loaded trained model actually uses shock inputs.
    models_for_ui = load_models()
    model_feature_names = set()
    if "preprocessor" in models_for_ui:
        try:
            _, _, feat_names_ui = models_for_ui["preprocessor"]
            model_feature_names = set(feat_names_ui)
        except Exception:
            model_feature_names = set()
    shock_feature_candidates = {
        "experienced_drought",
        "experienced_illness",
        "experienced_death",
        "experienced_crop_loss",
        "n_shocks",
        "shock_breadth",
        "is_multi_shock",
    }
    uses_shock_inputs = any(c in model_feature_names for c in shock_feature_candidates)

    # Debug helper: show which features were detected (small caption).
    # This is shown only briefly to help diagnose why shock inputs might be hidden.
    try:
        if st.session_state.get("debug_show_model_features", False):
            st.caption(f"Loaded preprocessor features: {len(model_feature_names)}")
    except Exception:
        pass

    # Defaults used when shocks are not part of the trained feature set.
    drought = False
    illness = False
    death = False
    crop_loss = False

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("#### Geography & Year")
        c1, c2, c3 = st.columns(3)
        region     = c1.selectbox("Region", REGION_LIST)
        settlement = c2.selectbox("Settlement", ["Rural","Urban","Small town","Large town"])
        
        # Allow selection of survey year
        year_options = ["2021–22 (W5, Latest)", "2018–19 (W4)", "2015–16 (W3)", "2013–14 (W2)", "2011–12 (W1)", "2027(Projection)"]
        year_selected = c3.selectbox("Survey Year", year_options, index=5)
        
        # Map year to wave number
        year_to_wave = {
            "2021–22 (W5, Latest)": 5,
            "2018–19 (W4)": 4,
            "2015–16 (W3)": 3,
            "2013–14 (W2)": 2,
            "2011–12 (W1)": 1,
            "2027(Projection)": 5,  # Use latest model for 2027 projection
        }
        wave_num = year_to_wave[year_selected]
        wave = year_selected
        
        if "2027" in year_selected:
            st.info("📊 **2027(Projection):** Using the latest trained model (2021–22 data). Actual 2027 data not yet available.")


        st.markdown("#### Household Demographics")
        c4, c5, c6 = st.columns(3)
        hh_size  = c4.number_input("Household size",    1, 20, 5)
        adulteq  = c5.number_input("Adult equivalents", 0.5, 15.0, 3.5, 0.5)
        head_age = c6.number_input("Head age (years)",  15, 90, 42)

        st.markdown("#### Head Characteristics")
        c7, c8 = st.columns(2)
        head_sex = c7.radio("Head sex", ["Male","Female"], horizontal=True)
        head_edu = c8.selectbox("Head education",
                                ["No education","Read/write only","Primary","Secondary+"])

        st.markdown("#### Housing Quality")
        c9, c10, c11 = st.columns(3)
        rooms        = c9.number_input("Rooms", 1, 15, 2)
        has_elec     = c10.checkbox("Has electricity", value=False)
        floor_type   = c11.selectbox("Floor type", ["Earth/Mud","Wood planks","Cement/tiles"])

        c12, c13, c14 = st.columns(3)
        water_src  = c12.selectbox("Water source", ["Piped/borehole","Protected well","River/rain"])
        toilet     = c13.selectbox("Toilet type",  ["Flush/VIP","Pit latrine","Open defecation"])
        fuel_type  = c14.selectbox("Cooking fuel", ["Electricity/gas","Kerosene","Wood/charcoal/dung"])

        st.markdown("#### Assets Owned")
        ca1, ca2, ca3 = st.columns(3)
        owns_phone  = ca1.checkbox("Mobile phone",   value=False)
        owns_tv     = ca2.checkbox("Television",     value=False)
        owns_fridge = ca3.checkbox("Refrigerator",   value=False)

        st.markdown("#### Labour & Enterprise")
        cl1, cl2, cl3 = st.columns(3)
        any_wage      = cl1.checkbox("Any wage earner",      value=True)
        n_workers     = cl2.number_input("# Wage earners",   0, 10, 1)
        has_enterprise= cl3.checkbox("Non-farm enterprise",  value=False)

        if uses_shock_inputs:
            st.markdown("#### Shocks Experienced")
            cs1, cs2, cs3, cs4 = st.columns(4)
            drought   = cs1.checkbox("Drought")
            illness   = cs2.checkbox("Illness")
            death     = cs3.checkbox("Death")
            crop_loss = cs4.checkbox("Crop/livestock loss")
        else:
            st.caption(
                "Shock inputs are hidden because the currently loaded trained model "
                "does not include household shock features."
            )

        submitted = st.form_submit_button(
            "Predict Wealth Quintile",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    # ── Feature encoding ───────────────────────────────────────────────────────
    settle_map = {"Urban":0, "Rural":1, "Small town":2, "Large town":3}
    edu_map    = {"No education":1,"Read/write only":2,"Primary":3,"Secondary+":4}
    floor_map  = {"Earth/Mud":0.0, "Wood planks":0.25, "Cement/tiles":1.0}
    water_q    = 1 if water_src == "Piped/borehole" else 0
    toilet_q   = 1 if toilet.startswith("Flush") else 0
    fuel_q     = 1 if fuel_type.startswith("Electricity") else 0

    housing_score = np.mean([
        has_elec, water_q, toilet_q, fuel_q,
        floor_map[floor_type],
        min(rooms/10, 1.0),
        int(owns_phone), int(owns_tv)*.5, int(owns_fridge)*.5,
    ])

    feature_dict = {
        # Core demographics
        "hh_size":              hh_size,
        "adulteq":              adulteq,
        "head_age":             head_age,
        "head_age_sq":          head_age**2,
        "head_sex":             0 if head_sex=="Female" else 1,
        "head_edu_level":       edu_map[head_edu],
        "head_literate":        1 if edu_map[head_edu] >= 2 else 0,
        "is_female_headed":     1 if head_sex=="Female" else 0,
        # Housing
        "rooms":                rooms,
        "has_electricity":      int(has_elec),
        "improved_water":       water_q,
        "improved_sanitation":  toilet_q,
        "clean_fuel":           fuel_q,
        "floor_quality":        floor_map[floor_type],
        "roof_quality":         0.7,
        "housing_score":        housing_score,
        "housing_quality_idx":  housing_score,
        # Assets
        "owns_phone":           int(owns_phone),
        "owns_tv":              int(owns_tv),
        "owns_fridge":          int(owns_fridge),
        "modern_asset_score":   int(owns_phone)*1 + int(owns_tv)*2 + int(owns_fridge)*3,
        "has_any_modern_asset": 1 if (owns_phone or owns_tv or owns_fridge) else 0,
        "assets_per_member":    (int(owns_phone)+int(owns_tv)+int(owns_fridge)) / hh_size,
        # Employment & enterprise
        "hh_any_wage_earner":   int(any_wage),
        "hh_n_workers":         n_workers,
        "hh_avg_weeks_worked":  0.0,
        "has_nonfarm_enterprise": int(has_enterprise),
        "enterprise_asset_count": 0,
        "is_fully_dependent":   1 if not any_wage else 0,
        "labour_intensity":     min(n_workers/hh_size, 1.0),
        # Derived metrics
        "log_hh_size":          np.log1p(hh_size),
        "dependency_ratio":     max(0, (hh_size - adulteq) / hh_size),
        "adults_ratio":         min(adulteq / hh_size, 1.0),
        "is_large_hh":          1 if hh_size >= 7 else 0,
        "is_single_person":     1 if hh_size == 1 else 0,
        # Geospatial & temporal
        "region":               region,
        "settlement":           settle_map[settlement],
        "is_urban":             1 if settlement=="Urban" else 0,
        "is_addis":             1 if region=="ADDIS ABABA" else 0,
        "is_peripheral":        1 if region in ["AFAR","SOMALI","GAMBELA","BENISHANGUL GUMUZ"] else 0,
        "wave":                 wave_num,
        "post_covid":           1 if wave_num == 5 else 0,
        "has_full_housing":     1 if wave_num in (1,2,3) else 0,
        "has_enterprise_data":  1 if wave_num in (1,2,3) else 0,
        # Head characteristics
        "head_prime_working_age": 1 if 25<=head_age<=55 else 0,
        "head_elderly":         1 if head_age>=60 else 0,
        "educated_prime_head":  1 if (edu_map[head_edu]>=3 and 25<=head_age<=55) else 0,
        # Housing quality codes (for compatibility with preprocessing)
        "roof":                 2,
        "wall":                 2,
        "floor":                1,
        "water":                1 if water_q else 3,
        "toilet":               1 if toilet_q else 3,
        "fuel":                 1 if fuel_q else 4,
    }

    input_df = pd.DataFrame([feature_dict])

    # ── Try loaded preprocessor + model ────────────────────────────────────
    pred_q    = None
    pred_prob = None

    # Try session state first
    wp     = st.session_state.get("wp")
    splits = st.session_state.get("splits")

    if wp and splits:
        try:
            from data_preprocesor import DataPreprocessor
            dp  = DataPreprocessor()
            dp.pipeline_ = st.session_state.get("dp_pipeline")
            if dp.pipeline_ is None:
                raise ValueError("No pipeline in session")
            X_new    = dp.pipeline_.transform(input_df[[c for c in dp.feature_names_ if c in input_df.columns]])
            pred_q   = int(wp.best_model_.predict(X_new)[0])
            pred_prob= wp.best_model_.predict_proba(X_new)[0]
        except Exception:
            wp = None

    # Try saved model files
    if pred_q is None:
        models = load_models()
        if "preprocessor" in models and "best_model" in models:
            try:
                pipeline_, _, feat_names_ = models["preprocessor"]
                # Build full dataframe matching the preprocessor's expected features.
                # Only use features that the preprocessor trained on
                full = pd.DataFrame(index=[0], columns=feat_names_)
                for c in feat_names_:
                    if c in feature_dict:
                        full.loc[0, c] = feature_dict[c]
                    elif c in input_df.columns:
                        full.loc[0, c] = input_df.iloc[0].get(c)
                    else:
                        # Default fill based on feature type
                        if c in ['region', 'settlement', 'head_edu_level', 'roof', 'wall', 'floor', 'water', 'toilet', 'fuel']:
                            full.loc[0, c] = "missing"  # Categorical - will be imputed
                        elif c in ['head_sex', 'is_female_headed', 'head_literate', 'hh_any_wage_earner', 
                                  'has_nonfarm_enterprise', 'has_electricity', 'owns_phone', 'owns_tv', 'owns_fridge',
                                  'post_covid', 'has_full_housing', 'has_enterprise_data', 'is_large_hh', 
                                  'is_single_person', 'improved_water', 'improved_sanitation', 'clean_fuel',
                                  'has_any_modern_asset', 'is_fully_dependent', 'is_urban', 'is_addis',
                                  'is_peripheral', 'head_prime_working_age', 'head_elderly', 'educated_prime_head']:
                            full.loc[0, c] = 0.0  # Binary - default to 0
                        else:
                            full.loc[0, c] = np.nan  # Numeric - will be imputed by median
                
                # Convert all numeric columns to float64 (required by sklearn pipeline)
                for c in full.columns:
                    if c not in ['region', 'settlement', 'head_edu_level', 'roof', 'wall', 'floor', 'water', 'toilet', 'fuel']:
                        full[c] = full[c].astype('float64')
                X_new = pipeline_.transform(full)
                
                # Handle classification model → direct quintile prediction
                model = models["best_model"]
                pred_q = int(model.predict(X_new)[0])
                pred_prob = model.predict_proba(X_new)[0]
            except Exception as model_err:
                # Silent fallback to heuristic - error will be shown via heuristic caption
                pass
                pass

    # Heuristic fallback
    if pred_q is None:
        score = (
            housing_score * 2.5
            + (int(owns_phone)+int(owns_tv)*2+int(owns_fridge)*3) * 0.4
            + int(has_elec) * 0.6
            + (edu_map[head_edu]-1) * 0.25
            + int(any_wage) * 0.4
            + int(has_enterprise) * 0.3
            + int(settlement=="Urban") * 0.5
            + int(region=="ADDIS ABABA") * 0.8
            - int(crop_loss) * 0.3
            - int(drought) * 0.2
            - max(0, (hh_size-5)*0.05)
        )
        pred_q = max(1, min(5, int(np.round(np.clip(score*1.6, 1, 5)))))
        probs  = np.zeros(5)
        probs[pred_q-1] = 0.5
        for q in range(5):
            probs[q] += 0.1 * max(0, 1 - abs(q+1-pred_q))
        pred_prob = probs / probs.sum()
        
        # Show heuristic message only when using fallback
        st.caption("Using heuristic estimation - train models for ML-based prediction.")

    # ── Result display ─────────────────────────────────────────────────────
    lbl, _, icon = QUINTILE_LABELS[pred_q]

    c_res, c_prob = st.columns([1, 1.2], gap="large")

    with c_res:
        st.markdown(f"""
        <div class="pred-result">
            <div style="font-size:1.1rem;color:#8B949E;margin-bottom:0.5rem;">
                Predicted Wealth Quintile
            </div>
            <div class="pred-quintile">{icon} Q{pred_q}</div>
            <div class="pred-label">{lbl}</div>
            <div style="margin-top:1rem;font-size:0.82rem;color:#8B949E;line-height:1.6;">
                <strong style="color:#52B788;">Region:</strong> {region} &nbsp;|&nbsp;
                <strong style="color:#52B788;">Wave:</strong> {wave}<br>
                <strong style="color:#52B788;">Settlement:</strong> {settlement} &nbsp;|&nbsp;
                <strong style="color:#52B788;">HH size:</strong> {hh_size}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary of key inputs
        st.markdown("**Key inputs summary**")
        summary_features = ["Housing score", "Asset count", "Wage earners", "Head age", "Education"]
        summary_values = [
            f"{housing_score:.2f}",
            str(int(owns_phone)+int(owns_tv)+int(owns_fridge)),
            str(n_workers),
            str(head_age),
            head_edu,
        ]
        if uses_shock_inputs:
            summary_features.insert(3, "Shocks")
            summary_values.insert(3, str(int(drought)+int(illness)+int(death)+int(crop_loss)))

        summary = pd.DataFrame({"Feature": summary_features, "Value": summary_values})
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with c_prob:
        if pred_prob is not None:
            apply_plt_style()
            fig, ax = plt.subplots(figsize=(7, 4))
            quintiles = [f"Q{i}" for i in range(1,6)]
            bars = ax.bar(quintiles, pred_prob * 100, color=COLORS5,
                           edgecolor="none", width=0.6)
            bars[pred_q-1].set_edgecolor("#F6C94E")
            bars[pred_q-1].set_linewidth(3)
            ax.set_ylabel("Predicted Probability (%)")
            ax.set_title("Quintile Probability Distribution", fontweight="bold")
            for bar, p in zip(bars, pred_prob):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f"{p*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
            ax.set_ylim(0, max(pred_prob)*130)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Contextual interpretation
        interpretations = {
            1: "This household falls in the <strong>poorest 20%</strong> of Ethiopian households. "
               "Characteristics include: limited or no asset ownership, earth-floor dwelling, "
               "reliance on surface water, and high shock exposure.",
            2: "This household is in the <strong>lower-middle quintile (Q2)</strong>. "
               "Some access to basic services but still facing significant welfare gaps.",
            3: "This household is at the <strong>national median (Q3)</strong> — "
               "moderate asset ownership and housing quality typical of rural Ethiopia.",
            4: "This household is in the <strong>upper-middle quintile (Q4)</strong>. "
               "Better access to utilities, assets, and labour market participation.",
            5: "This household is in the <strong>wealthiest 20% (Q5)</strong>. "
               "Strong asset base, quality housing, and urban location are key drivers.",
        }
        st.markdown(
            f'<div class="info-box">{interpretations[pred_q]}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: ABOUT 
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    sec_header("", "About This Project")

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown("""
        <div style="color:#8B949E;font-size:0.93rem;line-height:1.8;">

        <h4 style="color:#E6EDF3;margin-bottom:0.5rem;">Project Summary</h4>
        This project applies the <strong style="color:#F6C94E;">CRISP-DM methodology</strong>
        to predict Ethiopian household wealth quintiles from five rounds of the
        World Bank Living Standards Measurement Study (LSMS/ESS) surveys
        covering 2011–2022.

        <br><br>
        The prediction target <code style="color:#52B788;">cons_quint</code> is a
        consumption-expenditure-based welfare quintile (1=poorest → 5=wealthiest),
        computed by the World Bank from comprehensive expenditure diaries.
        Because the target is derived from consumption data, all consumption
        aggregates are strictly excluded from the feature set to prevent
        <strong style="color:#C8102E;">data leakage</strong>.

        <br><br>
        <h4 style="color:#E6EDF3;margin-top:1rem;">Key Technical Contributions</h4>
        <ul>
        <li><strong>W2 SPSS decoder</strong>: Pure-Python bytecode SAV reader (no pyreadstat)
            with automatic 8-char truncation correction</li>
        <li><strong>7-step survey-aware imputation</strong>: MNAR flags, W2 cross-wave
            donor fill, group medians, KNN</li>
        <li><strong>Domain feature engineering</strong>: 7 groups grounded in poverty
            literature (housing index, asset score, dependency, vulnerability)</li>
        <li><strong>Per-region models</strong>: 11 region-specific classifiers for
            regional wealth ranking and pairwise comparison</li>
        <li><strong>Unsupervised analysis</strong>: K-Means, hierarchical clustering,
            PCA and t-SNE projections</li>
        </ul>

        <h4 style="color:#E6EDF3;margin-top:1rem;">Data Source</h4>
        World Bank LSMS-ISA / Ethiopian Socioeconomic Survey (ESS)<br>
        <code style="color:#52B788;">https://microdata.worldbank.org/catalog/2053</code>
        <br>License: World Bank Open Data (CC BY 4.0)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="color:#8B949E;font-size:0.93rem;line-height:1.8;">

        <h4 style="color:#C8102E;">Ethical Considerations</h4>

        <div class="warn-box">
        <strong>Fairness & Bias</strong>: Wealth prediction models trained on
        historical survey data may reflect systemic inequalities (e.g. gender gaps,
        regional disparities). Predictions should never be used to deny services
        or discriminate against individuals.
        </div>

        <div class="warn-box" style="margin-top:0.8rem;">
        <strong>Privacy</strong>: All data is fully anonymised by the World Bank.
        No personally identifiable information (PII) is stored or used. Household
        IDs in the raw data are survey codes only.
        </div>

        <div class="warn-box" style="margin-top:0.8rem;">
        <strong>Conflict-affected data (W5)</strong>: Wave 5 (2021-22) was collected
        during the Tigray conflict. Tigray region data may be incomplete or
        unrepresentative. The model includes an <code>is_tigray_conflict</code>
        flag to capture this structural break.
        </div>

        <div class="info-box" style="margin-top:0.8rem;">
        <strong>Intended use</strong>: This system is designed for academic research
        and policy analysis only. It should not be used as the sole basis for
        welfare targeting, benefit allocation, or any consequential decision
        affecting individual households.
        </div>

        <h4 style="color:#E6EDF3;margin-top:1.2rem;">Team</h4>
        <strong>Instructor:</strong> Petros Abebe (MSc) <br>
        <strong>Course:</strong> InSy3056 Data Science Application<br>
        </strong> Debre Berhan University<br>
        College of Computing, Department of Information Systems<br>
        <strong>Academic Year:</strong> 2025/26

        <div class='team-card' style='margin-top:0.6rem;'>
            <div class='team-title' style='font-size:0.9rem;'>Project Team</div>
            <table class='team-table'>
                <thead>
                    <tr>
                        <th class='team-no'>No.</th>
                        <th>Name</th>
                        <th class='team-id-cell'>ID No.</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td class='team-no'>1</td><td>Mesfin Maru</td><td class='team-id-cell'>DBU1702015</td></tr>
                    <tr><td class='team-no'>2</td><td>Ayires Zebene</td><td class='team-id-cell'>DBU1601409</td></tr>
                    <tr><td class='team-no'>3</td><td>Melkamsew Alehegn</td><td class='team-id-cell'>DBU1601356</td></tr>
                    <tr><td class='team-no'>4</td><td>Kalkidan Minda</td><td class='team-id-cell'>DBU1601280</td></tr>
                    <tr><td class='team-no'>5</td><td>Sofia Mohammed</td><td class='team-id-cell'>DBU1501477</td></tr>
                    <tr><td class='team-no'>6</td><td>Samrawit Assefa</td><td class='team-id-cell'>DBU1601585</td></tr>
                </tbody>
            </table>
        </div>
        </div>
        """, unsafe_allow_html=True)

    sec_header("", "Core Methods and Tools Used")
    methods = pd.DataFrame([
        ("Data preparation",   "MissingValueHandler",   "Survey-aware imputation strategies including mean, median, mode, constant, KNN, iterative and forward fill"),
        ("Data preparation",   "DataCleaner",           "Missing-value handling, outlier detection and capping, and coverage-based data quality controls"),
        ("Data preparation",   "DataPreprocessor",      "ColumnTransformer pipeline with stratified splits, one-hot encoding, ordinal encoding and scaling"),
        ("Exploratory analysis","Univariate analysis",   "Distribution summaries, histograms, box plots, density plots and frequency tables"),
        ("Exploratory analysis","Bivariate analysis",    "Quintile comparison plots, Kruskal-Wallis, Chi-square and Mann-Whitney U testing"),
        ("Exploratory analysis","Multivariate analysis", "Correlation analysis, region-settlement heatmaps and PCA-based structure review"),
        ("Supervised learning", "Classification pipeline","Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, AdaBoost, Gradient Boosting, XGBoost and LightGBM"),
        ("Supervised learning", "Regression pipeline",    "Ordinal prediction metrics including MAE, RMSE and R² for wealth quintile estimation"),
        ("Unsupervised learning","Clustering analyzer",   "K-Means with elbow and silhouette support, hierarchical clustering and DBSCAN"),
        ("Unsupervised learning","Dimensionality reduction","PCA scree analysis, t-SNE projection and LDA-based class separation"),
        ("Model evaluation",   "Evaluator",             "Stratified K-Fold validation, learning curves, validation curves and GridSearchCV"),
        ("Model evaluation",   "Statistical testing",   "Paired t-test and Wilcoxon signed-rank comparison across models"),
    ], columns=["Area","Method/Tool","Details"])
    st.dataframe(methods, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if   "Home"            in page:  page_home()
    elif "Data Explorer"   in page:  page_data_explorer()
    elif "EDA"             in page:  page_eda()
    elif "Preprocessing"   in page:  page_preprocessing()
    elif "Modelling"       in page:  page_modelling()
    elif "Regional"        in page:  page_regional_map()
    elif "Predict"         in page:  page_predict()
    elif "About"           in page:  page_about()

    render_status_bar()

    # Footer
    st.markdown("""
    <div class="footer">
        Ethiopian Household Wealth Predictor &nbsp;·&nbsp;
        <span style="color:#30363D;">
            Data: World Bank LSMS-ISA ESS (2011–2022)
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()