# app.py — SB-style radar (multi-position with role metrics)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from pathlib import Path
import io
import re

st.set_page_config(page_title="Player Comparison — SB Radar", layout="wide")

# ---------------- Theme ----------------
COL_A = "#C81E1E"
COL_B = "#1D4ED8"
FILL_A = (200/255, 30/255, 30/255, 0.60)
FILL_B = (29/255, 78/255, 216/255, 0.60)

PAGE_BG   = "#FFFFFF"
AX_BG     = "#FFFFFF"

GRID_BAND_A = "#FFFFFF"
GRID_BAND_B = "#E5E7EB"
RING_COLOR  = "#D1D5DB"
RING_LW     = 1.0

LABEL_COLOR = "#0F172A"
TITLE_FS    = 26
SUB_FS      = 12
AXIS_FS     = 10
TICK_FS     = 7
TICK_COLOR  = "#9CA3AF"
MINUTES_FS    = 10
MINUTES_COLOR = "#374151"

NUM_RINGS   = 11
INNER_HOLE  = 10

# -------------- Data ---------------
@st.cache_data(show_spinner=False)
def load_df():
    p = Path(__file__).with_name("WORLDJUNE25.csv")
    return pd.read_csv(p) if p.exists() else None

df = load_df()
if df is None:
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not up:
        st.warning("Upload dataset to continue.")
        st.stop()
    df = pd.read_csv(up)

required = {"Player","League","Team","Position","Minutes played","Age"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# -------------- Role metrics ---------------
CB_METRICS = [
    "Aerial duels per 90","Aerial duels won, %","Defensive duels per 90",
    "Defensive duels won, %","PAdj Interceptions","Passes per 90",
    "Accurate passes, %","Progressive passes per 90","Forward passes per 90",
    "Progressive runs per 90","Dribbles per 90","Accurate passes, %"
]

FB_METRICS = [
    "Aerial duels won, %","Defensive duels per 90","Defensive duels won, %",
    "PAdj Interceptions","Passes per 90","Progressive passes per 90",
    "Forward passes per 90","Progressive runs per 90","Dribbles per 90",
    "xA per 90","Passes to penalty area per 90"
]

CM_METRICS = [
    "Defensive duels per 90","Defensive duels won, %","PAdj Interceptions",
    "Passes per 90","Accurate passes, %","Progressive passes per 90",
    "Non-penalty goals per 90","Progressive runs per 90","Dribbles per 90",
    "xA per 90","Passes to penalty area per 90"
]

ATT_METRICS = [
    "Aerial duels won, %","Defensive duels per 90","Passes per 90",
    "Accurate passes, %","Passes to penalty area per 90","Deep completions per 90",
    "Non-penalty goals per 90","xG per 90","Progressive runs per 90",
    "Dribbles per 90","xA per 90"
]

ST_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90",
    "Dribbles per 90","Successful dribbles, %","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90"
]

ROLE_METRICS = {
    "Centre Backs": CB_METRICS,
    "Full Backs": FB_METRICS,
    "Midfielders": CM_METRICS,
    "Attackers": ATT_METRICS,
    "Forwards": ST_METRICS,
}

# -------------- Label cleaner ---------------
def clean_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Forward passes per 90", "Forward passes")
    s = s.replace("Progressive passes per 90", "Prog passes")
    s = s.replace("Progressive runs per 90", "Prog runs")
    s = s.replace("Passes to penalty area per 90", "Passes to PA")
    s = s.replace("Deep completions per 90", "Deep completions")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Aerial duels won, %", "Aerial %")
    s = s.replace("Defensive duels per 90", "Def duels")
    s = s.replace("Defensive duels won, %", "Def duels %")
    s = s.replace("Successful dribbles, %", "Dribble %")
    s = s.replace("Dribbles per 90", "Dribbles")
    s = s.replace("Accurate passes, %", "Pass %")
    s = s.replace("PAdj Interceptions", "Adj Interceptions")
    s = re.sub(r"\s*per\s*90", "", s, flags=re.I)
    return s

# ------------ Position group switch ------------
pos_group = st.radio(
    "Position Group",
    ("Centre Backs", "Full Backs", "Midfielders", "Attackers", "Forwards"),
    horizontal=True,
    index=0
)

# ------------ Position filters ------------
def attackers_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    prefixes = ('RWF','LWF','LAMF','RAMF','AMF','RW,','LW,')
    return s.isin(['RW','LW']) | s.str.startswith(prefixes)

def group_mask(series: pd.Series, group: str) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    if group == "Centre Backs":
        return s.str.startswith(('LCB','RCB','CB'))
    elif group == "Full Backs":
        return s.str.startswith(('LB','LWB','RB','RWB'))
    elif group == "Midfielders":
        return s.str.startswith(('LCMF','RCMF','LDMF','RDMF','DMF'))
    elif group == "Attackers":
        return attackers_mask(series)
    elif group == "Forwards":
        return s.str.startswith(('CF',))
    else:
        return pd.Series([True]*len(series), index=series.index)

# -------------- Sidebar --------------
with st.sidebar:
    st.header("Controls")

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"]            = pd.to_numeric(df["Age"], errors="coerce")

    min_minutes, max_minutes = st.slider("Minutes filter", 0, 5000, (500, 5000))
    min_age, max_age         = st.slider(
        "Age filter",
        int(np.nanmin(df["Age"]) if pd.notna(df["Age"]).any() else 14),
        int(np.nanmax(df["Age"]) if pd.notna(df["Age"]).any() else 40),
        (16, 33)
    )

    mask_picker = group_mask(df["Position"], pos_group)
    picker_pool = df[mask_picker].copy()
    players = sorted(picker_pool["Player"].dropna().unique().tolist())
    if len(players) < 2:
        st.error("Not enough players for this filter.")
        st.stop()

    pA = st.selectbox("Player A (red)", players, index=0)
    pB = st.selectbox("Player B (blue)", players, index=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    role_defaults = [m for m in ROLE_METRICS.get(pos_group, []) if m in df.columns]

    metrics = st.multiselect(
        "Metrics",
        [c for c in df.columns if c in numeric_cols],
        role_defaults
    )
    if len(metrics) < 5:
        st.warning("Pick at least 5 metrics.")
        st.stop()

    sort_by_gap = st.checkbox("Sort axes by biggest gap", False)
    show_avg    = st.checkbox("Show pool average (thin line)", True)

# ---------------- Radar + rest of your code (unchanged) ----------------
# (same as before: build pool, percentiles, radar drawer, download buttons)
