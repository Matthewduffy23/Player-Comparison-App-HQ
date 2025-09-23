# app.py — SB-style radar with BIG top tabs (5 roles)
# Percentiles for plotting + raw-value ring labels
# Solid fills, alternating bands, subtle tick labels, 11 rings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from pathlib import Path
import io
import re

st.set_page_config(page_title="Player Comparison — SB Radar", layout="wide")

# ====== make the tabs large & obvious at the very top ======
st.markdown(
    """
    <style>
      /* make tabs big, pill-like, and spaced */
      section[data-testid="stTabs"] div[role="tablist"] {
        gap: 10px;
      }
      section[data-testid="stTabs"] button[role="tab"] {
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 700;
        border: 1px solid #E5E7EB;
        background: #F8FAFC;
      }
      section[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #E5F0FF;
        border-color: #93C5FD;
        color: #1D4ED8;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Player Comparison — SB Radar")

# ---------------- Theme ----------------
COL_A = "#C81E1E"          # deep red
COL_B = "#1D4ED8"          # deep blue
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

df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
df["Age"]            = pd.to_numeric(df["Age"], errors="coerce")

# -------------- Role metrics (your spec) ---------------
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
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Accurate passes, %", "Pass %")
    s = s.replace("Forward passes per 90", "Forward passes")
    s = s.replace("Progressive passes per 90", "Prog passes")
    s = s.replace("Progressive runs per 90", "Prog runs")
    s = s.replace("Passes to penalty area per 90", "Passes to PA")
    s = s.replace("Deep completions per 90", "Deep completions")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Aerial duels won, %", "Aerial %")
    s = s.replace("Defensive duels per 90", "Defensive duels")
    s = s.replace("Defensive duels won, %", "Defensive duel %")
    s = s.replace("Successful dribbles, %", "Dribble %")
    s = s.replace("Dribbles per 90", "Dribbles")
    s = s.replace("PAdj Interceptions", "PAdj Interceptions")
    s = s.replace("Shots per 90", "Shots")
    s = re.sub(r"\s*per\s*90", "", s, flags=re.I)
    return s

# -------------- Position filters --------------
def attackers_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    prefixes = ('RWF','LWF','LAMF','RAMF','AMF','RW,','LW,')
    return s.isin(['RW','LW']) | s.str.startswith(prefixes)

def group_mask(series: pd.Series, group: str) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    if group == "Centre Backs":
        return s.str.startswith(('LCB','RCB','CB'))
    if group == "Full Backs":
        return s.str.startswith(('LB','LWB','RB','RWB'))
    if group == "Midfielders":
        return s.str.startswith(('LCMF','RCMF','LDMF','RDMF','DMF'))
    if group == "Attackers":
        return attackers_mask(series)
    if group == "Forwards":
        return s.str.startswith(('CF',))
    return pd.Series([True]*len(series), index=series.index)

# -------------- Radar drawer --------------
ring_radii = np.linspace(INNER_HOLE, 100, NUM_RINGS)

def draw_radar(labels, A_r, B_r, ticks, headerA, subA, subA2, headerB, subB, subB2,
               show_avg=False, AVG_r=None):
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    Ar = np.concatenate([A_r, A_r[:1]])
    Br = np.concatenate([B_r, B_r[:1]])

    fig = plt.figure(figsize=(13.2, 8.0), dpi=260)
    fig.patch.set_facecolor(PAGE_BG)

    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(AX_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=AXIS_FS, color=LABEL_COLOR, fontweight=600)
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(NUM_RINGS-1):
        r0, r1 = ring_radii[i], ring_radii[i+1]
        band = GRID_BAND_A if i % 2 == 0 else GRID_BAND_B
        ax.add_artist(Wedge((0,0), r1, 0, 360, width=(r1-r0),
                            transform=ax.transData._b, facecolor=band,
                            edgecolor="none", zorder=0.8))

    ring_t = np.linspace(0, 2*np.pi, 361)
    for r in ring_radii:
        ax.plot(ring_t, np.full_like(ring_t, r), color=RING_COLOR, lw=RING_LW, zorder=0.9)

    start_idx = 2
    for i, ang in enumerate(theta):
        vals = ticks[i][start_idx:]
        for rr, v in zip(ring_radii[start_idx:], vals):
            ax.text(ang, rr-1.8, f"{v:.1f}", ha="center", va="center",
                    fontsize=TICK_FS, color=TICK_COLOR, zorder=1.1)

    ax.add_artist(Circle((0,0), radius=INNER_HOLE-0.6, transform=ax.transData._b,
                         color=PAGE_BG, zorder=1.2, ec="none"))

    if show_avg and AVG_r is not None:
        Avg = np.concatenate([AVG_r, AVG_r[:1]])
        ax.plot(theta_closed, Avg, lw=1.5, color="#94A3B8", ls="--", alpha=0.9, zorder=2.2)

    ax.plot(theta_closed, Ar, color=COL_A, lw=2.2, zorder=3)
    ax.fill(theta_closed, Ar, color=FILL_A, zorder=2.5)

    ax.plot(theta_closed, Br, color=COL_B, lw=2.2, zorder=3)
    ax.fill(theta_closed, Br, color=FILL_B, zorder=2.5)

    ax.set_rlim(0, 105)

    fig.text(0.12, 0.96,  headerA, color=COL_A, fontsize=TITLE_FS, fontweight="bold", ha="left")
    fig.text(0.12, 0.935, subA,    color=COL_A, fontsize=SUB_FS,      ha="left")
    fig.text(0.12, 0.915, subA2,   color=MINUTES_COLOR, fontsize=MINUTES_FS, ha="left")

    fig.text(0.88, 0.96,  headerB, color=COL_B, fontsize=TITLE_FS, fontweight="bold", ha="right")
    fig.text(0.88, 0.935, subB,    color=COL_B, fontsize=SUB_FS,      ha="right")
    fig.text(0.88, 0.915, subB2,   color=MINUTES_COLOR, fontsize=MINUTES_FS, ha="right")

    if show_avg and AVG_r is not None:
        fig.text(0.2, 0.1, "— Average / 50th Percentile | Stats per 90",
                 color="#6B7280", fontsize=8, ha="center")

    return fig

def fmt_minutes(x):
    try:
        if pd.notna(x):
            return f"{int(round(float(x))):,} mins"
    except Exception:
        pass
    return "Minutes: N/A"

# -------------- Page builder per role --------------
def build_role_page(role_name: str):
    role_defaults = [m for m in ROLE_METRICS.get(role_name, []) if m in df.columns]

    left, right = st.columns([1, 3], vertical_alignment="top")
    with left:
        st.subheader(f"Controls ({role_name.lower()} only)")

        min_minutes, max_minutes = st.slider("Minutes filter", 0, 5000, (500, 5000), key=f"{role_name}_mins")
        min_age, max_age = st.slider(
            "Age filter",
            int(np.nanmin(df["Age"]) if pd.notna(df["Age"]).any() else 14),
            int(np.nanmax(df["Age"]) if pd.notna(df["Age"]).any() else 40),
            (16, 33),
            key=f"{role_name}_age"
        )

        mask_picker = group_mask(df["Position"], role_name)
        picker_pool = df[mask_picker].copy()
        players = sorted(picker_pool["Player"].dropna().unique().tolist())
        if len(players) < 2:
            st.error("Not enough players for this filter.")
            st.stop()

        pA = st.selectbox("Player A (red)", players, index=0, key=f"{role_name}_pA")
        pB = st.selectbox("Player B (blue)", players, index=1, key=f"{role_name}_pB")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        metrics = st.multiselect("Metrics",
                                 [c for c in df.columns if c in numeric_cols],
                                 role_defaults,
                                 key=f"{role_name}_metrics")
        if len(metrics) < 5:
            st.warning("Pick at least 5 metrics.")
            st.stop()

        sort_by_gap = st.checkbox("Sort axes by biggest gap", False, key=f"{role_name}_gap")
        show_avg    = st.checkbox("Show pool average (thin line)", True, key=f"{role_name}_avg")

    # ----- data slice -----
    rowA = df[df["Player"] == pA].iloc[0]
    rowB = df[df["Player"] == pB].iloc[0]

    union_leagues = {rowA["League"], rowB["League"]}
    mask_pool = (
        group_mask(df["Position"], role_name) &
        df["League"].isin(union_leagues) &
        df["Minutes played"].between(min_minutes, max_minutes) &
        df["Age"].between(min_age, max_age)
    )
    pool = df[mask_pool].copy()

    missing_m = [m for m in metrics if m not in pool.columns]
    if missing_m:
        st.error(f"Missing metric columns: {missing_m}")
        st.stop()

    for m in metrics:
        pool[m] = pd.to_numeric(pool[m], errors="coerce")
    pool = pool.dropna(subset=metrics)
    if pool.empty:
        st.warning("No players remain in pool after filters.")
        st.stop()

    labels = [clean_label(m) for m in metrics]
    pool_pct = pool[metrics].rank(pct=True) * 100.0

    def pct_for(player: str) -> np.ndarray:
        idx = pool[pool["Player"] == player].index
        if len(idx) == 0:
            return np.full(len(metrics), np.nan)
        return pool_pct.loc[idx, :].mean(axis=0).values

    A_r = pct_for(pA)
    B_r = pct_for(pB)
    AVG_r = np.full(len(metrics), 50.0)

    axis_min = pool[metrics].min().values
    axis_max = pool[metrics].max().values
    pad = (axis_max - axis_min) * 0.07
    axis_min = axis_min - pad
    axis_max = axis_max + pad
    axis_ticks = [np.linspace(axis_min[i], axis_max[i], NUM_RINGS) for i in range(len(labels))]

    if sort_by_gap:
        order = np.argsort(-np.abs(A_r - B_r))
        labels    = [labels[i] for i in order]
        A_r       = A_r[order]
        B_r       = B_r[order]
        AVG_r     = AVG_r[order]
        axis_ticks = [axis_ticks[i] for i in order]

    minsA = fmt_minutes(rowA.get("Minutes played"))
    minsB = fmt_minutes(rowB.get("Minutes played"))
    headerA = f"{pA}"
    subA    = f"{rowA['Team']} — {rowA['League']}"
    subA2   = f"{minsA}"
    headerB = f"{pB}"
    subB    = f"{rowB['Team']} — {rowB['League']}"
    subB2   = f"{minsB}"

    with right:
        fig = draw_radar(labels, A_r, B_r, axis_ticks,
                         headerA, subA, subA2, headerB, subB, subB2,
                         show_avg=show_avg, AVG_r=AVG_r)
        st.pyplot(fig, use_container_width=True)

        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=340, bbox_inches="tight")
        st.download_button("⬇️ Download PNG", data=buf_png.getvalue(),
                           file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar_SB.png",
                           mime="image/png", key=f"{role_name}_png")

        buf_svg = io.BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
        st.download_button("⬇️ Download SVG", data=buf_svg.getvalue(),
                           file_name=f"{pA.replace(' ','_')}_vs_{pB.replace(' ','_')}_radar_SB.svg",
                           mime="image/svg+xml", key=f"{role_name}_svg")

# ====== THE BIG, CLEAR TOGGLE ======
tabs = st.tabs(["Centre Backs", "Full Backs", "Midfielders", "Attackers", "Forwards"])

with tabs[0]:
    build_role_page("Centre Backs")
with tabs[1]:
    build_role_page("Full Backs")
with tabs[2]:
    build_role_page("Midfielders")
with tabs[3]:
    build_role_page("Attackers")
with tabs[4]:
    build_role_page("Forwards")





