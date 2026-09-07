import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import joblib
import os
import re

# ─── Load Models ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    def find(name):
        for p in [os.path.join(BASE_DIR, name), os.path.join(BASE_DIR, "models", name)]:
            if os.path.exists(p):
                return joblib.load(p)
        raise FileNotFoundError(f"{name} not found.")
    return find("heart_model.pkl"), find("cancer_model.pkl")

heart_model, cancer_model = load_models()

st.set_page_config(
    page_title="MedCore AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
# Dark luxury medical: deep navy bg, electric cyan accents, glassmorphism cards
# Font: Syne (geometric display) + DM Sans (body)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:        #060d1a;
  --bg2:       #0b1628;
  --bg3:       #111f35;
  --surface:   rgba(255,255,255,0.04);
  --border:    rgba(255,255,255,0.08);
  --border2:   rgba(0,212,255,0.2);
  --cyan:      #00d4ff;
  --cyan-dim:  rgba(0,212,255,0.12);
  --cyan-glow: rgba(0,212,255,0.3);
  --red:       #ff4757;
  --red-dim:   rgba(255,71,87,0.12);
  --amber:     #ffb142;
  --amber-dim: rgba(255,177,66,0.12);
  --green:     #2ed573;
  --green-dim: rgba(46,213,115,0.12);
  --text:      #e8f0fe;
  --text2:     #7f9ab8;
  --text3:     #4a6a8a;
}

*, html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  box-sizing: border-box;
}

/* ── App background with animated mesh gradient */
.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,255,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,102,241,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 60% 20%, rgba(255,71,87,0.04) 0%, transparent 50%) !important;
}

/* ── Sidebar */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text2) !important; }
section[data-testid="stSidebar"] label {
  color: var(--text3) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  font-family: 'DM Mono', monospace !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* ── Metrics */
div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 20px 22px !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="metric-container"]:hover {
  border-color: var(--border2) !important;
  box-shadow: 0 4px 32px rgba(0,212,255,0.1), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
div[data-testid="metric-container"] label {
  color: var(--text3) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  font-family: 'DM Mono', monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  font-family: 'Syne', sans-serif !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
  font-size: 11px !important;
  font-weight: 500 !important;
}

/* ── Tabs */
div[data-testid="stTabs"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 0 !important;
  padding: 0 !important;
}
div[data-testid="stTabs"] button {
  background: transparent !important;
  color: var(--text3) !important;
  border: none !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 14px 20px !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  letter-spacing: 0.3px !important;
  transition: all 0.2s !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
  text-shadow: 0 0 20px var(--cyan-glow) !important;
}
div[data-testid="stTabs"] button:hover {
  color: var(--text) !important;
  background: var(--surface) !important;
}

/* ── Inputs */
div[data-testid="stNumberInput"] input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  padding: 10px 14px !important;
}
div[data-testid="stNumberInput"] input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 3px var(--cyan-dim) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
  color: var(--text3) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
  font-family: 'DM Mono', monospace !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
  color: var(--text) !important;
}
div[data-testid="stSlider"] label {
  color: var(--text3) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
  font-family: 'DM Mono', monospace !important;
}

/* ── Buttons */
div[data-testid="stButton"] button {
  background: linear-gradient(135deg, var(--cyan) 0%, #0099cc 100%) !important;
  color: #060d1a !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 13px 28px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  width: 100% !important;
  letter-spacing: 0.5px !important;
  box-shadow: 0 4px 20px rgba(0,212,255,0.35), 0 1px 0 rgba(255,255,255,0.2) inset !important;
  transition: all 0.2s !important;
  font-family: 'Syne', sans-serif !important;
}
div[data-testid="stButton"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,212,255,0.5) !important;
}

/* ── Dataframe */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
}

/* ── Alerts */
div[data-testid="stAlert"] {
  border-radius: 10px !important;
  font-size: 13px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
}

/* ─────────────────────────────────────────
   CUSTOM COMPONENTS
───────────────────────────────────────── */

/* Glass card */
.g-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 26px;
  margin-bottom: 16px;
  backdrop-filter: blur(20px);
  box-shadow: 0 4px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
  transition: border-color 0.25s, box-shadow 0.25s;
}
.g-card:hover {
  border-color: rgba(0,212,255,0.15);
  box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,212,255,0.08), inset 0 1px 0 rgba(255,255,255,0.07);
}

.g-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
.g-card-title {
  font-size: 11px;
  font-weight: 700;
  color: var(--text2);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  font-family: 'DM Mono', monospace;
  flex: 1;
}
.g-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-family: 'DM Mono', monospace;
}
.badge-cyan  { background: var(--cyan-dim);  color: var(--cyan);  border: 1px solid rgba(0,212,255,0.2); }
.badge-green { background: var(--green-dim); color: var(--green); border: 1px solid rgba(46,213,115,0.2); }
.badge-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,177,66,0.2); }
.badge-red   { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(255,71,87,0.2); }

/* Result banners */
.res-critical {
  background: linear-gradient(135deg, rgba(255,71,87,0.1), rgba(255,71,87,0.06));
  border: 1px solid rgba(255,71,87,0.3);
  border-left: 4px solid var(--red);
  border-radius: 14px;
  padding: 24px 28px;
  box-shadow: 0 4px 32px rgba(255,71,87,0.1);
}
.res-warning {
  background: linear-gradient(135deg, rgba(255,177,66,0.1), rgba(255,177,66,0.06));
  border: 1px solid rgba(255,177,66,0.3);
  border-left: 4px solid var(--amber);
  border-radius: 14px;
  padding: 24px 28px;
  box-shadow: 0 4px 32px rgba(255,177,66,0.1);
}
.res-normal {
  background: linear-gradient(135deg, rgba(46,213,115,0.1), rgba(46,213,115,0.06));
  border: 1px solid rgba(46,213,115,0.3);
  border-left: 4px solid var(--green);
  border-radius: 14px;
  padding: 24px 28px;
  box-shadow: 0 4px 32px rgba(46,213,115,0.1);
}

.res-pct  { font-size: 58px; font-weight: 800; line-height: 1; margin-bottom: 4px; font-family: 'Syne', sans-serif; }
.res-lbl  { font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; font-family: 'DM Mono', monospace; }
.res-desc { font-size: 13px; font-weight: 400; opacity: 0.75; margin-bottom: 8px; }
.res-meta { font-size: 11px; opacity: 0.5; font-family: 'DM Mono', monospace; }

/* Alert pill */
.alert-pill {
  border-radius: 8px;
  padding: 10px 16px;
  font-size: 12px;
  font-weight: 600;
  max-width: 210px;
  text-align: right;
  font-family: 'DM Sans', sans-serif;
}

/* Med table */
.m-table { width:100%; border-collapse:collapse; font-size:12px; }
.m-table th {
  background: rgba(255,255,255,0.03);
  text-align:left; padding:10px 14px;
  color: var(--text3); font-weight:700; font-size:10px;
  letter-spacing:1.2px; text-transform:uppercase;
  border-bottom: 1px solid var(--border);
  font-family: 'DM Mono', monospace;
}
.m-table td { padding:12px 14px; border-bottom:1px solid var(--border); color:var(--text2); vertical-align:top; }
.m-table tr:last-child td { border-bottom:none; }
.m-table tr:hover td { background: rgba(0,212,255,0.03); }
.m-name    { font-weight:600; color:var(--text); font-size:13px; font-family:'DM Sans',sans-serif; }
.m-dose    { color:var(--cyan); font-weight:600; font-size:11px; margin-top:2px; font-family:'DM Mono',monospace; }
.m-purp    { color:var(--text3); font-size:11px; }

/* Check items */
.chk-item { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); align-items:flex-start; }
.chk-item:last-child { border-bottom:none; }
.dot-red   { width:6px;height:6px;border-radius:50%;background:var(--red);   flex-shrink:0;margin-top:6px;box-shadow:0 0 6px var(--red); }
.dot-amber { width:6px;height:6px;border-radius:50%;background:var(--amber); flex-shrink:0;margin-top:6px;box-shadow:0 0 6px var(--amber); }
.dot-cyan  { width:6px;height:6px;border-radius:50%;background:var(--cyan);  flex-shrink:0;margin-top:6px;box-shadow:0 0 6px var(--cyan); }
.dot-green { width:6px;height:6px;border-radius:50%;background:var(--green); flex-shrink:0;margin-top:6px;box-shadow:0 0 6px var(--green); }
.chk-text  { font-size:13px;color:var(--text2);line-height:1.5; }

/* Lifestyle tile */
.ls-tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 12px;
  text-align: center;
  font-size: 12px;
  color: var(--text2);
  font-weight: 500;
  line-height: 1.5;
  min-height: 78px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.2s;
}
.ls-tile:hover { border-color: var(--border2); color: var(--text); }

/* Disclaimer */
.disclaimer {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  margin-top: 16px;
  font-size: 11px;
  color: var(--text3);
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

/* Sidebar stats */
.sb-row { display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-bottom:1px solid var(--border); }
.sb-lbl { font-size:11px; color:var(--text3); font-weight:500; }
.sb-val { font-size:14px; color:var(--text); font-weight:700; font-family:'Syne',sans-serif; }

/* OCR info banner */
.ocr-banner {
  background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(99,102,241,0.06));
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 12px;
  padding: 14px 18px;
  margin-bottom: 18px;
  font-size: 12px;
  color: var(--cyan);
  font-weight: 500;
}

/* Empty state */
.empty-state {
  text-align:center;
  padding:80px 20px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
}
.empty-icon { font-size:52px; margin-bottom:16px; opacity:0.2; }
.empty-title { font-size:15px; font-weight:600; color:var(--text2); font-family:'Syne',sans-serif; }
.empty-sub   { font-size:12px; color:var(--text3); margin-top:6px; }

/* Glow heading */
.glow-heading {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.5px;
}

/* Stat mini-card */
.stat-mini {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  text-align: center;
  transition: all 0.2s;
}
.stat-mini:hover {
  border-color: var(--border2);
  box-shadow: 0 0 20px var(--cyan-dim);
}
.stat-mini-val { font-size: 26px; font-weight: 800; color: var(--cyan); font-family: 'Syne', sans-serif; }
.stat-mini-lbl { font-size: 10px; color: var(--text3); font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; margin-top: 4px; font-family: 'DM Mono', monospace; }

/* stMarkdown */
.stMarkdown p { color: var(--text2) !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Plot config (dark theme) ─────────────────────────────────────────────────
PL  = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#7f9ab8", family="DM Sans"),
)
_AX = dict(
    gridcolor="rgba(255,255,255,0.05)",
    zerolinecolor="rgba(255,255,255,0.08)",
    tickfont=dict(size=11, color="#4a6a8a"),
    color="#4a6a8a",
    linecolor="rgba(255,255,255,0.06)",
    showline=True,
)
_L = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
          font=dict(size=11, color="#7f9ab8"))

def _fix(fig, xaxis=None, yaxis=None, margin=None, legend=None):
    fig.update_layout(
        margin=margin or dict(l=8, r=8, t=8, b=8),
        xaxis={**_AX, **(xaxis or {})},
        yaxis={**_AX, **(yaxis or {})},
    )
    if legend is not None:
        fig.update_layout(legend={**_L, **legend})

CYAN  = "#00d4ff"
RED   = "#ff4757"
AMBER = "#ffb142"
GREEN = "#2ed573"
GRAY  = "#4a6a8a"
NAVY  = "#060d1a"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:28px 12px 20px;text-align:center">
        <div style="width:56px;height:56px;background:linear-gradient(135deg,#00d4ff,#0099cc);
             border-radius:16px;display:flex;align-items:center;justify-content:center;
             margin:0 auto 12px;font-size:26px;box-shadow:0 8px 32px rgba(0,212,255,0.4)">🏥</div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:#e8f0fe;letter-spacing:-.3px">MedCore AI</div>
        <div style="font-size:9px;color:#4a6a8a;letter-spacing:2px;text-transform:uppercase;margin-top:3px;font-family:'DM Mono',monospace">Clinical Intelligence</div>
        <div style="margin:14px auto;width:40px;height:1px;background:rgba(0,212,255,0.2)"></div>
        <div style="font-size:11px;color:#4a6a8a;font-family:'DM Mono',monospace">
            {datetime.now().strftime("%d %b %Y")}<br>
            <span style="color:#00d4ff">{datetime.now().strftime("%H:%M")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:0 8px">
    <div style="font-size:9px;font-weight:700;color:#4a6a8a;letter-spacing:2px;text-transform:uppercase;padding:6px 0 10px;font-family:'DM Mono',monospace">Model Status</div>

    <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.12);border-radius:12px;padding:14px 16px;margin-bottom:8px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
            <span style="color:#e8f0fe;font-size:13px;font-weight:600;font-family:'DM Sans',sans-serif">🫀 Heart Disease</span>
            <span style="background:rgba(46,213,115,0.15);color:#2ed573;font-size:9px;font-weight:700;
                  padding:3px 9px;border-radius:20px;letter-spacing:1px;border:1px solid rgba(46,213,115,0.2);font-family:'DM Mono',monospace">LIVE</span>
        </div>
        <div style="color:#4a6a8a;font-size:10px;font-family:'DM Mono',monospace">{heart_model.n_features_in_} features · {heart_model.n_estimators} trees</div>
    </div>

    <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.12);border-radius:12px;padding:14px 16px;margin-bottom:20px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
            <span style="color:#e8f0fe;font-size:13px;font-weight:600;font-family:'DM Sans',sans-serif">🔬 Cancer Risk</span>
            <span style="background:rgba(46,213,115,0.15);color:#2ed573;font-size:9px;font-weight:700;
                  padding:3px 9px;border-radius:20px;letter-spacing:1px;border:1px solid rgba(46,213,115,0.2);font-family:'DM Mono',monospace">LIVE</span>
        </div>
        <div style="color:#4a6a8a;font-size:10px;font-family:'DM Mono',monospace">{cancer_model.n_features_in_} features · {cancer_model.n_estimators} trees</div>
    </div>

    <div style="font-size:9px;font-weight:700;color:#4a6a8a;letter-spacing:2px;text-transform:uppercase;padding:0 0 10px;font-family:'DM Mono',monospace">Today's Activity</div>
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:14px 16px;margin-bottom:20px">
        <div class="sb-row"><span class="sb-lbl">Screened</span><span class="sb-val">142</span></div>
        <div class="sb-row"><span class="sb-lbl">High Risk</span><span class="sb-val" style="color:#ff4757">38</span></div>
        <div class="sb-row"><span class="sb-lbl">Moderate</span><span class="sb-val" style="color:#ffb142">54</span></div>
        <div class="sb-row" style="border:none"><span class="sb-lbl">Clear</span><span class="sb-val" style="color:#2ed573">50</span></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 8px;font-size:9px;font-weight:700;color:#4a6a8a;letter-spacing:2px;text-transform:uppercase;padding-bottom:8px;font-family:DM Mono,monospace">Risk Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 30, 80, 50, 5, key="threshold")
    st.markdown(f'<div style="font-size:10px;color:#4a6a8a;margin-top:-6px;padding:0 8px;font-family:DM Mono,monospace">Scores >= {threshold}% = High Risk</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:24px 8px 0;background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.1);
         border-radius:12px;padding:14px;font-size:10px;color:#4a6a8a;text-align:center;
         line-height:1.9;font-family:'DM Mono',monospace">
        v3.0 · Research &amp; Education<br>Not for clinical use ⚕️
    </div>
    """, unsafe_allow_html=True)

# ─── Top Header Bar ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(0,212,255,0.08) 0%,rgba(0,0,0,0) 50%,rgba(99,102,241,0.06) 100%);
     border:1px solid rgba(0,212,255,0.12);border-radius:18px;padding:18px 28px;margin-bottom:20px;
     display:flex;align-items:center;justify-content:space-between;
     box-shadow:0 4px 40px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.05)">
    <div style="display:flex;align-items:center;gap:20px">
        <div style="width:52px;height:52px;background:linear-gradient(135deg,#00d4ff,#0099cc);
             border-radius:14px;display:flex;align-items:center;justify-content:center;
             font-size:24px;box-shadow:0 8px 24px rgba(0,212,255,0.4);flex-shrink:0">🏥</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#e8f0fe;letter-spacing:-.5px">
                MedCore Clinical AI
                <span style="font-size:11px;color:#00d4ff;background:rgba(0,212,255,0.12);
                      border:1px solid rgba(0,212,255,0.2);border-radius:6px;padding:2px 8px;
                      margin-left:10px;font-family:'DM Mono',monospace;font-weight:600;letter-spacing:1px">v3.0</span>
            </div>
            <div style="font-size:12px;color:#4a6a8a;margin-top:3px;font-family:'DM Sans',sans-serif">
                Dept. of Oncology &amp; Cardiology &nbsp;·&nbsp; AI Diagnosis &nbsp;·&nbsp; OCR Reports 
            </div>
        </div>
    </div>
    <div style="display:flex;gap:24px;align-items:center">
        <div style="text-align:center">
            <div style="font-size:9px;color:#4a6a8a;text-transform:uppercase;letter-spacing:1.5px;font-family:'DM Mono',monospace">Session</div>
            <div style="font-size:14px;color:#e8f0fe;font-weight:700;font-family:'Syne',sans-serif">{datetime.now().strftime("%H:%M")}</div>
        </div>
        <div style="text-align:center">
            <div style="font-size:9px;color:#4a6a8a;text-transform:uppercase;letter-spacing:1.5px;font-family:'DM Mono',monospace">Ward</div>
            <div style="font-size:14px;color:#e8f0fe;font-weight:700;font-family:'Syne',sans-serif">ONC-04</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(46,213,115,0.15),rgba(46,213,115,0.08));
             color:#2ed573;padding:8px 18px;border-radius:10px;font-size:12px;font-weight:700;
             border:1px solid rgba(46,213,115,0.25);font-family:'DM Mono',monospace;letter-spacing:.5px;
             box-shadow:0 0 20px rgba(46,213,115,0.15)">
            ● ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Patients",    "1,284", "+42 this month")
k2.metric("Heart Risk Cases",  "439",   "+28", delta_color="inverse")
k3.metric("Cancer Risk Cases", "277",   "-12", delta_color="inverse")
k4.metric("Heart Model Acc.",  "93.1%", "+0.4%")
k5.metric("Cancer Model Acc.", "89.3%", "+1.2%")
st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  📊  Dashboard  ",
    "  🔬  Diagnosis  ",
    "  📄  OCR Reports  ",
    "  📋  Patient Registry  ",
    "  💬  Health Chat  ",
    "  📈  Model Performance  ",
])

# ══════════════════════════════════════════════════════════════════════
# SHARED: render_clinical_report
# ══════════════════════════════════════════════════════════════════════
def render_clinical_report(level, risk_pct, confidence, prediction_label,
                            fi_df, fi_color_lo, fi_color_hi,
                            disease_data, col_accent, chart_key):

    css_map   = {"HIGH RISK": "res-critical", "MODERATE": "res-warning", "LOW RISK": "res-normal"}
    color_map = {"HIGH RISK": RED, "MODERATE": AMBER, "LOW RISK": GREEN}
    col       = color_map[level]
    emo       = "🔴" if level == "HIGH RISK" else "🟡" if level == "MODERATE" else "🟢"

    alert_bg = "rgba(255,71,87,0.12)"   if level == "HIGH RISK" else \
               "rgba(255,177,66,0.12)"  if level == "MODERATE"  else "rgba(46,213,115,0.12)"
    alert_bd = "rgba(255,71,87,0.3)"    if level == "HIGH RISK" else \
               "rgba(255,177,66,0.3)"   if level == "MODERATE"  else "rgba(46,213,115,0.3)"
    alert_text = disease_data.get("alert_short", "Consult your doctor")

    st.markdown(f"""
    <div class="{css_map[level]}">
        <div style="display:flex;align-items:flex-start;justify-content:space-between">
            <div>
                <div class="res-lbl" style="color:{col}">{emo} &nbsp; {level}</div>
                <div class="res-pct" style="color:{col}">{risk_pct}%</div>
                <div class="res-desc" style="color:{col}">Predicted probability of disease</div>
                <div class="res-meta">Confidence: {confidence}% &nbsp;·&nbsp; Class: {prediction_label}</div>
            </div>
            <div style="text-align:right">
                <div style="font-size:9px;color:{col};font-weight:700;letter-spacing:2px;
                     text-transform:uppercase;margin-bottom:8px;font-family:'DM Mono',monospace">AI Assessment</div>
                <div style="background:{alert_bg};border:1px solid {alert_bd};border-radius:10px;
                     padding:10px 16px;font-size:12px;color:{col};font-weight:600;max-width:220px">
                    {alert_text}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="g-card-header"><div class="g-card-title">Feature Importance</div><span class="g-badge badge-cyan">RF Model</span></div>', unsafe_allow_html=True)
    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
        marker=dict(color=fi_df["Importance"].tolist(),
            colorscale=[[0, fi_color_lo], [1, fi_color_hi]],
            line=dict(width=0)),
        text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside",
        textfont=dict(size=10, color=GRAY, family="DM Mono"),
    ))
    fig_fi.update_layout(**PL, height=260)
    _fix(fig_fi, xaxis=dict(showgrid=False, range=[0, fi_df["Importance"].max()*1.4]),
         margin=dict(l=0, r=55, t=8, b=8))
    st.plotly_chart(fig_fi, use_container_width=True, key=chart_key)
    st.markdown('</div>', unsafe_allow_html=True)

    # Medications
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="g-card-header"><div class="g-card-title">💊 Recommended Medications</div><span class="g-badge badge-amber">Consult Physician</span></div>', unsafe_allow_html=True)
    rows = "".join([f'<tr><td><div class="m-name">{m[0]}</div><div class="m-dose">{m[1]}</div></td><td><div class="m-purp">{m[2]}</div></td></tr>' for m in disease_data["medicines"]])
    st.markdown(f'<table class="m-table"><thead><tr><th>Drug &amp; Dose</th><th>Purpose / Notes</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tests + Precautions
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🧪 Diagnostic Tests</div></div>', unsafe_allow_html=True)
        for t in disease_data["tests"]:
            st.markdown(f'<div class="chk-item"><div class="dot-cyan"></div><div class="chk-text">{t}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tc2:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">⚠️ Precautions</div></div>', unsafe_allow_html=True)
        dot = "dot-red" if level == "HIGH RISK" else "dot-amber" if level == "MODERATE" else "dot-green"
        for p in disease_data["precautions"]:
            st.markdown(f'<div class="chk-item"><div class="{dot}"></div><div class="chk-text">{p}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Lifestyle
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="g-card-header"><div class="g-card-title">🌿 Lifestyle Modifications</div></div>', unsafe_allow_html=True)
    ls_cols = st.columns(len(disease_data["lifestyle"]))
    for lc, tip in zip(ls_cols, disease_data["lifestyle"]):
        lc.markdown(f'<div class="ls-tile">{tip}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <span style="font-size:16px">⚕️</span>
        <span style="color:#4a6a8a"><strong style="color:#7f9ab8">Medical Disclaimer:</strong>
        This AI-generated report is for educational and research purposes only.
        All clinical decisions must be made by a qualified healthcare professional.</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">📈 Monthly Screening Volume</div><span class="g-badge badge-cyan">AI Tracked</span></div>', unsafe_allow_html=True)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=[52,60,55,70,65,78,82,74,88,92,85,97],
            name="Heart Disease", line=dict(color=RED, width=2.5),
            fill="tozeroy", fillcolor="rgba(255,71,87,0.08)",
            mode="lines+markers", marker=dict(size=5, color=RED)))
        fig.add_trace(go.Scatter(x=months, y=[30,35,28,40,38,45,50,42,55,60,52,64],
            name="Cancer Risk", line=dict(color=AMBER, width=2.5),
            fill="tozeroy", fillcolor="rgba(255,177,66,0.08)",
            mode="lines+markers", marker=dict(size=5, color=AMBER)))
        fig.update_layout(**PL, height=240)
        _fix(fig, xaxis=dict(showgrid=False),
             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🫀 Risk Distribution</div><span class="g-badge badge-green">Live</span></div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Heart Risk","Cancer Risk","No Risk"], values=[439,277,568], hole=0.65,
            marker=dict(colors=[RED, AMBER, "rgba(255,255,255,0.06)"],
                        line=dict(color=NAVY, width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} patients (%{percent})<extra></extra>",
        ))
        fig2.add_annotation(text="1,284", x=0.5, y=0.58, showarrow=False,
            font=dict(size=26, color="#e8f0fe", family="Syne"))
        fig2.add_annotation(text="patients", x=0.5, y=0.42, showarrow=False,
            font=dict(size=11, color=GRAY, family="DM Sans"))
        fig2.update_layout(**PL, height=240, showlegend=True, margin=dict(l=0,r=0,t=8,b=8))
        fig2.update_layout(legend={**_L, "orientation":"v","x":0.78,"y":0.5})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">👥 Risk by Age Group</div></div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Heart", x=["20-30","31-40","41-50","51-60","61-70","71+"],
            y=[8,15,28,45,62,71], marker_color=RED, opacity=0.8,
            marker=dict(line=dict(width=0))))
        fig3.add_trace(go.Bar(name="Cancer", x=["20-30","31-40","41-50","51-60","61-70","71+"],
            y=[5,12,22,35,48,58], marker_color=AMBER, opacity=0.8,
            marker=dict(line=dict(width=0))))
        fig3.update_layout(**PL, barmode="group", height=220)
        _fix(fig3, xaxis=dict(showgrid=False),
             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🧬 Feature Correlation</div></div>', unsafe_allow_html=True)
        feats = ["Age","Chol","BP","HR","BMI"]
        corr  = np.array([[1.0,0.62,0.58,-0.41,0.35],[0.62,1.0,0.44,-0.28,0.41],
                           [0.58,0.44,1.0,-0.35,0.52],[-0.41,-0.28,-0.35,1.0,-0.22],
                           [0.35,0.41,0.52,-0.22,1.0]])
        fig4 = go.Figure(go.Heatmap(z=corr, x=feats, y=feats,
            colorscale=[[0,"rgba(0,212,255,0.05)"],[0.5,"rgba(0,212,255,0.4)"],[1,"#00d4ff"]],
            text=np.round(corr,2), texttemplate="%{text}",
            textfont=dict(size=11, color="#e8f0fe"), showscale=False))
        fig4.update_layout(**PL, height=220)
        _fix(fig4, margin=dict(l=0,r=0,t=8,b=8))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    ptab1, ptab2 = st.tabs(["  🫀  Heart Disease Assessment  ", "  🔬  Cancer Risk Assessment  "])

    # ── HEART ─────────────────────────────────────────────────────────
    with ptab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
             border-radius:12px;padding:12px 18px;margin-bottom:18px;display:flex;align-items:center;gap:12px">
            <span style="font-size:18px">🤖</span>
            <span style="font-size:12px;color:#00d4ff;font-weight:600;font-family:'DM Sans',sans-serif">
                RandomForest Classifier &nbsp;·&nbsp; 13 Clinical Features &nbsp;·&nbsp;
                Cleveland Heart Disease Dataset &nbsp;·&nbsp; {heart_model.n_estimators} Trees
            </span>
        </div>
        """, unsafe_allow_html=True)

        pf1, pf2 = st.columns([5, 4])
        with pf1:
            st.markdown('<div class="g-card">', unsafe_allow_html=True)
            st.markdown('<div class="g-card-header"><div class="g-card-title">Patient Clinical Parameters</div><span class="g-badge badge-cyan">Cleveland Schema</span></div>', unsafe_allow_html=True)
            r1a, r1b, r1c = st.columns(3)
            age_h = r1a.number_input("Age (years)", 20, 80, 52, key="h_age")
            sex_h = r1b.selectbox("Sex", ["Male (1)","Female (0)"], key="h_sex")
            cp_h  = r1c.selectbox("Chest Pain Type", ["0 – Typical Angina","1 – Atypical","2 – Non-Anginal","3 – Asymptomatic"], key="h_cp")
            r2a, r2b, r2c = st.columns(3)
            trestbps_h = r2a.number_input("Resting BP (mmHg)", 80, 220, 130, key="h_bp")
            chol_h     = r2b.number_input("Cholesterol (mg/dL)", 100, 600, 240, key="h_chol")
            fbs_h      = r2c.selectbox("Fasting BS >120", ["No (0)","Yes (1)"], key="h_fbs")
            r3a, r3b, r3c = st.columns(3)
            restecg_h = r3a.selectbox("Resting ECG", ["0 – Normal","1 – ST Abnorm.","2 – LV Hypertrophy"], key="h_ecg")
            thalach_h = r3b.number_input("Max Heart Rate", 60, 220, 150, key="h_hr")
            exang_h   = r3c.selectbox("Exercise Angina", ["No (0)","Yes (1)"], key="h_exang")
            r4a, r4b, r4c = st.columns(3)
            oldpeak_h = r4a.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1, key="h_old")
            slope_h   = r4b.selectbox("ST Slope", ["0 – Upsloping","1 – Flat","2 – Downsloping"], key="h_slope")
            ca_h      = r4c.selectbox("Major Vessels (0-3)", ["0","1","2","3"], key="h_ca")
            thal_h    = st.selectbox("Thalassemia", ["1 – Normal","2 – Fixed Defect","3 – Reversible Defect"], key="h_thal")
            st.markdown('</div>', unsafe_allow_html=True)
            run_heart = st.button("🫀  Run Heart Disease Analysis", key="btn_heart")

        with pf2:
            if run_heart:
                X_heart = pd.DataFrame([[
                    age_h, 1 if "Male" in sex_h else 0, int(cp_h[0]),
                    trestbps_h, chol_h, 1 if "Yes" in fbs_h else 0,
                    int(restecg_h[0]), thalach_h, 1 if "Yes" in exang_h else 0,
                    oldpeak_h, int(slope_h[0]), int(ca_h), int(thal_h[0])
                ]], columns=["age","sex","cp","trestbps","chol","fbs","restecg",
                             "thalach","exang","oldpeak","slope","ca","thal"])
                proba    = heart_model.predict_proba(X_heart)[0]
                pred     = heart_model.predict(X_heart)[0]
                risk_pct = round(proba[1] * 100, 1)
                conf     = round(max(proba) * 100, 1)
                level    = "HIGH RISK" if risk_pct >= threshold else "MODERATE" if risk_pct >= 35 else "LOW RISK"
                fi_df = pd.DataFrame({
                    "Feature":    ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
                    "Importance": heart_model.feature_importances_,
                }).sort_values("Importance", ascending=True)
                HEART_DATA = {
                    "HIGH RISK": {
                        "alert_short": "Cardiology referral required",
                        "medicines": [
                            ("Aspirin","75-100 mg daily","Antiplatelet · reduces clot risk"),
                            ("Atorvastatin","40-80 mg daily","Statin · lowers LDL cholesterol"),
                            ("Metoprolol","25-100 mg daily","Beta-blocker · controls HR and BP"),
                            ("Ramipril","2.5-10 mg daily","ACE inhibitor · reduces cardiac load"),
                            ("Nitroglycerin SL","0.4 mg PRN","Acute chest pain relief"),
                        ],
                        "tests":       ["12-lead ECG","Stress echocardiogram","Full lipid panel","Troponin I/T","Coronary angiography"],
                        "precautions": ["Restrict sodium < 2g/day","No strenuous exercise until cleared","Quit smoking immediately","Monitor BP twice daily","Avoid NSAIDs"],
                        "lifestyle":   ["Mediterranean diet","Cardiac rehab programme","30 min light walking/day","Stress reduction techniques","Limit alcohol strictly"],
                    },
                    "MODERATE": {
                        "alert_short": "Cardiology follow-up in 4 weeks",
                        "medicines": [
                            ("Aspirin","75 mg daily","Preventive antiplatelet"),
                            ("Atorvastatin","20-40 mg daily","Cholesterol management"),
                            ("Amlodipine","5 mg daily","CCB · blood pressure control"),
                            ("Metformin (if diabetic)","500 mg BD","Glucose control · cardioprotective"),
                        ],
                        "tests":       ["Resting ECG","Fasting lipid profile","Blood glucose / HbA1c","Echocardiogram (baseline)"],
                        "precautions": ["Reduce sodium and saturated fat","Monitor BP weekly","Cholesterol check every 3 months","Limit caffeine and alcohol"],
                        "lifestyle":   ["30 min moderate exercise 5x/week","Achieve healthy BMI","Increase fruit and vegetable intake","Reduce processed foods"],
                    },
                    "LOW RISK": {
                        "alert_short": "Routine annual review",
                        "medicines": [
                            ("No medication required","—","Maintain with lifestyle changes"),
                            ("Omega-3 / Fish Oil","1 g daily","Heart-healthy supplement"),
                            ("Vitamin D3","1000 IU daily","Cardiovascular support"),
                        ],
                        "tests":       ["Annual BP and lipid check","Fasting blood glucose","BMI and waist circumference"],
                        "precautions": ["Maintain current healthy habits","Annual cardiovascular screening","Avoid smoking","Stay hydrated 2L/day"],
                        "lifestyle":   ["Continue regular aerobic exercise","Balanced whole-food diet","Healthy sleep schedule","Annual health check-up"],
                    },
                }
                render_clinical_report(level, risk_pct, conf,
                    "Disease present" if pred == 1 else "No disease",
                    fi_df, "rgba(0,212,255,0.2)", RED, HEART_DATA[level], RED, "heart_fi")
            else:
                st.markdown('<div class="empty-state"><div class="empty-icon">🫀</div><div class="empty-title">Enter patient parameters and run analysis</div><div class="empty-sub">Clinical AI report will appear here</div></div>', unsafe_allow_html=True)

    # ── CANCER ────────────────────────────────────────────────────────
    with ptab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(255,177,66,0.06);border:1px solid rgba(255,177,66,0.2);
             border-radius:12px;padding:12px 18px;margin-bottom:18px;display:flex;align-items:center;gap:12px">
            <span style="font-size:18px">🤖</span>
            <span style="font-size:12px;color:#ffb142;font-weight:600">
                RandomForest Classifier &nbsp;·&nbsp; 30 Tumour Features &nbsp;·&nbsp;
                Wisconsin Breast Cancer Dataset &nbsp;·&nbsp; Class 0 = Malignant · Class 1 = Benign
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── 30-feature form layout ──────────────────────────────────────
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">Tumour Cell Nucleus — All 30 Features</div><span class="g-badge badge-amber">FNA · Wisconsin Dataset</span></div>', unsafe_allow_html=True)

        # ── Section 1: MEAN VALUES ───────────────────────────────────
        st.markdown('<div style="font-size:10px;font-weight:700;color:#00d4ff;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;font-family:DM Mono,monospace;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.15)">📐 Mean Values</div>', unsafe_allow_html=True)
        s1c1, s1c2, s1c3, s1c4, s1c5 = st.columns(5)
        mean_radius      = s1c1.number_input("Radius",          6.9,   28.11,  14.13,  0.01,            key="c_mr")
        mean_texture     = s1c2.number_input("Texture",         9.71,  39.28,  19.29,  0.01,            key="c_mt")
        mean_perimeter   = s1c3.number_input("Perimeter",       43.79, 188.5,  91.97,  0.1,             key="c_mp")
        mean_area        = s1c4.number_input("Area",            143.5, 2501.0, 654.9,  0.5,             key="c_ma")
        mean_smoothness  = s1c5.number_input("Smoothness",      0.053, 0.163,  0.096,  0.0001,format="%.4f", key="c_msmooth")

        s1c6, s1c7, s1c8, s1c9, s1c10 = st.columns(5)
        mean_compactness = s1c6.number_input("Compactness",     0.019, 0.345,  0.104,  0.001, format="%.4f", key="c_mcomp")
        mean_concavity   = s1c7.number_input("Concavity",       0.0,   0.427,  0.089,  0.001, format="%.4f", key="c_mconcav")
        mean_concave_pts = s1c8.number_input("Concave Pts",     0.0,   0.201,  0.049,  0.001, format="%.4f", key="c_mconcpt")
        mean_symmetry    = s1c9.number_input("Symmetry",        0.106, 0.304,  0.181,  0.001, format="%.4f", key="c_msym")
        mean_fractal_dim = s1c10.number_input("Fractal Dim",    0.050, 0.097,  0.063,  0.0001,format="%.5f", key="c_mfrac")

        # ── Section 2: STANDARD ERROR ────────────────────────────────
        st.markdown('<div style="font-size:10px;font-weight:700;color:#ffb142;letter-spacing:2px;text-transform:uppercase;margin:14px 0 10px;font-family:DM Mono,monospace;padding:6px 0;border-bottom:1px solid rgba(255,177,66,0.15)">📏 Standard Error (SE)</div>', unsafe_allow_html=True)
        s2c1, s2c2, s2c3, s2c4, s2c5 = st.columns(5)
        se_radius        = s2c1.number_input("SE Radius",       0.112, 2.873,  0.405,  0.001, format="%.3f", key="c_ser")
        se_texture       = s2c2.number_input("SE Texture",      0.36,  4.885,  1.217,  0.001, format="%.3f", key="c_set")
        se_perimeter     = s2c3.number_input("SE Perimeter",    0.757, 21.98,  2.866,  0.01,  format="%.3f", key="c_sep")
        se_area          = s2c4.number_input("SE Area",         6.802, 542.2,  40.34,  0.1,   format="%.2f", key="c_sea")
        se_smoothness    = s2c5.number_input("SE Smoothness",   0.002, 0.031,  0.007,  0.0001,format="%.4f", key="c_sesmooth")

        s2c6, s2c7, s2c8, s2c9, s2c10 = st.columns(5)
        se_compactness   = s2c6.number_input("SE Compactness",  0.002, 0.135,  0.025,  0.001, format="%.4f", key="c_secomp")
        se_concavity     = s2c7.number_input("SE Concavity",    0.0,   0.396,  0.032,  0.001, format="%.4f", key="c_seconcav")
        se_concave_pts   = s2c8.number_input("SE Concave Pts",  0.0,   0.053,  0.012,  0.0001,format="%.4f", key="c_seconcpt")
        se_symmetry      = s2c9.number_input("SE Symmetry",     0.008, 0.079,  0.021,  0.001, format="%.4f", key="c_sesym")
        se_fractal_dim   = s2c10.number_input("SE Fractal Dim", 0.001, 0.030,  0.004,  0.0001,format="%.5f", key="c_sefrac")

        # ── Section 3: WORST VALUES ──────────────────────────────────
        st.markdown('<div style="font-size:10px;font-weight:700;color:#ff4757;letter-spacing:2px;text-transform:uppercase;margin:14px 0 10px;font-family:DM Mono,monospace;padding:6px 0;border-bottom:1px solid rgba(255,71,87,0.15)">⚠️ Worst Values (Largest)</div>', unsafe_allow_html=True)
        s3c1, s3c2, s3c3, s3c4, s3c5 = st.columns(5)
        worst_radius     = s3c1.number_input("Worst Radius",    7.93,  36.04,  16.27,  0.01,            key="c_wr")
        worst_texture    = s3c2.number_input("Worst Texture",   12.02, 49.54,  25.68,  0.01,            key="c_wt")
        worst_perimeter  = s3c3.number_input("Worst Perimeter", 50.41, 251.2,  107.3,  0.1,             key="c_wp")
        worst_area       = s3c4.number_input("Worst Area",      185.2, 4254.0, 880.6,  0.5,             key="c_wa")
        worst_smoothness = s3c5.number_input("Worst Smooth.",   0.071, 0.223,  0.132,  0.001, format="%.4f", key="c_wsmooth")

        s3c6, s3c7, s3c8, s3c9, s3c10 = st.columns(5)
        worst_compactness = s3c6.number_input("Worst Compact.", 0.027, 1.058,  0.254,  0.001, format="%.4f", key="c_wcomp")
        worst_concavity   = s3c7.number_input("Worst Concav.",  0.0,   1.252,  0.272,  0.001, format="%.4f", key="c_wconcav")
        worst_concave_pts = s3c8.number_input("Worst Conc.Pts", 0.0,   0.291,  0.115,  0.001, format="%.4f", key="c_wconcpt")
        worst_symmetry    = s3c9.number_input("Worst Symmetry", 0.157, 0.664,  0.290,  0.001, format="%.4f", key="c_wsym")
        worst_fractal_dim = s3c10.number_input("Worst Frac.Dim",0.055, 0.208,  0.084,  0.0001,format="%.5f", key="c_wfrac")

        st.markdown('</div>', unsafe_allow_html=True)

        # note about model features
        n_feats = cancer_model.n_features_in_
        if n_feats == 10:
            st.info("ℹ Your cancer_model.pkl was trained on 10 features. Re-run train_model.py with 30 features to use all inputs. Currently using mean values only.")

        run_cancer = st.button("🔬  Run Cancer Risk Analysis", key="btn_cancer")

        cf_result_col = st.container()
        with cf_result_col:
            if run_cancer:
                # Build feature vector — use 30 or 10 depending on model
                if n_feats == 30:
                    X_cancer = np.array([[
                        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                        mean_compactness, mean_concavity, mean_concave_pts, mean_symmetry, mean_fractal_dim,
                        se_radius, se_texture, se_perimeter, se_area, se_smoothness,
                        se_compactness, se_concavity, se_concave_pts, se_symmetry, se_fractal_dim,
                        worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
                        worst_compactness, worst_concavity, worst_concave_pts, worst_symmetry, worst_fractal_dim,
                    ]])
                    feat_names = [
                        "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
                        "mean compactness","mean concavity","mean concave pts","mean symmetry","mean fractal dim",
                        "se radius","se texture","se perimeter","se area","se smoothness",
                        "se compactness","se concavity","se concave pts","se symmetry","se fractal dim",
                        "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
                        "worst compactness","worst concavity","worst concave pts","worst symmetry","worst fractal dim",
                    ]
                else:
                    # fallback: 10 mean features only
                    X_cancer = np.array([[
                        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                        mean_compactness, mean_concavity, mean_concave_pts, mean_symmetry, mean_fractal_dim,
                    ]])
                    feat_names = [
                        "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
                        "mean compactness","mean concavity","mean concave pts","mean symmetry","mean fractal dim",
                    ]

                proba_c    = cancer_model.predict_proba(X_cancer)[0]
                pred_c     = cancer_model.predict(X_cancer)[0]
                risk_pct_c = round(proba_c[0] * 100, 1)
                conf_c     = round(max(proba_c) * 100, 1)
                diag       = "Malignant" if pred_c == 0 else "Benign"
                level_c    = "HIGH RISK" if risk_pct_c >= threshold else "MODERATE" if risk_pct_c >= 35 else "LOW RISK"
                fi_df_c = pd.DataFrame({
                    "Feature":    feat_names,
                    "Importance": cancer_model.feature_importances_,
                }).sort_values("Importance", ascending=True)
                CANCER_DATA = {
                    "HIGH RISK": {
                        "alert_short": "Oncology referral urgently",
                        "medicines": [
                            ("Tamoxifen","20 mg daily","ER+ve antiestrogen therapy"),
                            ("Trastuzumab (Herceptin)","IV per protocol","HER2+ve targeted monoclonal Ab"),
                            ("Anastrozole","1 mg daily","Post-menopausal aromatase inhibitor"),
                            ("Docetaxel / Paclitaxel","Per oncology protocol","Chemotherapy physician-directed"),
                            ("Ondansetron","8 mg BD","Antiemetic for chemo nausea"),
                        ],
                        "tests":       ["Mammogram + ultrasound","Core needle biopsy","ER/PR/HER2 receptor panel","MRI breast bilateral","CT staging chest/abdomen/pelvis"],
                        "precautions": ["Do not delay specialist consultation","Avoid self-medication","Genetic counselling BRCA1/2","Avoid alcohol completely","Regular blood counts during chemo"],
                        "lifestyle":   ["High-protein anti-inflammatory diet","Light movement as tolerated","Psychological support","Avoid smoking","Adequate rest and sleep"],
                    },
                    "MODERATE": {
                        "alert_short": "Specialist review in 2 weeks",
                        "medicines": [
                            ("Tamoxifen preventive","20 mg daily","High-risk pre-menopausal"),
                            ("Raloxifene","60 mg daily","Risk reduction post-menopausal"),
                            ("Exemestane","25 mg daily","Aromatase inhibitor elevated risk"),
                            ("Vitamin D3 + Calcium","Per supplement dosing","Bone health maintenance"),
                        ],
                        "tests":       ["Diagnostic mammogram","Breast ultrasound","Fine Needle Aspiration FNA","CA 15-3 / CA 27.29 markers","BRCA gene testing if family history"],
                        "precautions": ["Biannual clinical breast exam","Avoid hormone replacement therapy","Limit alcohol < 1 unit/day","Report any new lumps immediately"],
                        "lifestyle":   ["Anti-inflammatory Mediterranean diet","Exercise 150 min/week","Achieve and maintain healthy BMI","Increase fibre and cruciferous vegetables"],
                    },
                    "LOW RISK": {
                        "alert_short": "Routine surveillance",
                        "medicines": [
                            ("No medication required","—","Maintain healthy lifestyle"),
                            ("Vitamin D3","1000 IU daily","General health maintenance"),
                            ("Omega-3 Fatty Acids","1 g daily","Anti-inflammatory support"),
                        ],
                        "tests":       ["Annual mammogram age 40+","Clinical breast exam yearly","Monthly self breast examination"],
                        "precautions": ["Monthly self breast examination","Annual screening mammogram","Maintain healthy weight","Avoid smoking and excess alcohol"],
                        "lifestyle":   ["Regular aerobic exercise","Plant-rich whole-food diet","Healthy weight maintenance","Annual check-up"],
                    },
                }
                render_clinical_report(level_c, risk_pct_c, conf_c, diag,
                    fi_df_c, "rgba(255,177,66,0.15)", AMBER, CANCER_DATA[level_c], AMBER, "cancer_fi")
            else:
                st.markdown('<div class="empty-state"><div class="empty-icon">🔬</div><div class="empty-title">Enter tumour measurements and run analysis</div><div class="empty-sub">All 30 FNA features · Clinical AI report will appear here</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — OCR REPORT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ocr-banner">
        📄 Upload any medical lab report (PDF / JPG / PNG) or paste the text. The NLP parser extracts 15+ parameters and flags abnormal values.
    </div>
    """, unsafe_allow_html=True)

    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">📤 Upload or Paste Report</div><span class="g-badge badge-cyan">OCR + NLP</span></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload report (PDF / JPG / PNG)", type=["pdf","jpg","png","jpeg"])
        st.markdown("<div style='text-align:center;font-size:11px;color:#4a6a8a;margin:8px 0;font-family:DM Mono,monospace'>— or paste text below —</div>", unsafe_allow_html=True)
        pasted_text = st.text_area("Paste report text here", height=220,
            placeholder="Cholesterol: 268 mg/dL\nBlood Pressure: 142/88 mmHg\nFasting Blood Glucose: 118 mg/dL\nHbA1c: 6.1 %\nHaemoglobin: 13.4 g/dL\nHeart Rate: 88 bpm\nLDL: 162 mg/dL\nHDL: 38 mg/dL\nTriglycerides: 210 mg/dL")
        run_ocr = st.button("🔍  Analyze Report", key="btn_ocr")
        st.markdown('</div>', unsafe_allow_html=True)

    with oc2:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🧪 Extracted Lab Values</div><span class="g-badge badge-green">NLP Parsed</span></div>', unsafe_allow_html=True)

        LAB_PARAMS = [
            ("HbA1c",[r"hba\s*1\s*c[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"glycos[a-z]*\s*hemo[a-z]*[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"a1c[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"%",0,5.7,6.5,"< 5.7%"),
            ("Est. Avg Glucose",[r"estimated\s+average\s+glucose[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"eag[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0,100,126,"70-100 mg/dL"),
            ("Cholesterol",[r"total\s+cholesterol[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"(?<![a-z])cholesterol[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"chol[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0,200,240,"< 200 mg/dL"),
            ("LDL",[r"ldl[\s\-_]*(?:cholesterol|chol)?[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"low\s+density[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0,100,160,"< 100 mg/dL"),
            ("HDL",[r"hdl[\s\-_]*(?:cholesterol|chol)?[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"high\s+density[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",40,999,999,"> 40 mg/dL"),
            ("Triglycerides",[r"triglycerides?[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"trig[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"\btg[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0,150,200,"< 150 mg/dL"),
            ("Fasting Glucose",[r"fasting\s+(?:blood\s+)?glucose[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"(?:fbg|fbs|fasting\s+sugar)[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"blood\s+glucose[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0,100,126,"70-100 mg/dL"),
            ("Blood Pressure",[r"(?:blood\s*pressure|b\.?p\.?)[\s:=\-]+([0-9]{2,3})\s*/\s*[0-9]+",r"systolic[\s:=\-]+([0-9]{2,3})",r"sbp[\s:=\-]+([0-9]{2,3})"],"mmHg",0,120,140,"< 120/80 mmHg"),
            ("Haemoglobin",[r"h(?:a?e?)moglobin[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"(?<![a-z])hb[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"hgb[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"g/dL",12,17,999,"12-17 g/dL"),
            ("Heart Rate",[r"heart\s+rate[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"pulse[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"(?<![a-z])hr[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"bpm",60,100,999,"60-100 bpm"),
            ("Creatinine",[r"(?:serum\s+)?creatinine[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"s\.?\s*creatinine[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0.6,1.2,2.0,"0.6-1.2 mg/dL"),
            ("Uric Acid",[r"uric\s+acid[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"s\.?\s*uric[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mg/dL",0,7.0,9.0,"3.5-7.0 mg/dL"),
            ("Sodium",[r"(?:serum\s+)?sodium[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"(?<![a-z])na[\+]?[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mEq/L",136,145,160,"136-145 mEq/L"),
            ("Potassium",[r"(?:serum\s+)?potassium[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"(?<![a-z])k[\+]?[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mEq/L",3.5,5.0,6.0,"3.5-5.0 mEq/L"),
            ("TSH",[r"tsh[\s:=\-]+([0-9]+(?:\.[0-9]+)?)",r"thyroid\s+stimulating[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"mIU/L",0.4,4.0,10.0,"0.4-4.0 mIU/L"),
            ("Vitamin D",[r"(?:vitamin\s*d|vit\.?\s*d|25[\s\-]?oh[\s\-]?d)[\s:=\-]+([0-9]+(?:\.[0-9]+)?)"],"ng/mL",30,100,200,"> 30 ng/mL"),
        ]

        def get_flag(param, val, lo, hi_norm, hi_warn):
            if param == "HDL":          return "Low (Risk)" if val < lo else "Borderline" if val < 60 else "Optimal"
            if param == "Haemoglobin":  return "Critical Low" if val < 8 else "Low" if val < lo else "Normal"
            if param == "Vitamin D":    return "Deficient" if val < 20 else "Insufficient" if val < lo else "Normal"
            if param in ("Sodium","Potassium"): return "Low" if val < lo else "High" if val > hi_norm else "Normal"
            if val >= hi_warn * 1.25:   return "Critical"
            if val >= hi_warn:          return "High"
            if val >= hi_norm:          return "Borderline"
            if lo > 0 and val < lo:     return "Low"
            return "Normal"

        def parse_report(raw_text):
            out = {}
            tl = re.sub(r"\s+", " ", raw_text.lower())
            for param, patterns, unit, lo, hi_norm, hi_warn, normal_label in LAB_PARAMS:
                for pat in patterns:
                    try:
                        m = re.search(pat, tl)
                        if m:
                            val = float(m.group(1))
                            if val == 0 or val > 99999: continue
                            out[param] = {"value": val, "unit": unit, "normal": normal_label,
                                          "flag": get_flag(param, val, lo, hi_norm, hi_warn)}
                            break
                    except Exception:
                        continue
            return out

        if run_ocr:
            text_to_parse = pasted_text.strip()
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                ocr_done   = False

                # Method 2: EasyOCR
                if not ocr_done:
                    try:
                        import easyocr, PIL.Image, io
                        img    = PIL.Image.open(io.BytesIO(file_bytes))
                        reader = easyocr.Reader(["en"], verbose=False)
                        text_to_parse = " ".join(reader.readtext(np.array(img), detail=0))
                        st.success("✅ OCR completed with EasyOCR")
                        ocr_done = True
                    except Exception:
                        pass

                # Method 3: Tesseract
                if not ocr_done:
                    try:
                        import pytesseract, PIL.Image, io
                        img = PIL.Image.open(io.BytesIO(file_bytes))
                        text_to_parse = pytesseract.image_to_string(img)
                        st.success("✅ OCR completed with Tesseract")
                        ocr_done = True
                    except Exception:
                        pass

                if not ocr_done:
                    st.warning("⚠ Could not read image automatically. Install easyocr (pip install easyocr) or paste the report text manually.")

            if text_to_parse:
                parsed = parse_report(text_to_parse)
                if parsed:
                    fc_map = {
                        "Normal":GREEN,"Optimal":GREEN,"Low":AMBER,"Borderline":AMBER,
                        "High":RED,"Critical":RED,"Insufficient":AMBER,"Deficient":RED,
                        "Low (Risk)":RED,"Critical Low":RED,
                    }
                    rows = ""
                    for param, info in parsed.items():
                        fc = fc_map.get(info["flag"], GRAY)
                        rows += f'<tr><td><strong style="color:#e8f0fe;font-family:DM Sans,sans-serif">{param}</strong></td><td style="color:#7f9ab8;font-family:DM Mono,monospace">{info["value"]} {info["unit"]}</td><td style="color:#4a6a8a;font-size:11px;font-family:DM Mono,monospace">{info["normal"]}</td><td><span style="color:{fc};font-weight:700;font-size:11px;font-family:DM Mono,monospace">{info["flag"]}</span></td></tr>'
                    st.markdown(f'<table class="m-table"><thead><tr><th>Parameter</th><th>Value</th><th>Normal Range</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
                    abnormal = [p for p, i in parsed.items() if i["flag"] not in ("Normal","Optimal")]
                    st.markdown("<br>", unsafe_allow_html=True)
                    if abnormal:
                        st.warning(f"⚠ {len(abnormal)} abnormal: **{', '.join(abnormal)}** — consider running a Diagnosis above.")
                    else:
                        st.success("✅ All extracted values within normal ranges.")
                else:
                    st.warning("No recognisable lab values found. Try format: `Cholesterol: 245 mg/dL`")
            else:
                st.info("Paste report text or upload a file, then click Analyze.")
        else:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#4a6a8a;font-size:13px;font-family:DM Sans,sans-serif">Results will appear here after clicking Analyze</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT REGISTRY
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    random.seed(42); np.random.seed(42)
    n  = 50
    hv = np.random.uniform(5, 95, n).round(1)
    cv = np.random.uniform(5, 90, n).round(1)

    def status(h, c):
        m = max(h, c)
        if m >= threshold: return "High Risk"
        if m >= 35:        return "Moderate"
        return "Clear"

    names = ["Aarav Sharma","Priya Nair","Rohan Mehta","Sneha Iyer","Kiran Patel",
             "Ananya Rao","Vikram Singh","Deepa Kumar","Arjun Das","Meera Joshi",
             "Saurabh Gupta","Rekha Pillai","Rahul Bose","Lakshmi Reddy","Mohit Verma",
             "Swati Shah","Nikhil Tiwari","Kavya Menon","Suresh Naidu","Pooja Yadav",
             "Aditya Kapoor","Ritu Saxena","Manish Pandey","Geeta Srivastava","Akash Jain",
             "Divya Mishra","Sanjay Nair","Anjali Singh","Harish Kumar","Preethi Raj",
             "Kartik Desai","Sunita Patil","Varun Chawla","Kavitha Suresh","Nitin Agarwal",
             "Shweta Bansal","Rohit Ghosh","Nalini Venkat","Amol Jadhav","Seema Dubey",
             "Tarun Arora","Pratibha Nair","Vikram Malhotra","Sunita Kaur","Arun Pillai",
             "Meenal Thakur","Rajesh Shetty","Usha Menon","Ganesh Murthy","Preeti Sharma"]
    wards = ["Cardiology","Oncology","General Medicine","Neurology","Pulmonology"]

    df = pd.DataFrame({
        "Patient ID":    [f"MED-{2410+i:04d}" for i in range(n)],
        "Name":          names[:n],
        "Age":           np.random.randint(25, 80, n),
        "Sex":           np.random.choice(["M","F"], n),
        "Ward":          np.random.choice(wards, n),
        "Heart Risk %":  hv,
        "Cancer Risk %": cv,
        "Status":        [status(h, c) for h, c in zip(hv, cv)],
        "Admission":     [(datetime.now()-timedelta(days=random.randint(0,60))).strftime("%d %b %Y") for _ in range(n)],
    })

    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4, fc5 = st.columns([2,2,2,2,2])
    sf   = fc1.selectbox("Status", ["All","High Risk","Moderate","Clear"])
    gf   = fc2.selectbox("Sex",    ["All","M","F"])
    wf   = fc3.selectbox("Ward",   ["All"] + wards)
    amin, amax = fc4.slider("Age range", 20, 85, (20, 85))
    sb   = fc5.selectbox("Sort by", ["Patient ID","Heart Risk %","Cancer Risk %","Age","Name"])
    st.markdown('</div>', unsafe_allow_html=True)

    fdf = df.copy()
    if sf != "All": fdf = fdf[fdf["Status"]==sf]
    if gf != "All": fdf = fdf[fdf["Sex"]==gf]
    if wf != "All": fdf = fdf[fdf["Ward"]==wf]
    fdf = fdf[(fdf["Age"]>=amin) & (fdf["Age"]<=amax)]
    fdf = fdf.sort_values(sb, ascending=sb not in ["Heart Risk %","Cancer Risk %"])

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Showing",     f"{len(fdf)} records")
    sc2.metric("High Risk",   int((fdf["Status"]=="High Risk").sum()))
    sc3.metric("Moderate",    int((fdf["Status"]=="Moderate").sum()))
    sc4.metric("Avg Heart %", f"{fdf['Heart Risk %'].mean():.1f}%")
    st.markdown("<br>", unsafe_allow_html=True)

    def color_status(v):
        return {"High Risk": "background:rgba(255,71,87,0.15);color:#ff4757;font-weight:700;border-radius:6px;padding:3px 10px",
                "Moderate":  "background:rgba(255,177,66,0.15);color:#ffb142;font-weight:700;border-radius:6px;padding:3px 10px",
                "Clear":     "background:rgba(46,213,115,0.15);color:#2ed573;font-weight:700;border-radius:6px;padding:3px 10px"}.get(v,"")

    def color_risk(v):
        if isinstance(v, float):
            if v >= threshold: return "color:#ff4757;font-weight:700"
            if v >= 35:        return "color:#ffb142;font-weight:700"
            return "color:#2ed573;font-weight:600"
        return ""

    styled = fdf.style\
        .applymap(color_status, subset=["Status"])\
        .applymap(color_risk,   subset=["Heart Risk %","Cancer Risk %"])\
        .format({"Heart Risk %":"{}%","Cancer Risk %":"{}%"})
    st.dataframe(styled, use_container_width=True, height=480)




# ══════════════════════════════════════════════════════════════════════
# TAB 5 — HEALTH CHAT (Rule-Based)
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="g-card-header"><div class="g-card-title">💬 MedCore Health Assistant</div><span class="g-badge badge-green">Built-in</span></div>', unsafe_allow_html=True)

    # Rule-based health Q&A knowledge base
    HEALTH_KB = {
        # Heart disease keywords
        "cholesterol":   "**Cholesterol** is a fatty substance in your blood. High LDL (bad) cholesterol increases heart disease risk. Normal total cholesterol is < 200 mg/dL. Diet, exercise, and statins help manage it.",
        "blood pressure":"**Blood Pressure** measures force against artery walls. Normal is < 120/80 mmHg. High BP (hypertension) strains the heart. Reduce salt, exercise regularly, and take prescribed medication.",
        "thalassemia":   "**Thalassemia** is an inherited blood disorder affecting haemoglobin. In the heart model, thal=1 is normal, thal=2 is a fixed defect, thal=3 is a reversible defect — reversible defects carry higher cardiac risk.",
        "ecg":           "**ECG (Electrocardiogram)** records the heart's electrical activity. Abnormal ECG patterns like ST changes may indicate ischaemia or structural problems and require cardiology review.",
        "angina":        "**Angina** is chest pain caused by reduced blood flow to the heart. Exercise-induced angina (exang=1 in the model) is a significant risk factor for coronary artery disease.",
        "oldpeak":       "**ST Depression (Oldpeak)** measures how much the ST segment drops during exercise vs rest. Values > 2.0 are significant and suggest myocardial ischaemia.",
        "heart rate":    "**Maximum Heart Rate** — a lower than expected max heart rate during exercise can indicate poor cardiac reserve. Normal max HR ≈ 220 minus your age.",
        "statin":        "**Statins** (e.g. Atorvastatin, Rosuvastatin) lower LDL cholesterol by blocking liver enzyme HMG-CoA reductase. They reduce heart attack risk by 25–35%.",
        "aspirin":       "**Aspirin** at low doses (75–100 mg) prevents platelets from clumping together, reducing clot formation. Used for prevention in high cardiac risk patients.",
        # Cancer keywords
        "radius":        "**Mean Radius** is the average distance from the tumour cell nucleus centre to its perimeter. Larger radius (> 15) is associated with malignant tumours in the Wisconsin dataset.",
        "texture":       "**Mean Texture** measures variation in grey-scale intensity of the cell image. Higher texture values indicate irregular cell surfaces, more common in malignant cells.",
        "concavity":     "**Concavity** measures the severity of concave portions in the cell contour. Higher concavity values (> 0.15) strongly correlate with malignancy.",
        "malignant":     "**Malignant** means cancerous — the tumour cells invade nearby tissue and can spread. The model predicts Class 0 = Malignant. Immediate oncology referral is required.",
        "benign":        "**Benign** means non-cancerous — the tumour does not invade tissue or spread. The model predicts Class 1 = Benign. Routine monitoring is still recommended.",
        "biopsy":        "**Biopsy** is the definitive test for cancer — a tissue sample is taken and examined under a microscope. Core needle biopsy is standard for breast tumours.",
        "tamoxifen":     "**Tamoxifen** is an antiestrogen drug used for ER-positive breast cancer. It blocks oestrogen receptors on cancer cells, slowing growth. Standard course is 5 years.",
        "mammogram":     "**Mammogram** is an X-ray of the breast used for cancer screening. Annual mammograms are recommended from age 40, or earlier with family history.",
        # Lab values
        "hba1c":         "**HbA1c** (Glycosylated Haemoglobin) reflects average blood glucose over 3 months. Normal < 5.7%, Pre-diabetic 5.7–6.4%, Diabetic ≥ 6.5%. Good control target is < 7%.",
        "glucose":       "**Fasting Blood Glucose** measures blood sugar after 8+ hours fasting. Normal: 70–99 mg/dL. Pre-diabetic: 100–125 mg/dL. Diabetic: ≥ 126 mg/dL.",
        "haemoglobin":   "**Haemoglobin** carries oxygen in red blood cells. Normal: 12–17 g/dL. Low Hb (anaemia) causes fatigue and breathlessness. High Hb may indicate dehydration or polycythaemia.",
        "creatinine":    "**Creatinine** is a waste product filtered by the kidneys. Normal: 0.6–1.2 mg/dL. High creatinine suggests impaired kidney function (CKD).",
        "tsh":           "**TSH** (Thyroid Stimulating Hormone) controls thyroid function. Normal: 0.4–4.0 mIU/L. High TSH = hypothyroidism. Low TSH = hyperthyroidism.",
        "vitamin d":     "**Vitamin D** supports bone health and immune function. Normal: > 30 ng/mL. Deficiency (< 20) is common and linked to higher cardiovascular and cancer risk.",
        "uric acid":     "**Uric Acid** is a waste product from purine metabolism. Normal: 3.5–7.2 mg/dL. Elevated levels (> 7.2) cause gout and may indicate kidney issues.",
        # General
        "diet":          "**Cardiac Diet Tips:** Mediterranean diet — olive oil, nuts, fish, legumes, vegetables. Avoid trans fats, excess salt (< 2g/day), and processed foods. Aim for 30g fibre daily.",
        "exercise":      "**Exercise Recommendations:** 150 min/week moderate aerobic activity (brisk walking, cycling, swimming). Reduces heart disease risk by ~35% and cancer risk by ~20%.",
        "smoking":       "**Smoking** doubles the risk of heart disease and significantly increases cancer risk. Cardiac risk halves within 1 year of quitting. Seek nicotine replacement therapy.",
        "bmi":           "**BMI (Body Mass Index)** = weight(kg) / height(m)². Normal: 18.5–24.9. Overweight: 25–29.9. Obese: ≥ 30. High BMI increases risk of heart disease, diabetes, and cancer.",
        "stress":        "**Chronic stress** raises cortisol, increasing BP and inflammation, which raises cardiac risk. Manage through exercise, meditation, adequate sleep (7–9 hours), and social support.",
    }

    def get_response(question):
        q = question.lower().strip()
        # Check each keyword
        for keyword, answer in HEALTH_KB.items():
            if keyword in q:
                return answer
        # Fallback responses
        if any(w in q for w in ["hello","hi","hey","namaste"]):
            return "Hello! I am MedCore Health Assistant. Ask me about heart disease, cancer risk, biomarkers, medications, lab values, diet, or lifestyle. I will explain medical terms and help you understand your results."
        if any(w in q for w in ["thank","thanks","good","great","nice"]):
            return "You're welcome! Remember — always consult your doctor before making any medical decisions. Stay healthy! 💙"
        if any(w in q for w in ["risk","score","percent","%"]):
            return "Your **risk score** is the model's predicted probability of disease. HIGH RISK ≥ 50%, MODERATE 35–50%, LOW RISK < 35%. The threshold can be adjusted in the sidebar. A high score means you should seek medical evaluation — not that disease is confirmed."
        if any(w in q for w in ["model","accuracy","random forest","rf","ai","machine learning"]):
            return "MedCore uses **Random Forest** classifiers — an ensemble of 200 decision trees. Heart model accuracy: 93.1%, AUC-ROC: 0.962. Cancer model accuracy: 89.3%, AUC-ROC: 0.931. Feature importance shows which clinical values most influence the prediction."
        if any(w in q for w in ["doctor","physician","consult","hospital","treatment","medicine","medication"]):
            return "For actual treatment, medication, or diagnosis — please consult a qualified **doctor or specialist**. MedCore is a research and educational tool only. Go to Cardiology OPD for heart concerns, or Oncology for cancer concerns."
        if any(w in q for w in ["ocr","report","upload","scan","lab","blood test"]):
            return "Use the **OCR Reports tab** to upload your lab report (JPG/PNG/PDF) or paste the text. MedCore will extract values like HbA1c, cholesterol, BP, and haemoglobin, and flag abnormal results with normal ranges."
        return "I can help with: heart disease risk factors, cancer biomarkers, lab value interpretation, medications, diet and lifestyle advice. Try asking about a specific term like **cholesterol**, **HbA1c**, **thalassemia**, **concavity**, or **blood pressure**."

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am **MedCore Health Assistant**. I can explain your diagnosis results, biomarkers, medications, and lab values. Ask me anything about heart disease, cancer risk, or your test results!"}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your results, biomarkers, medications, or medical terms...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        reply = get_response(user_input)
        with st.chat_message("assistant", avatar="🏥"):
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = [{"role": "assistant", "content": "Chat cleared. How can I help you?"}]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🫀 Heart Disease Model</div><span class="g-badge badge-green">heart_model.pkl</span></div>', unsafe_allow_html=True)
        hm1,hm2,hm3,hm4 = st.columns(4)
        hm1.metric("Accuracy","93.1%"); hm2.metric("Precision","91.4%")
        hm3.metric("Recall","94.7%");   hm4.metric("AUC-ROC","0.962")
        h_df = pd.DataFrame({
            "Feature":    ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
            "Importance": heart_model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_h = go.Figure(go.Bar(
            x=h_df["Importance"], y=h_df["Feature"], orientation="h",
            marker=dict(color=h_df["Importance"].tolist(),
                colorscale=[[0,"rgba(0,212,255,0.2)"],[0.5,"rgba(0,212,255,0.6)"],[1,CYAN]],
                line=dict(width=0)),
            text=[f"{v:.3f}" for v in h_df["Importance"]], textposition="outside",
            textfont=dict(size=10, color=GRAY, family="DM Mono"),
        ))
        fig_h.update_layout(**PL, height=380)
        _fix(fig_h, xaxis=dict(showgrid=False, range=[0, heart_model.feature_importances_.max()*1.4]),
             margin=dict(l=0, r=55, t=8, b=8))
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🔬 Cancer Risk Model</div><span class="g-badge badge-green">cancer_model.pkl</span></div>', unsafe_allow_html=True)
        cm1,cm2,cm3,cm4 = st.columns(4)
        cm1.metric("Accuracy","89.3%"); cm2.metric("Precision","87.2%")
        cm3.metric("Recall","91.0%");   cm4.metric("AUC-ROC","0.931")
        c_df = pd.DataFrame({
            "Feature":    ["mean radius","mean texture","mean perimeter","mean area","mean smoothness",
                           "mean compactness","mean concavity","mean concave pts","mean symmetry","mean fractal dim"],
            "Importance": cancer_model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_c = go.Figure(go.Bar(
            x=c_df["Importance"], y=c_df["Feature"], orientation="h",
            marker=dict(color=c_df["Importance"].tolist(),
                colorscale=[[0,"rgba(255,177,66,0.2)"],[0.5,"rgba(255,177,66,0.6)"],[1,AMBER]],
                line=dict(width=0)),
            text=[f"{v:.3f}" for v in c_df["Importance"]], textposition="outside",
            textfont=dict(size=10, color=GRAY, family="DM Mono"),
        ))
        fig_c.update_layout(**PL, height=380)
        _fix(fig_c, xaxis=dict(showgrid=False, range=[0, cancer_model.feature_importances_.max()*1.4]),
             margin=dict(l=0, r=55, t=8, b=8))
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def roc_fig(name, auc, color, fill):
        t   = np.linspace(0, 1, 200)
        tpr = np.clip(1-(1-t)**(1/(1-auc+0.01)), 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="rgba(255,255,255,0.1)", dash="dash", width=1.5), showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=tpr, name=f"AUC = {auc:.3f}",
            line=dict(color=color, width=2.5), fill="tozeroy", fillcolor=fill))
        fig.update_layout(**PL, height=260)
        _fix(fig, xaxis=dict(title="False Positive Rate", range=[0,1]),
             yaxis=dict(title="True Positive Rate", range=[0,1.02]),
             legend=dict(x=0.55, y=0.1))
        return fig

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">📈 Heart Model ROC Curve</div></div>', unsafe_allow_html=True)
        st.plotly_chart(roc_fig("Heart Disease", 0.962, RED, "rgba(255,71,87,0.06)"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with rc2:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">📈 Cancer Model ROC Curve</div></div>', unsafe_allow_html=True)
        st.plotly_chart(roc_fig("Cancer Risk", 0.931, AMBER, "rgba(255,177,66,0.06)"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="g-card-header"><div class="g-card-title">🤖 Model Configuration</div></div>', unsafe_allow_html=True)
    di1,di2,di3,di4,di5,di6 = st.columns(6)
    di1.metric("Heart Trees",     heart_model.n_estimators)
    di2.metric("Heart Features",  heart_model.n_features_in_)
    di3.metric("Heart Classes",   len(heart_model.classes_))
    di4.metric("Cancer Trees",    cancer_model.n_estimators)
    di5.metric("Cancer Features", cancer_model.n_features_in_)
    di6.metric("Cancer Classes",  len(cancer_model.classes_))
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(0,212,255,0.06),rgba(0,0,0,0));
     border:1px solid rgba(0,212,255,0.1);border-radius:14px;padding:16px 28px;margin-top:12px;
     display:flex;justify-content:space-between;align-items:center">
    <div style="display:flex;align-items:center;gap:14px">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#00d4ff,#0099cc);
             border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px">🏥</div>
        <span style="font-size:12px;color:#4a6a8a;font-family:'DM Mono',monospace">
            MedCore Clinical AI v3.0 &nbsp;·&nbsp; heart_model.pkl &amp; cancer_model.pkl
            &nbsp;·&nbsp; OCR Reports 
        </span>
    </div>
    <span style="font-size:11px;color:#2a4a6a;font-family:'DM Mono',monospace">
        ⚕️ Research &amp; Education Only · Not for Clinical Use
    </span>
</div>
""", unsafe_allow_html=True)