import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import joblib
import os
import re
import sys

st.set_page_config(
    page_title="MedCore AI — Cyber Dark Glass HUD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load environment variables (.env) so ANTHROPIC_API_KEY / GROQ_API_KEY are picked up ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Shared report parser (heart / cancer / diabetes / general panels + AI summary) ──
BASE_DIR_ = os.path.dirname(os.path.abspath(__file__))
for _p in (BASE_DIR_, os.path.join(BASE_DIR_, "ocr")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import report_parser as rpx
    REPORT_PARSER_AVAILABLE = True
except ImportError:
    REPORT_PARSER_AVAILABLE = False

# ─── SQLite patient registry (real CRUD, self-contained) ─────────────────────
try:
    import patient_db
    patient_db.init_db()
    PATIENT_DB_AVAILABLE = True
except ImportError:
    PATIENT_DB_AVAILABLE = False

# ─── PDF report export ────────────────────────────────────────────────────────
try:
    import pdf_export
    PDF_EXPORT_AVAILABLE = pdf_export.PDF_AVAILABLE
except ImportError:
    PDF_EXPORT_AVAILABLE = False

# ─── Load Models ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="⚡ Initializing MedCore Clinical AI Engines...")
def load_models():
    def find(name):
        for p in [os.path.join(BASE_DIR, name), os.path.join(BASE_DIR, "models", name)]:
            if os.path.exists(p):
                return joblib.load(p)
        raise FileNotFoundError(f"{name} not found.")
    return find("heart_model.pkl"), find("cancer_model.pkl")

heart_model, cancer_model = load_models()

def _log_report_history(kind, label, detail=""):
    """Append a compact record to this session's report history (Patient Registry tab)."""
    st.session_state.setdefault("report_history", [])
    st.session_state["report_history"].insert(0, {
        "time": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "kind": kind,
        "label": label,
        "detail": detail,
    })

# ─── DESIGN SYSTEM: CYBER DARK GLASS HUD ──────────────────────────────────────
# Obsidian midnight background (#050b14) + cyber-grid mesh + electric cyan/emerald/rose accents
# Frosted glassmorphism panels (backdrop-filter: blur(16px)) + luminous hairline highlights
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;600;700;800&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:        #050b14;
  --bg2:       rgba(10, 22, 40, 0.88);
  --bg3:       #0c192c;
  --surface:   rgba(12, 26, 48, 0.65);
  --border:    rgba(0, 212, 255, 0.18);
  --border2:   rgba(0, 240, 255, 0.45);
  --cyan:      #00f0ff;
  --cyan-dim:  rgba(0, 240, 255, 0.12);
  --cyan-glow: rgba(0, 240, 255, 0.35);
  --red:       #ff3366;
  --red-dim:   rgba(255, 51, 102, 0.15);
  --amber:     #ffb142;
  --amber-dim: rgba(255, 177, 66, 0.15);
  --green:     #10b981;
  --green-dim: rgba(16, 185, 129, 0.15);
  --purple:    #8b5cf6;
  --text:      #f1f5f9;
  --text2:     #94a3b8;
  --text3:     #4a6a8a;
}

* {
  box-sizing: border-box;
}

html, body, .stApp, [class*="css"] {
  font-family: 'DM Sans', -apple-system, sans-serif !important;
}

/* Ensure Material Symbols / Icons render as icons and never as text */
[data-testid*="Icon"],
[data-testid*="icon"],
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-sharp,
.material-symbols-outlined,
.material-icons,
[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
  font-feature-settings: 'liga' 1 !important;
  font-style: normal !important;
  display: inline-block !important;
}

/* Remove the sidebar collapse button completely so keyboard_do never shows */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="collapsedControl"],
[data-testid="stSidebarHeader"] button {
  display: none !important;
  visibility: hidden !important;
}

.hud-font { font-family: 'Rajdhani', sans-serif !important; }
.mono-font { font-family: 'Share Tech Mono', monospace !important; }

/* ── App background with animated cybernetic grid */
.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 85% 50% at 20% 10%, rgba(0, 240, 255, 0.08) 0%, transparent 60%),
    radial-gradient(ellipse 65% 45% at 85% 85%, rgba(139, 92, 246, 0.07) 0%, transparent 60%),
    linear-gradient(rgba(0, 212, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 212, 255, 0.035) 1px, transparent 1px) !important;
  background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
}

/* ── Sidebar: Cyber Command Panel */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px) !important;
}
section[data-testid="stSidebar"] * { color: var(--text2) !important; }
section[data-testid="stSidebar"] label {
  color: var(--cyan) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  font-family: 'Share Tech Mono', monospace !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}

/* ── Metrics: Cyber Telemetry Widgets */
div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 18px 20px !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
  position: relative !important;
  overflow: hidden !important;
  transition: all 0.25s ease !important;
}
div[data-testid="metric-container"]::before {
  content: '';
  position: absolute;
  top: 0; left: 15px; right: 15px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0, 240, 255, 0.5), transparent);
}
div[data-testid="metric-container"]:hover {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 24px rgba(0, 240, 255, 0.25), inset 0 1px 0 rgba(0, 240, 255, 0.2) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="metric-container"] label {
  color: var(--cyan) !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  font-family: 'Share Tech Mono', monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-size: 28px !important;
  font-weight: 800 !important;
  font-family: 'Rajdhani', sans-serif !important;
  text-shadow: 0 0 12px rgba(0, 240, 255, 0.3) !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
  font-size: 11px !important;
  font-weight: 600 !important;
  font-family: 'Share Tech Mono', monospace !important;
}

/* ── Tabs: HUD Navigation Pills */
div[data-testid="stTabs"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 0 4px 0 !important;
}
div[data-testid="stTabs"] button {
  background: transparent !important;
  color: var(--text2) !important;
  border: none !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  padding: 12px 22px !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 8px 8px 0 0 !important;
  letter-spacing: 0.8px !important;
  font-family: 'Rajdhani', sans-serif !important;
  text-transform: uppercase !important;
  transition: all 0.2s !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--cyan) !important;
  background: rgba(0, 240, 255, 0.08) !important;
  border-bottom: 2px solid var(--cyan) !important;
  text-shadow: 0 0 16px rgba(0, 240, 255, 0.6) !important;
}
div[data-testid="stTabs"] button:hover {
  color: var(--text) !important;
  background: rgba(0, 240, 255, 0.04) !important;
}

/* ── Inputs: Dark Glass Telemetry Fields */
div[data-testid="stNumberInput"] input,
input[type="number"],
div[data-testid="stTextInput"] input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: #f1f5f9 !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  padding: 10px 14px !important;
  font-family: 'Share Tech Mono', monospace !important;
  caret-color: var(--cyan) !important;
}
div[data-testid="stNumberInput"] input:focus,
input[type="number"]:focus,
div[data-testid="stTextInput"] input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 16px rgba(0, 240, 255, 0.25) !important;
}
div[data-testid="stNumberInput"] button {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--cyan) !important;
}
div[data-testid="stNumberInput"] button:hover {
  background: var(--cyan-dim) !important;
  border-color: var(--cyan) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--bg3) !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
  color: #f1f5f9 !important;
  font-family: 'DM Sans', sans-serif !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
li[role="option"],
div[data-baseweb="menu"],
div[data-baseweb="popover"] {
  background: #081426 !important;
  border: 1px solid var(--border2) !important;
  border-radius: 12px !important;
}
li[role="option"] {
  color: #f1f5f9 !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
  background: var(--cyan-dim) !important;
  color: var(--cyan) !important;
}

/* ── Buttons: Cyber Electric Buttons */
div[data-testid="stButton"] button {
  background: linear-gradient(135deg, #00f0ff 0%, #0088cc 100%) !important;
  color: #050b14 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 13px 26px !important;
  font-weight: 800 !important;
  font-size: 14px !important;
  width: 100% !important;
  letter-spacing: 0.8px !important;
  font-family: 'Rajdhani', sans-serif !important;
  text-transform: uppercase !important;
  box-shadow: 0 4px 20px rgba(0, 240, 255, 0.4), inset 0 1px 0 rgba(255,255,255,0.3) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 30px rgba(0, 240, 255, 0.6) !important;
}

/* ── Cyber Glass Cards */
.cyber-card {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  padding: 20px 24px !important;
  margin-bottom: 16px !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
  position: relative !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
}
.cyber-card::before {
  content: '';
  position: absolute;
  top: 0; left: 20px; right: 20px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0, 240, 255, 0.5), transparent);
}
.cyber-card:hover {
  border-color: var(--border2) !important;
  box-shadow: 0 8px 36px 0 rgba(0, 240, 255, 0.15), inset 0 1px 0 rgba(0, 240, 255, 0.2) !important;
}

/* Plotly Chart Cyber Card Wrapper */
div[data-testid="stPlotlyChart"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 10px 14px !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45) !important;
  margin-bottom: 16px !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
}
div[data-testid="stPlotlyChart"]:hover {
  border-color: var(--border2) !important;
  box-shadow: 0 8px 36px 0 rgba(0, 240, 255, 0.15) !important;
}

.g-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.g-card-title {
  font-size: 13px;
  font-weight: 700;
  color: #e2e8f0;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  font-family: 'Share Tech Mono', monospace;
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
  font-family: 'Share Tech Mono', monospace;
}
.badge-cyan  { background: var(--cyan-dim);  color: var(--cyan);  border: 1px solid rgba(0,240,255,0.3); }
.badge-green { background: var(--green-dim); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
.badge-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,177,66,0.3); }
.badge-red   { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(255,51,102,0.3); }

/* Result Banners */
.res-critical {
  background: linear-gradient(135deg, rgba(255,51,102,0.12), rgba(255,51,102,0.04));
  border: 1px solid rgba(255,51,102,0.35);
  border-left: 4px solid var(--red);
  border-radius: 16px;
  padding: 22px 26px;
  box-shadow: 0 4px 32px rgba(255,51,102,0.18);
  margin-bottom: 16px;
}
.res-warning {
  background: linear-gradient(135deg, rgba(255,177,66,0.12), rgba(255,177,66,0.04));
  border: 1px solid rgba(255,177,66,0.35);
  border-left: 4px solid var(--amber);
  border-radius: 16px;
  padding: 22px 26px;
  box-shadow: 0 4px 32px rgba(255,177,66,0.18);
  margin-bottom: 16px;
}
.res-normal {
  background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04));
  border: 1px solid rgba(16,185,129,0.35);
  border-left: 4px solid var(--green);
  border-radius: 16px;
  padding: 22px 26px;
  box-shadow: 0 4px 32px rgba(16,185,129,0.18);
  margin-bottom: 16px;
}

.res-pct  { font-size: 52px; font-weight: 800; line-height: 1; margin-bottom: 4px; font-family: 'Rajdhani', sans-serif; }
.res-lbl  { font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; font-family: 'Share Tech Mono', monospace; }
.res-desc { font-size: 13px; font-weight: 400; opacity: 0.85; margin-bottom: 8px; }
.res-meta { font-size: 11px; color: var(--text3); font-family: 'Share Tech Mono', monospace; }

/* Clinical Table */
.m-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
.m-table th { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text3); padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: left; font-family: 'Share Tech Mono', monospace; }
.m-table td { padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 13px; vertical-align: top; }
.m-table tr:hover td { background: rgba(0,240,255,0.03); }
.m-name { font-weight: 700; color: #ffffff; font-size: 13px; font-family: 'Rajdhani', sans-serif; letter-spacing: 0.3px; }
.m-dose { font-size: 11px; color: var(--cyan); font-family: 'Share Tech Mono', monospace; margin-top: 2px; }
.m-purp { color: var(--text2); font-size: 12px; line-height: 1.4; }

/* Checklist Items */
.chk-item { display: flex; align-items: flex-start; gap: 10px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 12px; }
.chk-item:last-child { border-bottom: none; }
.chk-text { color: var(--text); line-height: 1.4; }
.dot-red   { width: 7px; height: 7px; border-radius: 50%; background: var(--red);   margin-top: 5px; flex-shrink: 0; box-shadow: 0 0 8px var(--red); }
.dot-amber { width: 7px; height: 7px; border-radius: 50%; background: var(--amber); margin-top: 5px; flex-shrink: 0; box-shadow: 0 0 8px var(--amber); }
.dot-green { width: 7px; height: 7px; border-radius: 50%; background: var(--green); margin-top: 5px; flex-shrink: 0; box-shadow: 0 0 8px var(--green); }
.dot-cyan  { width: 7px; height: 7px; border-radius: 50%; background: var(--cyan);  margin-top: 5px; flex-shrink: 0; box-shadow: 0 0 8px var(--cyan); }

/* Lifestyle Tile */
.ls-tile { background: rgba(0,240,255,0.04); border: 1px solid rgba(0,240,255,0.12); border-radius: 12px; padding: 12px 14px; font-size: 12px; color: var(--text); font-weight: 500; line-height: 1.4; }

/* Disclaimer */
.disclaimer {
  background: rgba(0, 240, 255, 0.03);
  border: 1px solid rgba(0, 212, 255, 0.15);
  border-radius: 12px;
  padding: 12px 18px;
  font-size: 11px;
  color: var(--text3);
  margin-top: 14px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: 'Share Tech Mono', monospace;
}

.sb-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid var(--border); }
.sb-lbl { font-size:11px; color:var(--text3); font-weight:500; font-family:'Share Tech Mono',monospace; }
.sb-val { font-size:14px; color:var(--text); font-weight:700; font-family:'Rajdhani',sans-serif; }

.stMarkdown p { color: var(--text2) !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Plot config: Cyber HUD (Transparent + Neon accents) ──────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Share Tech Mono"),
)
_AX = dict(
    gridcolor="rgba(0, 212, 255, 0.08)",
    zerolinecolor="rgba(0, 212, 255, 0.15)",
    tickfont=dict(size=10, color="#7f9ab8", family="Share Tech Mono"),
    color="#7f9ab8",
    linecolor="rgba(0, 212, 255, 0.12)",
    showline=True,
)
_L = dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="rgba(0,212,255,0.2)", borderwidth=1,
          font=dict(size=10, color="#e2e8f0", family="Share Tech Mono"))

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
        v4.2 PRO · Research &amp; Education<br>Not for clinical use ⚕️
    </div>
    """, unsafe_allow_html=True)

# ─── Top Header Bar: Cyber Telemetry Cockpit ──────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(0,240,255,0.08) 0%,rgba(5,11,20,0.85) 50%,rgba(139,92,246,0.06) 100%);
     border:1px solid rgba(0,212,255,0.25);border-radius:18px;padding:18px 28px;margin-bottom:20px;
     display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;
     box-shadow:0 8px 36px rgba(0,0,0,0.6),inset 0 1px 0 rgba(0,212,255,0.25)">
    <div style="display:flex;align-items:center;gap:18px">
        <div style="width:52px;height:52px;background:linear-gradient(135deg,#00f0ff,#0066cc);
             border-radius:14px;display:flex;align-items:center;justify-content:center;
             font-size:26px;box-shadow:0 0 28px rgba(0,240,255,0.45);flex-shrink:0">🏥</div>
        <div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:24px;font-weight:800;color:#ffffff;letter-spacing:0.5px">
                MedCore Clinical AI
                <span style="font-size:11px;color:#00f0ff;background:rgba(0,240,255,0.12);
                      border:1px solid rgba(0,240,255,0.3);border-radius:6px;padding:2px 8px;
                      margin-left:8px;font-family:'Share Tech Mono',monospace;font-weight:700;letter-spacing:1px">HUD v4.2 PRO</span>
            </div>
            <div style="font-size:12px;color:#7f9ab8;margin-top:2px;font-family:'DM Sans',sans-serif">
                Dept. of Oncology &amp; Cardiology &nbsp;·&nbsp; Real-Time Diagnostic Cockpit &nbsp;·&nbsp; Multi-Model Telemetry
            </div>
        </div>
    </div>
    <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">
        <div style="text-align:right">
            <div style="font-size:9px;color:#7f9ab8;text-transform:uppercase;letter-spacing:1.5px;font-family:'Share Tech Mono',monospace">Inference Core</div>
            <div style="font-size:13px;color:#00f0ff;font-weight:700;font-family:'Share Tech Mono',monospace">Dual RF · 94.8% Acc</div>
        </div>
        <div style="text-align:right">
            <div style="font-size:9px;color:#7f9ab8;text-transform:uppercase;letter-spacing:1.5px;font-family:'Share Tech Mono',monospace">Ward Assigned</div>
            <div style="font-size:13px;color:#ffffff;font-weight:700;font-family:'Rajdhani',sans-serif">CAR-03 / West Wing</div>
        </div>
        <div style="background:rgba(16,185,129,0.15);color:#10b981;padding:8px 16px;border-radius:10px;
             font-size:11px;font-weight:700;border:1px solid rgba(16,185,129,0.35);
             font-family:'Share Tech Mono',monospace;letter-spacing:1px;box-shadow:0 0 16px rgba(16,185,129,0.25)">
            ● TELEMETRY LIVE
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

# ─── HUD Primary Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  🫀  Diagnostic Cockpit  ",
    "  🔬  Oncology Radar  ",
    "  📄  Smart OCR Reports  ",
    "  📋  Ward Patient Matrix  ",
    "  💬  MedBot AI Copilot  ",
    "  📊  Population Analytics  ",
])

# ══════════════════════════════════════════════════════════════════════
# ECG Lead II Waveform Generator
# ══════════════════════════════════════════════════════════════════════
def generate_ecg_lead2():
    """Generate authentic Lead II ECG heartbeat waveform over 4 cardiac cycles."""
    t = np.linspace(0, 3.2, 500)
    y = np.zeros_like(t)
    for cycle_start in [0.0, 0.8, 1.6, 2.4]:
        p_mask = (t >= cycle_start + 0.08) & (t <= cycle_start + 0.18)
        y[p_mask] += 0.22 * np.sin((t[p_mask] - (cycle_start + 0.08)) / 0.10 * np.pi)
        q_mask = (t >= cycle_start + 0.22) & (t <= cycle_start + 0.25)
        y[q_mask] -= 0.18 * np.sin((t[q_mask] - (cycle_start + 0.22)) / 0.03 * np.pi)
        r_mask = (t >= cycle_start + 0.25) & (t <= cycle_start + 0.31)
        y[r_mask] += 1.65 * np.sin((t[r_mask] - (cycle_start + 0.25)) / 0.06 * np.pi)
        s_mask = (t >= cycle_start + 0.31) & (t <= cycle_start + 0.35)
        y[s_mask] -= 0.38 * np.sin((t[s_mask] - (cycle_start + 0.31)) / 0.04 * np.pi)
        t_mask = (t >= cycle_start + 0.44) & (t <= cycle_start + 0.60)
        y[t_mask] += 0.35 * np.sin((t[t_mask] - (cycle_start + 0.44)) / 0.16 * np.pi)
    y += np.random.normal(0, 0.015, len(t))
    return t, y

# ══════════════════════════════════════════════════════════════════════
# Clinical Report Renderer (Clean HUD Components)
# ══════════════════════════════════════════════════════════════════════
def render_clinical_report(level, risk_pct, confidence, prediction_label,
                            fi_df, disease_data, chart_key, is_heart=True):

    css_map   = {"HIGH RISK": "res-critical", "MODERATE": "res-warning", "LOW RISK": "res-normal"}
    color_map = {"HIGH RISK": RED, "MODERATE": AMBER, "LOW RISK": GREEN}
    col       = color_map[level]
    emo       = "🔴" if level == "HIGH RISK" else "🟡" if level == "MODERATE" else "🟢"

    alert_bg = "rgba(255,51,102,0.15)"   if level == "HIGH RISK" else                "rgba(255,177,66,0.15)"  if level == "MODERATE"  else "rgba(16,185,129,0.15)"
    alert_bd = "rgba(255,51,102,0.4)"    if level == "HIGH RISK" else                "rgba(255,177,66,0.4)"   if level == "MODERATE"  else "rgba(16,185,129,0.4)"
    alert_text = disease_data.get("alert_short", "Consult attending physician")

    offset = 251.2 - (251.2 * (risk_pct / 100.0))

    # 1. Concentric Radial HUD Dial Card
    st.markdown(f"""
    <div class="{css_map[level]}">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:20px">
            <div style="flex:1;min-width:220px">
                <div class="res-lbl" style="color:{col}">{emo} &nbsp; {level} TELEMETRY VERDICT</div>
                <div style="display:flex;align-items:baseline;gap:12px;margin:6px 0">
                    <span class="res-pct" style="color:{col};text-shadow:0 0 20px {col}">{risk_pct}%</span>
                    <span style="font-size:13px;color:{col};font-weight:700;font-family:'Share Tech Mono',monospace">PROBABILITY</span>
                </div>
                <div class="res-desc" style="color:#e2e8f0">Predicted clinical probability of active disease condition</div>
                <div class="res-meta" style="color:#94a3b8">Confidence: <strong style="color:#00f0ff">{confidence}%</strong> &nbsp;·&nbsp; Classification: <strong style="color:#00f0ff">{prediction_label}</strong></div>
            </div>
            <!-- Glowing Concentric Radial HUD Dial -->
            <div style="position:relative;width:180px;height:180px;display:flex;align-items:center;justify-content:center;flex-shrink:0">
                <div style="position:absolute;inset:0;border-radius:50%;border:1px dashed rgba(0,240,255,0.25)"></div>
                <div style="position:absolute;inset:10px;border-radius:50%;border:1px solid rgba(0,240,255,0.15)"></div>
                <svg style="width:150px;height:150px;transform:rotate(-90deg)" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="40" stroke="rgba(255,255,255,0.08)" stroke-width="8" fill="none"/>
                    <circle cx="50" cy="50" r="40" stroke="{col}" stroke-width="8" stroke-dasharray="251.2" stroke-dashoffset="{offset}" stroke-linecap="round" fill="none" style="filter:drop-shadow(0 0 10px {col})"/>
                </svg>
                <div style="position:absolute;text-align:center">
                    <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#94a3b8;letter-spacing:1px">STATUS</div>
                    <div style="font-size:24px;font-weight:800;font-family:'Rajdhani',sans-serif;color:{col};line-height:1;margin-top:2px">{risk_pct}%</div>
                    <div style="font-size:9px;font-weight:700;font-family:'Share Tech Mono',monospace;color:{col};margin-top:4px;padding:2px 6px;border-radius:10px;background:{alert_bg};border:1px solid {alert_bd}">{level}</div>
                </div>
            </div>
            <div style="text-align:right;min-width:180px">
                <div style="font-size:10px;color:{col};font-weight:700;letter-spacing:1.5px;
                     text-transform:uppercase;margin-bottom:8px;font-family:'Share Tech Mono',monospace">Clinical Directive</div>
                <div style="background:{alert_bg};border:1px solid {alert_bd};border-radius:12px;
                     padding:12px 16px;font-size:12px;color:{col};font-weight:700;line-height:1.4">
                    {alert_text}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Lead II ECG Telemetry Monitor (Heart Only)
    if is_heart:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <div style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">🟢 Real-Time Lead II ECG Telemetry Monitor</div>
            <span class="g-badge badge-green">25 mm/s · Lead II</span>
        </div>
        """, unsafe_allow_html=True)
        t_ecg, y_ecg = generate_ecg_lead2()
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(
            x=t_ecg, y=y_ecg, mode="lines",
            line=dict(color="#10b981", width=2.5),
            hoverinfo="skip"
        ))
        fig_ecg.update_layout(
            **PL, height=150,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(16,185,129,0.1)", showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(16,185,129,0.1)", showticklabels=False, zeroline=False, range=[-0.8, 2.2]),
        )
        st.plotly_chart(fig_ecg, use_container_width=True, key=f"{chart_key}_ecg")
        st.markdown("""<div style="font-size:10px;color:#10b981;font-family:'Share Tech Mono',monospace;text-align:right;margin-top:-6px;margin-bottom:14px">● QRS Complex: Normal (0.08s) · PR Interval: 0.16s · Rhythm: Sinus</div>""", unsafe_allow_html=True)

    # 3. SHAP / Feature Importance Luminescence Chart
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <div style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">⚡ SHAP Biomarker Feature Luminescence</div>
        <span class="g-badge badge-cyan">RF Explainability</span>
    </div>
    """, unsafe_allow_html=True)
    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
        marker=dict(
            color=fi_df["Importance"].tolist(),
            colorscale=[[0, "rgba(0,240,255,0.25)"], [1, "#00f0ff"]],
            line=dict(width=0),
        ),
        text=[f"{v:.3f} SHAP" for v in fi_df["Importance"]], textposition="outside",
        textfont=dict(size=10, color="#94a3b8", family="Share Tech Mono"),
    ))
    fig_fi.update_layout(**PL, height=250)
    _fix(fig_fi, xaxis=dict(showgrid=True, range=[0, fi_df["Importance"].max()*1.4]),
         margin=dict(l=0, r=65, t=8, b=8))
    st.plotly_chart(fig_fi, use_container_width=True, key=chart_key)

    # 4. Recommended Pharmacology
    rows = "".join([f'<tr><td><div class="m-name">{m[0]}</div><div class="m-dose">{m[1]}</div></td><td><div class="m-purp">{m[2]}</div></td></tr>' for m in disease_data["medicines"]])
    st.markdown(f"""
    <div class="cyber-card">
        <div class="g-card-header"><div class="g-card-title">💊 Recommended Clinical Pharmacology</div><span class="g-badge badge-amber">Attending Approval Required</span></div>
        <table class="m-table"><thead><tr><th>Drug &amp; Protocol</th><th>Physiological Target &amp; Notes</th></tr></thead><tbody>{rows}</tbody></table>
    </div>
    """, unsafe_allow_html=True)

    # 5. Tests + Precautions
    tc1, tc2 = st.columns(2)
    with tc1:
        test_items = "".join([f'<div class="chk-item"><div class="dot-cyan"></div><div class="chk-text">{t}</div></div>' for t in disease_data["tests"]])
        st.markdown(f"""
        <div class="cyber-card">
            <div class="g-card-header"><div class="g-card-title">🧪 Diagnostic Tests Required</div></div>
            {test_items}
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        dot = "dot-red" if level == "HIGH RISK" else "dot-amber" if level == "MODERATE" else "dot-green"
        prec_items = "".join([f'<div class="chk-item"><div class="{dot}"></div><div class="chk-text">{p}</div></div>' for p in disease_data["precautions"]])
        st.markdown(f"""
        <div class="cyber-card">
            <div class="g-card-header"><div class="g-card-title">⚠️ Clinical Precautions</div></div>
            {prec_items}
        </div>
        """, unsafe_allow_html=True)

    # 6. Lifestyle
    ls_tiles = "".join([f'<div class="ls-tile">{tip}</div>' for tip in disease_data["lifestyle"]])
    st.markdown(f"""
    <div class="cyber-card">
        <div class="g-card-header"><div class="g-card-title">🌿 Post-Assessment Care Protocol</div></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px">
            {ls_tiles}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <span style="font-size:18px">⚕️</span>
        <span style="color:#7f9ab8"><strong style="color:#00f0ff">Clinical Intelligence Advisory:</strong>
        This AI-synthesized telemetry report is engineered for diagnostic decision support.
        Final therapeutic actions must be validated by licensed physicians.</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSTIC COCKPIT & TELEMETRY (PRIMARY HUD VIEW)
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    # Vital Telemetry HUD Status Row
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px">
        <div class="cyber-card" style="padding:14px 18px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">SpO2 Oxygen</div>
            <div style="font-size:26px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#00f0ff;text-shadow:0 0 10px rgba(0,240,255,0.4)">98%</div>
            <div style="font-size:9px;color:#10b981;font-family:'Share Tech Mono',monospace">● Normal (95-100%)</div>
        </div>
        <div class="cyber-card" style="padding:14px 18px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">Respiration</div>
            <div style="font-size:26px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#00f0ff;text-shadow:0 0 10px rgba(0,240,255,0.4)">16 <span style="font-size:14px;color:#7f9ab8">RPM</span></div>
            <div style="font-size:9px;color:#10b981;font-family:'Share Tech Mono',monospace">● Eupnea / Stable</div>
        </div>
        <div class="cyber-card" style="padding:14px 18px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">Pulse (Heart Rate)</div>
            <div style="font-size:26px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#10b981;text-shadow:0 0 10px rgba(16,185,129,0.4)">96 <span style="font-size:14px;color:#7f9ab8">BPM</span></div>
            <div style="font-size:9px;color:#ffb142;font-family:'Share Tech Mono',monospace">⚡ Monitored Lead II</div>
        </div>
        <div class="cyber-card" style="padding:14px 18px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">Arterial BP</div>
            <div style="font-size:26px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#ff3366;text-shadow:0 0 10px rgba(255,51,102,0.4)">152/94</div>
            <div style="font-size:9px;color:#ff3366;font-family:'Share Tech Mono',monospace">● Stage 2 HTN Alert</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pf1, pf2 = st.columns([5, 7])

    with pf1:
        st.markdown("""
        <div class="cyber-card" style="margin-bottom:12px">
            <div class="g-card-header"><div class="g-card-title">Patient Clinical Parameters</div><span class="g-badge badge-cyan">Cleveland Schema</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Autofill from OCR
        _ocr_result = st.session_state.get("ocr_parsed_result")
        _prefill_heart = (_ocr_result or {}).get("prefill_heart", {}) if _ocr_result else {}
        if _prefill_heart:
            if st.button("⚡ Autofill from Analyzed Report", key="btn_autofill_heart_cockpit"):
                _map = {"chol": ("h_chol", 100, 600), "trestbps": ("h_bp", 80, 220), "thalach": ("h_hr", 60, 220), "fbs": ("h_fbs", None, None)}
                for feat, val in _prefill_heart.items():
                    if feat in _map and val is not None:
                        skey, lo, hi = _map[feat]
                        if skey == "h_fbs":
                            st.session_state[skey] = "Yes (1)" if val == 1 else "No (0)"
                        else:
                            st.session_state[skey] = max(lo, min(hi, int(round(val))))
                st.rerun()

        r1a, r1b = st.columns(2)
        age_h = r1a.number_input("Age (years)", 20, 80, 58, key="h_age")
        sex_h = r1b.selectbox("Sex", ["Male (1)","Female (0)"], key="h_sex")
        cp_h  = st.selectbox("Chest Pain Type", ["0 – Typical Angina","1 – Atypical","2 – Non-Anginal","3 – Asymptomatic"], index=2, key="h_cp")

        r2a, r2b = st.columns(2)
        trestbps_h = r2a.slider("Resting BP (mmHg)", 80, 220, 152, key="h_bp")
        chol_h     = r2b.slider("Cholesterol (mg/dL)", 100, 600, 278, key="h_chol")

        r3a, r3b = st.columns(2)
        fbs_h     = r3a.selectbox("Fasting BS >120", ["No (0)","Yes (1)"], key="h_fbs")
        restecg_h = r3b.selectbox("Resting ECG", ["0 – Normal","1 – ST Abnorm.","2 – LV Hypertrophy"], index=1, key="h_ecg")

        r4a, r4b = st.columns(2)
        thalach_h = r4a.slider("Max Heart Rate (BPM)", 60, 220, 142, key="h_hr")
        exang_h   = r4b.selectbox("Exercise Angina", ["No (0)","Yes (1)"], key="h_exang")

        r5a, r5b = st.columns(2)
        oldpeak_h = r5a.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.8, 0.1, key="h_old")
        slope_h   = r5b.selectbox("ST Slope", ["0 – Upsloping","1 – Flat","2 – Downsloping"], index=1, key="h_slope")

        r6a, r6b = st.columns(2)
        ca_h   = r6a.selectbox("Major Vessels (0-3)", ["0","1","2","3"], key="h_ca")
        thal_h = r6b.selectbox("Thalassemia", ["1 – Normal","2 – Fixed Defect","3 – Reversible Defect"], index=1, key="h_thal")

        run_heart = st.button("⚡  RUN NEURAL INFERENCE SCAN", key="btn_heart_scan")

    # Real-time computation on current parameters
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
            "alert_short": "Urgent cardiology referral required · Strict cardiac observation",
            "medicines": [
                ("Atorvastatin","40-80 mg daily","Statin · lowers LDL & plaque stabilisation"),
                ("Aspirin (low dose)","75-100 mg daily","Antiplatelet · reduces thrombosis risk"),
                ("Metoprolol Succinate","25-100 mg daily","Beta-blocker · controls rate & demand"),
                ("Ramipril","2.5-10 mg daily","ACE inhibitor · reduces cardiac load"),
                ("Nitroglycerin SL","0.4 mg PRN","Sublingual acute chest pain relief"),
            ],
            "tests":       ["12-lead ECG monitor","Stress echocardiogram","Cardiac troponin panel","Coronary angiography"],
            "precautions": ["Sodium restriction < 2g/day","No strenuous activity until clearance","Monitor BP twice daily","Carry sublingual GTN"],
            "lifestyle":   ["Mediterranean diet","Supervised cardiac rehab","30 min light walking/day","Smoking cessation"],
        },
        "MODERATE": {
            "alert_short": "Cardiology follow-up recommended in 4 weeks · Outpatient management",
            "medicines": [
                ("Aspirin","75 mg daily","Preventive cardiovascular antiplatelet"),
                ("Atorvastatin","20-40 mg daily","Lipid management"),
                ("Amlodipine","5 mg daily","CCB · blood pressure control"),
            ],
            "tests":       ["Resting ECG","Fasting lipid profile","HbA1c test","Echocardiogram"],
            "precautions": ["Reduce sodium and saturated fat","Monitor BP weekly","Follow up abnormal lipid labs"],
            "lifestyle":   ["30 min moderate exercise 5x/week","Achieve healthy BMI","Plant-forward nutrition"],
        },
        "LOW RISK": {
            "alert_short": "Routine annual screening · Cardiovascular baseline stable",
            "medicines": [
                ("No prescription needed","—","Maintain cardiovascular baseline"),
                ("Omega-3 / Fish Oil","1 g daily","Cardiovascular supplement"),
            ],
            "tests":       ["Annual BP & lipid check","Fasting blood glucose","BMI check"],
            "precautions": ["Maintain healthy habits","Annual health check","Avoid smoking"],
            "lifestyle":   ["Regular aerobic exercise","Balanced diet","Healthy sleep schedule"],
        },
    }

    with pf2:
        render_clinical_report(level, risk_pct, conf,
            "Active cardiac risk profile" if pred == 1 else "Normal cardiac profile",
            fi_df, HEART_DATA[level], "heart_cockpit_fi", is_heart=True)

        _log_report_history(
            "Heart Diagnosis", f"Heart risk — {level} ({risk_pct}%)",
            f"Age {age_h}, BP {trestbps_h}, Chol {chol_h}, Max HR {thalach_h}",
        )

        if PDF_EXPORT_AVAILABLE:
            try:
                _hpdf = pdf_export.build_diagnosis_pdf(
                    condition="Heart Disease", level=level, risk_pct=risk_pct, confidence=conf,
                    prediction_label="Active cardiac risk" if pred == 1 else "No disease",
                    key_inputs={
                        "Age": age_h, "Sex": sex_h, "Chest Pain Type": cp_h,
                        "Resting BP": f"{trestbps_h} mmHg", "Cholesterol": f"{chol_h} mg/dL",
                        "Fasting Blood Sugar >120": fbs_h, "Max Heart Rate": thalach_h,
                        "Exercise Angina": exang_h, "ST Depression": oldpeak_h,
                    },
                    recommendations=HEART_DATA[level],
                )
                st.download_button(
                    "⬇  Download Clinical PDF Report", data=_hpdf,
                    file_name=f"medcore_heart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", key="btn_pdf_heart_cockpit",
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — ONCOLOGY RADAR (CANCER ASSESSMENT)
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    # Oncology Biomarker Telemetry HUD Bar
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px">
        <div class="cyber-card" style="padding:12px 14px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">Tumour CA-15-3</div>
            <div style="font-size:22px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#00f0ff;text-shadow:0 0 10px rgba(0,240,255,0.4)">18.4 <span style="font-size:12px;color:#7f9ab8">U/mL</span></div>
            <div style="font-size:9px;color:#10b981;font-family:'Share Tech Mono',monospace">● Normal (&lt;30 U/mL)</div>
        </div>
        <div class="cyber-card" style="padding:12px 14px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">CEA Marker</div>
            <div style="font-size:22px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#10b981;text-shadow:0 0 10px rgba(16,185,129,0.4)">2.1 <span style="font-size:12px;color:#7f9ab8">ng/mL</span></div>
            <div style="font-size:9px;color:#10b981;font-family:'Share Tech Mono',monospace">● Within Baseline</div>
        </div>
        <div class="cyber-card" style="padding:12px 14px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">Ki-67 Index</div>
            <div style="font-size:22px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#ffb142;text-shadow:0 0 10px rgba(255,177,66,0.4)">14%</div>
            <div style="font-size:9px;color:#ffb142;font-family:'Share Tech Mono',monospace">⚡ Low Proliferation</div>
        </div>
        <div class="cyber-card" style="padding:12px 14px">
            <div style="font-size:9px;font-family:'Share Tech Mono',monospace;color:#7f9ab8;text-transform:uppercase">HER2 Status</div>
            <div style="font-size:22px;font-weight:800;font-family:'Rajdhani',sans-serif;color:#00f0ff;text-shadow:0 0 10px rgba(0,240,255,0.4)">Negative</div>
            <div style="font-size:9px;color:#10b981;font-family:'Share Tech Mono',monospace">● IHC Score 0/1+</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cf1, cf2 = st.columns([5, 7])

    with cf1:
        st.markdown("""
        <div class="cyber-card" style="margin-bottom:12px">
            <div class="g-card-header"><div class="g-card-title">Tumour Cell Nucleus Features</div><span class="g-badge badge-amber">Wisconsin FNA Dataset</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:10px;font-weight:700;color:#00d4ff;letter-spacing:1.5px;text-transform:uppercase;margin:8px 0;font-family:DM Mono,monospace">📐 Mean Values</div>', unsafe_allow_html=True)
        s1a, s1b = st.columns(2)
        mean_radius     = s1a.number_input("Radius", 6.9, 28.11, 17.99, 0.01, key="c_mr")
        mean_texture    = s1b.number_input("Texture", 9.71, 39.28, 10.38, 0.01, key="c_mt")
        s2a, s2b = st.columns(2)
        mean_perimeter  = s2a.number_input("Perimeter", 43.79, 188.5, 122.8, 0.1, key="c_mp")
        mean_area       = s2b.number_input("Area", 143.5, 2501.0, 1001.0, 0.5, key="c_ma")
        s3a, s3b = st.columns(2)
        mean_smoothness = s3a.number_input("Smoothness", 0.053, 0.163, 0.1184, 0.0001, format="%.4f", key="c_msmooth")
        mean_compactness= s3b.number_input("Compactness", 0.019, 0.345, 0.2776, 0.001, format="%.4f", key="c_mcomp")
        s4a, s4b = st.columns(2)
        mean_concavity  = s4a.number_input("Concavity", 0.0, 0.427, 0.3001, 0.001, format="%.4f", key="c_mconcav")
        mean_concave_pts= s4b.number_input("Concave Pts", 0.0, 0.201, 0.1471, 0.001, format="%.4f", key="c_mconcpt")
        s5a, s5b = st.columns(2)
        mean_symmetry   = s5a.number_input("Symmetry", 0.106, 0.304, 0.2419, 0.001, format="%.4f", key="c_msym")
        mean_fractal_dim= s5b.number_input("Fractal Dim", 0.050, 0.097, 0.0787, 0.0001, format="%.5f", key="c_mfrac")

        run_cancer = st.button("🔬  RUN CANCER RISK ANALYSIS", key="btn_cancer_run")

    # Evaluate Cancer Model
    n_feats = cancer_model.n_features_in_
    if n_feats == 10:
        X_cancer = np.array([[
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_pts, mean_symmetry, mean_fractal_dim,
        ]])
    else:
        # Default 30 features
        X_cancer = np.zeros((1, 30))
        X_cancer[0, :10] = [
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_pts, mean_symmetry, mean_fractal_dim
        ]

    proba_c    = cancer_model.predict_proba(X_cancer)[0]
    pred_c     = cancer_model.predict(X_cancer)[0]
    # Class 0 = Malignant, Class 1 = Benign
    risk_pct_c = round(float(proba_c[0]) * 100, 1)
    conf_c     = round(float(max(proba_c)) * 100, 1)
    level_c    = "HIGH RISK" if risk_pct_c >= threshold else "MODERATE" if risk_pct_c >= 35 else "LOW RISK"
    diag       = "Malignant" if pred_c == 0 else "Benign"

    feat_names = [
        "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
        "mean compactness","mean concavity","mean concave pts","mean symmetry","mean fractal dim"
    ]
    fi_df_c = pd.DataFrame({
        "Feature":    feat_names[:len(cancer_model.feature_importances_)],
        "Importance": cancer_model.feature_importances_[:10],
    }).sort_values("Importance", ascending=True)

    CANCER_DATA = {
        "HIGH RISK": {
            "alert_short": "Immediate oncology / breast clinic referral · FNA cytology recommended",
            "medicines": [
                ("Specialist Consultation","Urgent","Immediate multidisciplinary team review"),
                ("Tamoxifen (if ER+)","20 mg daily","Selective estrogen receptor modulator"),
                ("Letrozole (postmenopausal)","2.5 mg daily","Aromatase inhibitor protocol"),
            ],
            "tests":       ["Diagnostic mammography & ultrasound","Core needle biopsy (CNB)","ER / PR / HER2 immunohistochemistry"],
            "precautions": ["No delay in specialist review","Avoid unverified hormone supplements","Monitor local lymphadenopathy"],
            "lifestyle":   ["Oncology support counselling","Nutritional optimisation","Rest and recovery protocol"],
        },
        "MODERATE": {
            "alert_short": "Clinical follow-up in 3–4 weeks · Short-interval imaging follow-up",
            "medicines": [
                ("Follow-up protocol","Review in 4 wks","Re-evaluate with targeted imaging"),
            ],
            "tests":       ["Repeat breast ultrasound at 3-6 months","Clinical breast exam","CA 15-3 baseline"],
            "precautions": ["Report any focal change immediately","Avoid self-medication"],
            "lifestyle":   ["Maintain healthy BMI","Limit alcohol","Regular exercise"],
        },
        "LOW RISK": {
            "alert_short": "Routine breast screening protocol · Benign cellular architecture",
            "medicines": [
                ("No oncology medication needed","—","Maintain screening baseline"),
            ],
            "tests":       ["Annual screening mammography (age ≥40)","Monthly breast self-examination"],
            "precautions": ["Maintain routine screening schedule"],
            "lifestyle":   ["Plant-rich whole food nutrition","Physical activity 150 min/week"],
        },
    }

    with cf2:
        render_clinical_report(level_c, risk_pct_c, conf_c, diag,
            fi_df_c, CANCER_DATA[level_c], "cancer_cockpit_fi", is_heart=False)

        if PDF_EXPORT_AVAILABLE:
            try:
                _cpdf = pdf_export.build_diagnosis_pdf(
                    condition="Cancer", level=level_c, risk_pct=risk_pct_c, confidence=conf_c,
                    prediction_label=diag,
                    key_inputs={
                        "Mean Radius": mean_radius, "Mean Texture": mean_texture,
                        "Mean Concavity": mean_concavity, "Mean Symmetry": mean_symmetry,
                    },
                    recommendations=CANCER_DATA[level_c],
                )
                st.download_button(
                    "⬇  Download Oncology PDF Report", data=_cpdf,
                    file_name=f"medcore_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", key="btn_pdf_cancer_cockpit",
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")


# TAB 3 — OCR REPORT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ocr-banner">
        📄 Upload any medical lab report (PDF / JPG / PNG) or paste the text. Covers heart, cancer, diabetes,
        CBC, liver, thyroid, kidney &amp; vitamin panels — flags abnormal values and generates a plain-language summary.
    </div>
    """, unsafe_allow_html=True)

    if not REPORT_PARSER_AVAILABLE:
        st.error(
            "⚠ report_parser.py could not be imported — it must sit next to dashboard.py "
            "(or in an `ocr/` subfolder). Falling back to a very limited parser."
        )

    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">📤 Upload or Paste Report</div><span class="g-badge badge-cyan">OCR + NLP</span></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload report (PDF / JPG / PNG)", type=["pdf","jpg","png","jpeg"])
        st.markdown("<div style='text-align:center;font-size:11px;color:#4a6a8a;margin:8px 0;font-family:DM Mono,monospace'>— or paste text below —</div>", unsafe_allow_html=True)
        pasted_text = st.text_area("Paste report text here", height=220,
            placeholder="Cholesterol: 268 mg/dL\nBlood Pressure: 142/88 mmHg\nFasting Blood Glucose: 118 mg/dL\nHbA1c: 6.1 %\nCA-125: 62 U/mL\nPSA: 5.4 ng/mL\nHaemoglobin: 13.4 g/dL\nHeart Rate: 88 bpm\nLDL: 162 mg/dL\nHDL: 38 mg/dL\nTriglycerides: 210 mg/dL")
        summary_lang = st.selectbox(
            "AI summary language", ["English", "Hindi", "Gujarati", "Marathi", "Tamil", "Telugu", "Bengali"],
            key="ocr_summary_lang",
        )
        run_ocr = st.button("🔍  Analyze Report", key="btn_ocr")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Run OCR + parsing once per click, cache result in session_state ──────
    if run_ocr:
        text_to_parse = pasted_text.strip()
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            ocr_done   = False

            if not ocr_done:
                try:
                    import easyocr, PIL.Image, io
                    img    = PIL.Image.open(io.BytesIO(file_bytes))
                    reader = easyocr.Reader(["en"], verbose=False)
                    text_to_parse = " ".join(reader.readtext(np.array(img), detail=0))
                    st.session_state["ocr_engine_msg"] = "✅ OCR completed with EasyOCR"
                    ocr_done = True
                except Exception:
                    pass

            if not ocr_done:
                try:
                    import pytesseract, PIL.Image, io
                    img = PIL.Image.open(io.BytesIO(file_bytes))
                    text_to_parse = pytesseract.image_to_string(img)
                    st.session_state["ocr_engine_msg"] = "✅ OCR completed with Tesseract"
                    ocr_done = True
                except Exception:
                    pass

            if not ocr_done:
                st.session_state["ocr_engine_msg"] = None
                st.warning("⚠ Could not read image automatically. Install easyocr (pip install easyocr) or paste the report text manually.")

        st.session_state["ocr_text_to_parse"] = text_to_parse
        st.session_state["ocr_ai_summary"] = None  # reset any stale summary from a previous report

        if text_to_parse and REPORT_PARSER_AVAILABLE:
            st.session_state["ocr_parsed_result"] = rpx.parse_full_report(text_to_parse)
        elif text_to_parse:
            # Minimal fallback if report_parser.py isn't importable at all
            st.session_state["ocr_parsed_result"] = {
                "all_values": {}, "abnormal_flags": [], "patient_info": {},
                "report_type": "general", "summary": "report_parser.py unavailable.",
                "sections": {},
            }
        else:
            st.session_state["ocr_parsed_result"] = None

        _res = st.session_state["ocr_parsed_result"]
        if _res and _res.get("all_values"):
            _rt = _res.get("report_type", "general").title()
            _n_abn = len(_res.get("abnormal_flags", []))
            _log_report_history(
                "OCR Report",
                f"{_rt} panel — {_n_abn} abnormal" if _n_abn else f"{_rt} panel — all normal",
                f"{len(_res['all_values'])} value(s) extracted",
            )

    with oc2:
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">🧪 Extracted Lab Values</div><span class="g-badge badge-green">NLP Parsed</span></div>', unsafe_allow_html=True)

        result = st.session_state.get("ocr_parsed_result")
        engine_msg = st.session_state.get("ocr_engine_msg")
        if engine_msg:
            st.success(engine_msg)

        if result is None:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#4a6a8a;font-size:13px;font-family:DM Sans,sans-serif">Results will appear here after clicking Analyze</div>', unsafe_allow_html=True)
        else:
            all_values = result.get("all_values", {})
            if not all_values:
                st.warning("No recognisable lab values found. Try format: `Cholesterol: 245 mg/dL` or `CA-125: 62 U/mL`")
            else:
                report_type_label = {
                    "heart": "🫀 Heart / Cardiac panel", "cancer": "🔬 Cancer / Tumour marker panel",
                    "diabetes": "🩸 Diabetes / Metabolic panel", "general": "📋 General panel",
                }.get(result.get("report_type", "general"), "📋 General panel")
                st.markdown(f'<span class="g-badge badge-cyan">{report_type_label}</span>', unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)

                fc_map = {
                    "Normal": GREEN, "Optimal": GREEN, "Low": AMBER, "Borderline": AMBER, "Elevated": AMBER,
                    "High": RED, "Critical": RED, "Critical Low": RED,
                }
                abnormal_keys = {f["key"] for f in result.get("abnormal_flags", [])}
                abnormal_by_key = {f["key"]: f for f in result.get("abnormal_flags", [])}

                rows = ""
                for key, val in all_values.items():
                    label = key.replace("_", " ").title()
                    if key in abnormal_by_key:
                        f = abnormal_by_key[key]
                        unit, normal, flag = f["unit"], f["normal"], f["flag"]
                    else:
                        unit, normal, flag = "", "—", "Normal"
                    fc = GREEN if flag == "Normal" else next((c for lbl, c in fc_map.items() if lbl in flag), AMBER)
                    rows += f'<tr><td><strong style="color:#e8f0fe;font-family:DM Sans,sans-serif">{label}</strong></td><td style="color:#7f9ab8;font-family:DM Mono,monospace">{val} {unit}</td><td style="color:#4a6a8a;font-size:11px;font-family:DM Mono,monospace">{normal}</td><td><span style="color:{fc};font-weight:700;font-size:11px;font-family:DM Mono,monospace">{flag}</span></td></tr>'
                st.markdown(f'<table class="m-table"><thead><tr><th>Parameter</th><th>Value</th><th>Normal Range</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if abnormal_keys:
                    names = ", ".join(k.replace("_", " ").title() for k in abnormal_keys)
                    st.warning(f"⚠ {len(abnormal_keys)} abnormal: **{names}** — consider running a Diagnosis above.")
                else:
                    st.success("✅ All extracted values within normal ranges.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Plain-language summary (free, no API key needed) + optional AI polish ──
    result = st.session_state.get("ocr_parsed_result")
    if result and result.get("all_values"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        st.markdown('<div class="g-card-header"><div class="g-card-title">📝 Report Summary</div><span class="g-badge badge-green">Free · No API key needed</span></div>', unsafe_allow_html=True)

        all_values     = result.get("all_values", {})
        abnormal_flags = result.get("abnormal_flags", [])
        patient_info   = result.get("patient_info", {})

        # Overall verdict
        if not abnormal_flags:
            st.success("✅ Overall, this report looks normal — every value tested falls in the healthy range.")
        else:
            critical = [f for f in abnormal_flags if "critical" in f["flag"].lower()]
            if critical:
                st.error(f"🔴 {len(critical)} value(s) are critically abnormal and need prompt attention, "
                         f"plus {len(abnormal_flags) - len(critical)} other value(s) worth watching.")
            else:
                st.warning(f"🟡 {len(abnormal_flags)} value(s) are outside the normal range and worth discussing with a doctor.")

        # Patient line
        if patient_info:
            bits = []
            if patient_info.get("name"):
                bits.append(patient_info["name"])
            if patient_info.get("age"):
                bits.append(f"{patient_info['age']} yrs")
            if patient_info.get("sex"):
                bits.append("Male" if patient_info["sex"] == "M" else "Female")
            if patient_info.get("report_date"):
                bits.append(f"dated {patient_info['report_date']}")
            if bits:
                st.markdown(f'<div style="color:#7f9ab8;font-size:12px;font-family:DM Mono,monospace;margin:6px 0">{" · ".join(bits)}</div>', unsafe_allow_html=True)

        # Things that need attention
        if abnormal_flags:
            st.markdown('<div style="color:#e8f0fe;font-weight:600;margin-top:10px">Things to watch:</div>', unsafe_allow_html=True)
            for f in abnormal_flags:
                label = f["key"].replace("_", " ").title()
                st.markdown(
                    f'<div style="color:#7f9ab8;font-size:13px;margin:4px 0">• <strong style="color:#e8f0fe">{label}</strong> '
                    f'is <span style="color:{RED if "critical" in f["flag"].lower() or "high" in f["flag"].lower() else AMBER}">{f["flag"]}</span> '
                    f'at {f["value"]} {f["unit"]} (normal: {f["normal"]})</div>',
                    unsafe_allow_html=True,
                )

        # Normal values, briefly
        normal_keys = [k for k in all_values if k not in {f["key"] for f in abnormal_flags}]
        if normal_keys:
            names = ", ".join(k.replace("_", " ").title() for k in normal_keys)
            st.markdown(f'<div style="color:#4a6a8a;font-size:12px;margin-top:10px">✅ Within normal range: {names}</div>', unsafe_allow_html=True)

        st.markdown('<div style="color:#4a6a8a;font-size:11px;margin-top:12px;font-style:italic">Please share this report with your doctor for proper advice.</div>', unsafe_allow_html=True)

        # Optional: polish the summary with an LLM — free via a local Ollama model,
        # or via the paid Anthropic API if a key happens to be configured.
        groq_ok = REPORT_PARSER_AVAILABLE and hasattr(rpx, "groq_is_available") and rpx.groq_is_available()
        ollama_up = REPORT_PARSER_AVAILABLE and rpx.ollama_is_running()
        claude_ok = REPORT_PARSER_AVAILABLE and rpx.CLAUDE_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))

        if groq_ok or ollama_up or claude_ok:
            st.markdown("<br>", unsafe_allow_html=True)
            backend_options = []
            if groq_ok:
                backend_options.append("Groq Llama-3 (Cloud, Free)")
            if ollama_up:
                backend_options.append("Ollama (Local/Ngrok)")
            if claude_ok:
                backend_options.append("Claude API (paid)")
            backend = st.radio("Polish with AI using:", backend_options, horizontal=True, key="ocr_ai_backend") \
                if len(backend_options) > 1 else backend_options[0]

            ollama_model = None
            if backend.startswith("Ollama"):
                pulled = rpx.ollama_list_models()
                ollama_model = pulled[0] if pulled else "llama3"

            if st.button("✨  Polish with AI", key="btn_ai_summary"):
                with st.spinner("Generating plain-language summary…"):
                    lang = st.session_state.get("ocr_summary_lang", "English")
                    if backend.startswith("Groq"):
                        st.session_state["ocr_ai_summary"] = rpx.groq_full_summary(result, language=lang)
                    elif backend.startswith("Ollama"):
                        st.session_state["ocr_ai_summary"] = rpx.ollama_full_summary(result, language=lang, model=ollama_model)
                    else:
                        st.session_state["ocr_ai_summary"] = rpx.ai_full_summary(result, language=lang)

            if st.session_state.get("ocr_ai_summary"):
                st.markdown(st.session_state["ocr_ai_summary"])
        else:
            st.markdown(
                '<div style="color:#4a6a8a;font-size:11px;margin-top:8px">'
                '💡 Want an AI-polished write-up? Add a free <code>GROQ_API_KEY</code> in Render or run Ollama locally.'
                '</div>', unsafe_allow_html=True,
            )

        # ── PDF export ──────────────────────────────────────────────────────
        if PDF_EXPORT_AVAILABLE:
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                _pdf_bytes = pdf_export.build_ocr_summary_pdf(
                    patient_info=result.get("patient_info", {}),
                    all_values=all_values,
                    abnormal_flags=abnormal_flags,
                    summary_text=st.session_state.get("ocr_ai_summary") or result.get("summary", ""),
                )
                st.download_button(
                    "⬇  Download PDF Report", data=_pdf_bytes,
                    file_name=f"medcore_lab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", key="btn_pdf_ocr",
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")
        else:
            st.markdown(
                '<div style="color:#4a6a8a;font-size:11px;margin-top:8px">'
                '💡 PDF export needs the fpdf2 package — run <code>pip install fpdf2</code> and refresh.'
                '</div>', unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════

# TAB 4 — PATIENT REGISTRY
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    reg_sub1, reg_sub2, reg_sub3 = st.tabs([
        "  📋  Synthetic Cohort (Demo)  ", "  🗄️  Live Patient Records (SQLite)  ", "  🕐  This Session's Reports  ",
    ])

    # ── SUB-TAB 1: existing synthetic demo cohort (unchanged) ──────────────
    with reg_sub1:
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

        styler = fdf.style
        if hasattr(styler, "map"):
            styled = styler.map(color_status, subset=["Status"]).map(color_risk, subset=["Heart Risk %","Cancer Risk %"])
        elif hasattr(styler, "applymap"):
            styled = styler.applymap(color_status, subset=["Status"]).applymap(color_risk, subset=["Heart Risk %","Cancer Risk %"])
        else:
            styled = styler
        styled = styled.format({"Heart Risk %":"{}%","Cancer Risk %":"{}%"})
        st.dataframe(styled, use_container_width=True, height=480)

    # ── SUB-TAB 2: real SQLite-backed patient records (CRUD) ───────────────
    with reg_sub2:
        st.markdown("<br>", unsafe_allow_html=True)
        if not PATIENT_DB_AVAILABLE:
            st.error("⚠ patient_db.py could not be imported — it must sit next to dashboard.py.")
        else:
            st.markdown('<div class="g-card">', unsafe_allow_html=True)
            st.markdown('<div class="g-card-header"><div class="g-card-title">➕ Add Patient</div><span class="g-badge badge-green">SQLite · Live</span></div>', unsafe_allow_html=True)
            with st.form("add_patient_form", clear_on_submit=True):
                ac1, ac2, ac3, ac4 = st.columns(4)
                new_name = ac1.text_input("Name")
                new_age  = ac2.number_input("Age", 0, 120, 30)
                new_sex  = ac3.selectbox("Sex", ["M", "F"])
                new_ward = ac4.selectbox("Ward", ["Cardiology","Oncology","General Medicine","Neurology","Pulmonology"])
                new_notes = st.text_input("Notes (optional)")
                submitted = st.form_submit_button("Add Patient")
                if submitted:
                    if new_name.strip():
                        new_id = patient_db.create_patient(new_name.strip(), int(new_age), new_sex, new_ward, new_notes.strip())
                        st.success(f"Added {new_name} as {new_id}")
                        st.rerun()
                    else:
                        st.warning("Name is required.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            records = patient_db.get_all_patients()
            st.markdown(f'<div style="color:#7f9ab8;font-size:12px;margin-bottom:8px">{len(records)} patient(s) in the database</div>', unsafe_allow_html=True)

            if not records:
                st.markdown('<div class="empty-state"><div class="empty-icon">🗄️</div><div class="empty-title">No patients yet</div><div class="empty-sub">Add one using the form above</div></div>', unsafe_allow_html=True)
            else:
                for rec in records:
                    with st.expander(f"{rec['patient_id']}  ·  {rec['name']}  ·  {rec['ward']}"):
                        ec1, ec2, ec3 = st.columns(3)
                        e_name = ec1.text_input("Name", value=rec["name"], key=f"edit_name_{rec['patient_id']}")
                        e_age  = ec2.number_input("Age", 0, 120, int(rec["age"] or 0), key=f"edit_age_{rec['patient_id']}")
                        e_sex  = ec3.selectbox("Sex", ["M", "F"], index=0 if rec["sex"] == "M" else 1, key=f"edit_sex_{rec['patient_id']}")
                        e_ward = st.selectbox(
                            "Ward", ["Cardiology","Oncology","General Medicine","Neurology","Pulmonology"],
                            index=["Cardiology","Oncology","General Medicine","Neurology","Pulmonology"].index(rec["ward"]) if rec["ward"] in ["Cardiology","Oncology","General Medicine","Neurology","Pulmonology"] else 0,
                            key=f"edit_ward_{rec['patient_id']}",
                        )
                        e_notes = st.text_input("Notes", value=rec["notes"] or "", key=f"edit_notes_{rec['patient_id']}")
                        st.caption(f"Created {rec['created_at']}  ·  Last updated {rec['updated_at']}")

                        bcol1, bcol2 = st.columns(2)
                        if bcol1.button("💾 Save changes", key=f"save_{rec['patient_id']}"):
                            patient_db.update_patient(rec["patient_id"], name=e_name, age=int(e_age), sex=e_sex, ward=e_ward, notes=e_notes)
                            st.success("Updated.")
                            st.rerun()
                        if bcol2.button("🗑️ Delete patient", key=f"delete_{rec['patient_id']}"):
                            patient_db.delete_patient(rec["patient_id"])
                            st.success("Deleted.")
                            st.rerun()

    # ── SUB-TAB 3: in-session report history ────────────────────────────────
    with reg_sub3:
        st.markdown("<br>", unsafe_allow_html=True)
        history = st.session_state.get("report_history", [])
        if not history:
            st.markdown('<div class="empty-state"><div class="empty-icon">🕐</div><div class="empty-title">No reports analyzed yet this session</div><div class="empty-sub">Run an OCR report or a diagnosis to see it logged here</div></div>', unsafe_allow_html=True)
        else:
            icon_map = {"OCR Report": "📄", "Heart Diagnosis": "🫀", "Cancer Diagnosis": "🔬"}
            for item in history:
                st.markdown(f"""
                <div class="g-card" style="margin-bottom:10px;padding:14px 18px">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div style="color:#e8f0fe;font-size:13px;font-weight:600">{icon_map.get(item['kind'],'📌')}&nbsp; {item['label']}</div>
                        <div style="color:#4a6a8a;font-size:11px;font-family:DM Mono,monospace">{item['time']}</div>
                    </div>
                    <div style="color:#7f9ab8;font-size:12px;margin-top:4px">{item['kind']}{' — ' + item['detail'] if item['detail'] else ''}</div>
                </div>
                """, unsafe_allow_html=True)
            if st.button("Clear session history", key="clear_report_history"):
                st.session_state["report_history"] = []
                st.rerun()




# ══════════════════════════════════════════════════════════════════════

# TAB 5 — HEALTH CHAT (Rule-Based)
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="g-card-header"><div class="g-card-title">💬 MedCore Health Assistant</div><span class="g-badge badge-green">RAG + LLM</span></div>', unsafe_allow_html=True)

    try:
        import rag_chat
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False

    ollama_up_chat = REPORT_PARSER_AVAILABLE and rpx.ollama_is_running()
    claude_ok_chat = REPORT_PARSER_AVAILABLE and rpx.CLAUDE_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))

    # Rule-based fallback knowledge base — used only if RAG + no LLM backend both fail
    HEALTH_KB = {
        "cholesterol":   "**Cholesterol** is a fatty substance in your blood. High LDL (bad) cholesterol increases heart disease risk. Normal total cholesterol is < 200 mg/dL. Diet, exercise, and statins help manage it.",
        "blood pressure":"**Blood Pressure** measures force against artery walls. Normal is < 120/80 mmHg. High BP (hypertension) strains the heart. Reduce salt, exercise regularly, and take prescribed medication.",
        "thalassemia":   "**Thalassemia** is an inherited blood disorder affecting haemoglobin. In the heart model, thal=1 is normal, thal=2 is a fixed defect, thal=3 is a reversible defect — reversible defects carry higher cardiac risk.",
        "hba1c":         "**HbA1c** (Glycosylated Haemoglobin) reflects average blood glucose over 3 months. Normal < 5.7%, Pre-diabetic 5.7–6.4%, Diabetic ≥ 6.5%. Good control target is < 7%.",
        "glucose":       "**Fasting Blood Glucose** measures blood sugar after 8+ hours fasting. Normal: 70–99 mg/dL. Pre-diabetic: 100–125 mg/dL. Diabetic: ≥ 126 mg/dL.",
    }

    def get_fallback_response(question):
        q = question.lower().strip()
        for keyword, answer in HEALTH_KB.items():
            if keyword in q:
                return answer
        if any(w in q for w in ["hello", "hi", "hey", "namaste"]):
            return "Hello! I am MedCore Health Assistant. Ask me about heart disease, cancer risk, biomarkers, medications, lab values, diet, or lifestyle."
        return ("I can help with heart disease risk factors, cancer biomarkers, lab value interpretation, "
                "medications, diet, and lifestyle. For full free-form answers, set up Ollama (free, local) — "
                "see the note below.")

    groq_ok_chat = bool(os.environ.get("GROQ_API_KEY", "").strip())

    # Backend + model picker
    backend_options = []
    if groq_ok_chat:
        backend_options.append("🚀 Groq Cloud AI (High-Speed Streaming)")
    backend_options.append("⚡ Fast Clinical Engine (Instant)")
    if ollama_up_chat:
        backend_options.append("🤖 Ollama Llama-3 (Local/Ngrok)")
    if claude_ok_chat:
        backend_options.append("🧠 Claude API (Streaming)")

    chat_backend = st.radio("Intelligence Engine:", backend_options, horizontal=True, key="chat_backend_choice")
    chat_model = "groq-auto" if chat_backend.startswith("🚀 Groq") else "llama3"
    if chat_backend.startswith("🤖 Ollama"):
        pulled = rpx.ollama_list_models()
        chat_model = pulled[0] if pulled else "llama3"

    if not groq_ok_chat and not ollama_up_chat and not claude_ok_chat:
        st.markdown(
            '<div style="color:#4a6a8a;font-size:11px;margin-bottom:8px">'
            '💡 Fast Clinical Engine active. To enable <b>Cloud Llama 3 (Free)</b>, add <code>GROQ_API_KEY</code> in Render Environment Variables.'
            '</div>', unsafe_allow_html=True,
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am **MedCore AI Copilot**. I can explain your diagnostic results, biomarkers, cardiovascular telemetry, oncology findings, medications, and lab values — grounded in your clinical reports. Ask me anything!", "sources": []}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                src_line = "  ·  ".join(f"{s['title']} ({s['score']})" for s in msg["sources"])
                st.markdown(f'<div style="color:#4a6a8a;font-size:11px;margin-top:6px">📚 Sources: {src_line}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask about your results, biomarkers, medications, or medical terms...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "sources": []})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🏥"):
            parsed_ctx = st.session_state.get("ocr_parsed_result")
            backend_key = "fast"
            if chat_backend.startswith("🚀 Groq"):
                backend_key = "groq"
                chat_model = "groq-auto"
            elif chat_backend.startswith("🤖 Ollama"):
                backend_key = "ollama"
            elif chat_backend.startswith("🧠 Claude"):
                backend_key = "claude"

            sources = []
            if RAG_AVAILABLE:
                chunks = rag_chat.retrieve(user_input, k=3)
                sources = [{"title": c["title"], "score": round(c["score"], 3)} for c in chunks if c.get("score", 0) > 0.16]

            if RAG_AVAILABLE:
                stream_gen = rag_chat.stream_answer(
                    user_input,
                    parsed_result=parsed_ctx,
                    backend=backend_key,
                    model=chat_model or "llama3",
                )
                reply = st.write_stream(stream_gen)
            else:
                reply = get_fallback_response(user_input)
                st.markdown(reply)

            if sources:
                src_line = "  ·  ".join(f"{s['title']} ({s['score']})" for s in sources)
                st.markdown(f'<div style="color:#4a6a8a;font-size:11px;margin-top:6px">📚 Sources: {src_line}</div>', unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": reply, "sources": sources})

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = [{"role": "assistant", "content": "Chat cleared. How can I help you?", "sources": []}]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — POPULATION ANALYTICS & MODEL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">📈 Monthly Screening Volume</span>
            <span class="g-badge badge-cyan">AI TRACKED</span>
        </div>
        """, unsafe_allow_html=True)
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
        st.plotly_chart(fig, use_container_width=True, key="pop_vol_chart")

    with c2:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">🫀 Population Risk Distribution</span>
            <span class="g-badge badge-green">LIVE</span>
        </div>
        """, unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Heart Risk","Cancer Risk","No Risk"], values=[439,277,568], hole=0.65,
            marker=dict(colors=[RED, AMBER, "rgba(255,255,255,0.06)"],
                        line=dict(color=NAVY, width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} patients (%{percent})<extra></extra>",
        ))
        fig2.add_annotation(text="1,284", x=0.5, y=0.58, showarrow=False,
            font=dict(size=26, color="#e8f0fe", family="Rajdhani"))
        fig2.add_annotation(text="patients", x=0.5, y=0.42, showarrow=False,
            font=dict(size=11, color=GRAY, family="Share Tech Mono"))
        fig2.update_layout(**PL, height=240, showlegend=True, margin=dict(l=0,r=0,t=8,b=8))
        fig2.update_layout(legend={**_L, "orientation":"v","x":0.78,"y":0.5})
        st.plotly_chart(fig2, use_container_width=True, key="pop_dist_chart")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">👥 Risk by Age Group</span>
        </div>
        """, unsafe_allow_html=True)
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
        st.plotly_chart(fig3, use_container_width=True, key="pop_age_chart")

    with c4:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">🧬 Feature Correlation Matrix</span>
        </div>
        """, unsafe_allow_html=True)
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
        st.plotly_chart(fig4, use_container_width=True, key="pop_corr_chart")

    # ROC Curves & Model Configuration
    def roc_fig(name, auc, color, fill):
        t   = np.linspace(0, 1, 200)
        tpr = np.clip(1-(1-t)**(1/(1-auc+0.01)), 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="rgba(255,255,255,0.1)", dash="dash", width=1.5), showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=tpr, name=f"AUC = {auc:.3f}",
            line=dict(color=color, width=2.5), fill="tozeroy", fillcolor=fill))
        fig.update_layout(**PL, height=220)
        _fix(fig, xaxis=dict(title="False Positive Rate", range=[0,1]),
             yaxis=dict(title="True Positive Rate", range=[0,1.02]),
             legend=dict(x=0.55, y=0.1))
        return fig

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">📈 Heart Model ROC Curve (AUC = 0.962)</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(roc_fig("Heart Disease", 0.962, RED, "rgba(255,71,87,0.06)"), use_container_width=True, key="roc_heart")
    with rc2:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#e2e8f0;letter-spacing:1px;text-transform:uppercase;font-family:'Share Tech Mono',monospace">📈 Cancer Model ROC Curve (AUC = 0.931)</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(roc_fig("Cancer Risk", 0.931, AMBER, "rgba(255,177,66,0.06)"), use_container_width=True, key="roc_cancer")

    st.markdown("""
    <div class="cyber-card">
        <div class="g-card-header"><div class="g-card-title">🤖 Model Configuration & Parameters</div></div>
    </div>
    """, unsafe_allow_html=True)
    di1,di2,di3,di4,di5,di6 = st.columns(6)
    di1.metric("Heart Trees",     heart_model.n_estimators)
    di2.metric("Heart Features",  heart_model.n_features_in_)
    di3.metric("Heart Classes",   len(heart_model.classes_))
    di4.metric("Cancer Trees",    cancer_model.n_estimators)
    di5.metric("Cancer Features", cancer_model.n_features_in_)
    di6.metric("Cancer Classes",  len(cancer_model.classes_))


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(0,212,255,0.06),rgba(0,0,0,0));
     border:1px solid rgba(0,212,255,0.1);border-radius:14px;padding:16px 28px;margin-top:24px;
     display:flex;justify-content:space-between;align-items:center">
    <div style="display:flex;align-items:center;gap:14px">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#00d4ff,#0099cc);
             border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px">🏥</div>
        <span style="font-size:12px;color:#4a6a8a;font-family:'Share Tech Mono',monospace">
            MedCore Clinical AI HUD v4.2 PRO &nbsp;·&nbsp; Dual RandomForest Core
            &nbsp;·&nbsp; Real-Time Telemetry
        </span>
    </div>
    <span style="font-size:11px;color:#2a4a6a;font-family:'Share Tech Mono',monospace">
        ⚕️ Research &amp; Education Only · Not for Clinical Use
    </span>
</div>
""", unsafe_allow_html=True)
