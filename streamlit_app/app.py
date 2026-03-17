import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
.stApp { background: #07080f; color: #d0d8e8; }
#MainMenu, footer, header { visibility: hidden; }

/* Remove ALL default Streamlit spacing */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] { display: none !important; }

/* Remove gaps between all vertical blocks */
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; margin: 0 !important; }

/* Remove all default element padding */
.element-container { margin: 0 !important; padding: 0 !important; }
.stMarkdown { margin: 0 !important; padding: 0 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1e2438; border-radius: 4px; }

/* ════════ TOP BAR ════════ */
.topbar {
    background: rgba(7,8,15,0.98);
    backdrop-filter: blur(24px);
    border-bottom: 1px solid #0f1420;
    padding: 0 3rem;
    display: flex; align-items: center; justify-content: space-between;
    height: 58px;
    position: sticky; top: 0; z-index: 1000;
    width: 100%;
}
.tb-brand { display:flex; align-items:center; gap:10px; }
.tb-logo {
    width:34px; height:34px;
    background:linear-gradient(135deg,#c9a227,#e8c547);
    border-radius:9px; display:flex; align-items:center; justify-content:center;
    font-size:16px; box-shadow:0 0 24px rgba(201,162,39,0.35);
}
.tb-name { font-size:0.92rem; font-weight:700; color:#f0e6c0; letter-spacing:-0.01em; line-height:1.2; }
.tb-ver  { font-size:0.58rem; color:#2d3650; font-family:'JetBrains Mono',monospace; }
.tb-stats { display:flex; align-items:stretch; }
.tb-stat {
    padding:0 1.5rem; border-left:1px solid #0f1420;
    display:flex; flex-direction:column; justify-content:center; align-items:flex-end;
}
.tb-stat:first-child { border-left:none; }
.tb-v { font-size:0.88rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1.2; }
.tb-l { font-size:0.52rem; color:#2d3650; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px; }

/* ════════ HERO ════════ */
.hero-banner {
    width: 100%;
    background:
        radial-gradient(ellipse 60% 80% at 0% 50%, rgba(201,162,39,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 100% 30%, rgba(139,92,246,0.03) 0%, transparent 60%),
        linear-gradient(180deg, #0c0f1c 0%, #0a0c18 50%, #07080f 100%);
    border-bottom: 1px solid #0f1420;
    padding: 4.5rem 3rem 4rem;
    position: relative; overflow: hidden;
    margin-top: 0 !important;
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(201,162,39,0.7) 25%, rgba(201,162,39,0.3) 60%, transparent 100%);
}

/* Override Streamlit column padding inside hero */
.hero-banner [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    align-items: center !important;
}
.hero-banner [data-testid="column"] {
    padding: 0 !important;
    margin: 0 !important;
}
.hero-banner [data-testid="stVerticalBlock"] {
    gap: 0.875rem !important;
    padding: 0 !important;
    margin: 0 !important;
}
.hero-banner .element-container {
    margin: 0 !important;
    padding: 0 !important;
}

.hero-tag {
    display:inline-flex; align-items:center; gap:7px;
    background:rgba(201,162,39,0.08); border:1px solid rgba(201,162,39,0.22);
    color:#c9a227; padding:0.25rem 0.9rem; border-radius:999px;
    font-size:0.62rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
    margin-bottom:1.5rem;
}
.ldot {
    width:6px; height:6px; background:#22c55e; border-radius:50%;
    box-shadow:0 0 10px rgba(34,197,94,0.7); animation:lp 2s ease-in-out infinite;
}
@keyframes lp { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.55)} }
.hero-h1 {
    font-size:3.75rem; font-weight:800; line-height:1.04;
    letter-spacing:-0.045em; margin:0 0 1.35rem; color:#f0e6c0;
}
.hero-h1 em { font-style:normal; color:#c9a227; }
.hero-p1 { font-size:0.975rem; color:#9aa4bc; line-height:1.78; margin:0 0 0.8rem; }
.hero-p2 { font-size:0.82rem; color:#505c7a; line-height:1.85; margin:0 0 2.25rem; }
.hero-chips { display:flex; gap:0.5rem; flex-wrap:wrap; }
.hchip {
    background:rgba(255,255,255,0.025); border:1px solid #1a2035; color:#505c7a;
    padding:0.22rem 0.78rem; border-radius:6px; font-size:0.65rem; font-weight:500;
    font-family:'JetBrains Mono',monospace;
}

/* Hero KPI cards */
.hkpi {
    background:rgba(255,255,255,0.02); border:1px solid #0f1420;
    border-radius:14px; padding:1.35rem 1.35rem 1.1rem;
    position:relative; overflow:hidden;
    transition:border-color 0.25s, transform 0.25s;
}
.hkpi:hover { transform:translateY(-3px); }
.hkpi-line { position:absolute; top:0; left:0; right:0; height:2px; border-radius:14px 14px 0 0; }
.hkpi-lbl { font-size:0.58rem; font-weight:700; color:#2d3650; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:0.55rem; }
.hkpi-val { font-size:1.85rem; font-weight:800; font-family:'JetBrains Mono',monospace; letter-spacing:-0.03em; line-height:1; }
.hkpi-sub { font-size:0.62rem; color:#2d3650; margin-top:0.4rem; }

/* ════════ TABS ════════ */
.stTabs [data-baseweb="tab-list"] {
    background:transparent !important; border-bottom:1px solid #0f1420 !important;
    gap:0 !important; padding:0 !important; justify-content:center !important;
    margin-bottom:0 !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important; border:none !important;
    border-bottom:2px solid transparent !important; color:#2d3650 !important;
    font-size:0.7rem !important; font-weight:700 !important;
    padding:0.85rem 2.25rem 0.95rem !important; border-radius:0 !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    transition:color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#6070a0 !important; }
.stTabs [aria-selected="true"] { color:#c9a227 !important; border-bottom:2px solid #c9a227 !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 2.5rem 3rem 0 !important; }

/* ════════ INPUTS ════════ */
.stNumberInput input, input[type="number"] {
    background:#0c0f1c !important; border:1px solid #1a2035 !important;
    color:#f0e6c0 !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.9rem !important; font-weight:500 !important;
}
.stNumberInput input:focus { border-color:#c9a227 !important; box-shadow:0 0 0 3px rgba(201,162,39,0.12) !important; outline:none !important; }
.stNumberInput [data-testid="stNumberInputContainer"] { background:#0c0f1c !important; border:1px solid #1a2035 !important; border-radius:8px !important; }
.stNumberInput button { background:#1a2035 !important; border:none !important; color:#4a5878 !important; }
.stNumberInput button:hover { background:#252e48 !important; color:#9aa4bc !important; }
.stNumberInput label { color:#9aa4bc !important; font-size:0.72rem !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
.stSlider > div > div > div > div { background:#c9a227 !important; }
.stSlider > div > div > div { background:#1a2035 !important; }
.stSlider label { color:#9aa4bc !important; font-size:0.72rem !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.05em !important; }
.stSlider > div > div p { color:#c9a227 !important; font-size:0.75rem !important; font-family:'JetBrains Mono',monospace !important; font-weight:600 !important; }

/* ════════ BUTTON ════════ */
.stButton > button {
    background:linear-gradient(135deg,#9a7215,#c9a227,#e8c547) !important;
    color:#07080f !important; border:none !important; border-radius:9px !important;
    font-weight:700 !important; font-size:0.82rem !important; padding:0.7rem 2rem !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    box-shadow:0 4px 24px rgba(201,162,39,0.22) !important; transition:all 0.22s !important;
}
.stButton > button:hover { box-shadow:0 8px 32px rgba(201,162,39,0.38) !important; transform:translateY(-2px) !important; }

/* ════════ METRICS ════════ */
div[data-testid="stMetric"] { background:#0c0f1c !important; border:1px solid #0f1420 !important; border-radius:11px !important; padding:1.1rem 1.35rem !important; }
div[data-testid="stMetric"] label { color:#505c7a !important; font-size:0.65rem !important; text-transform:uppercase !important; letter-spacing:0.1em !important; font-weight:700 !important; }
div[data-testid="stMetricValue"] { color:#f0e6c0 !important; font-size:1.45rem !important; font-weight:700 !important; font-family:'JetBrains Mono',monospace !important; }

/* ════════ TYPOGRAPHY ════════ */
h1,h2,h3,h4,h5 { color:#f0e6c0 !important; letter-spacing:-0.02em !important; }
.stMarkdown p  { color:#b0bcd0 !important; font-size:0.875rem !important; line-height:1.8 !important; }
.stMarkdown li { color:#b0bcd0 !important; font-size:0.875rem !important; line-height:1.95 !important; }
.stMarkdown h3 { color:#e8dfc0 !important; font-size:1.05rem !important; font-weight:700 !important; margin:1.75rem 0 0.875rem !important; }
.stMarkdown strong { color:#c9a227 !important; font-weight:700 !important; }
.stMarkdown code { background:#0c0f1c !important; color:#c9a227 !important; border-radius:5px !important; padding:2px 7px !important; font-size:0.8rem !important; border:1px solid #1a2035 !important; font-family:'JetBrains Mono',monospace !important; }
.stMarkdown table { background:#0c0f1c !important; border-collapse:collapse !important; width:100% !important; border-radius:10px !important; overflow:hidden !important; margin:0.875rem 0 !important; }
.stMarkdown th { background:#111525 !important; color:#6070a0 !important; font-size:0.63rem !important; text-transform:uppercase !important; letter-spacing:0.1em !important; padding:0.8rem 1.1rem !important; border-bottom:1px solid #1a2035 !important; font-weight:700 !important; }
.stMarkdown td { color:#b0bcd0 !important; padding:0.7rem 1.1rem !important; border-bottom:1px solid #0a0c14 !important; font-size:0.84rem !important; }
.stMarkdown tr:last-child td { border-bottom:none !important; }
.stMarkdown tr:hover td { background:rgba(201,162,39,0.025) !important; }
hr { border-color:#0f1420 !important; margin:1.75rem 0 !important; }

/* ════════ COMPONENT CLASSES ════════ */
.sec-lbl {
    font-size:0.6rem; font-weight:800; color:#2d3650;
    text-transform:uppercase; letter-spacing:0.2em;
    margin:2.75rem 0 1.1rem;
    display:flex; align-items:center; gap:0.875rem;
}
.sec-lbl::after { content:''; flex:1; height:1px; background:#0f1420; }

.pstep {
    background:#0c0f1c; border:1px solid #0a0c14;
    border-radius:11px; padding:1rem 1.25rem;
    display:flex; gap:1rem; margin-bottom:0.45rem;
    transition:border-color 0.22s, background 0.22s;
}
.pstep:hover { border-color:rgba(201,162,39,0.15); background:#0e1220; }
.pstep-num { font-size:0.56rem; font-weight:800; color:#9a7215; font-family:'JetBrains Mono',monospace; background:rgba(201,162,39,0.08); border:1px solid rgba(201,162,39,0.15); border-radius:5px; padding:3px 8px; white-space:nowrap; margin-top:2px; line-height:1.55; flex-shrink:0; }
.pstep-title { font-size:0.82rem; font-weight:600; color:#b0bcd0; margin-bottom:3px; }
.pstep-desc  { font-size:0.7rem; color:#505c7a; line-height:1.6; }

.smote-before { background:#0c0f1c; border:1px solid #0f1420; border-radius:12px; padding:1.5rem; }
.smote-after  { background:#0c0f1c; border:1px solid rgba(201,162,39,0.22); border-radius:12px; padding:1.5rem; }
.smote-title  { font-size:0.58rem; font-weight:800; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:1.25rem; }
.smote-row    { display:flex; justify-content:space-between; align-items:center; padding:0.55rem 0; border-bottom:1px solid #0f1420; }
.smote-row:last-of-type { border-bottom:none; }
.smote-class  { font-size:0.84rem; font-weight:600; }
.smote-count  { font-size:0.92rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.smote-bar    { height:6px; border-radius:4px; margin:0.4rem 0 0.9rem; }
.smote-stat   { margin-top:1.1rem; background:#111525; border-radius:9px; padding:0.875rem; }
.smote-stat-g { margin-top:1.1rem; background:rgba(201,162,39,0.06); border:1px solid rgba(201,162,39,0.15); border-radius:9px; padding:0.875rem; }
.smote-lbl    { font-size:0.62rem; color:#505c7a; margin-bottom:0.3rem; }
.smote-pct    { font-size:1.25rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1; }
.smote-note2  { font-size:0.62rem; margin-top:0.3rem; }
.smote-note   { background:rgba(201,162,39,0.04); border:1px solid rgba(201,162,39,0.1); border-left:2px solid #c9a227; border-radius:0 9px 9px 0; padding:0.9rem 1.25rem; margin-top:0.875rem; }
.smote-nt     { font-size:0.76rem; font-weight:700; color:#c9a227; margin-bottom:5px; }
.smote-nb     { font-size:0.76rem; color:#6070a0; line-height:1.7; }
.smote-nb b   { color:#ef4444 !important; font-weight:700; }

.pg-eye { font-size:0.6rem; font-weight:800; color:#9a7215; text-transform:uppercase; letter-spacing:0.2em; margin-bottom:0.4rem; }
.pg-h1  { font-size:2.35rem; font-weight:800; color:#f0e6c0; letter-spacing:-0.035em; margin:0 0 0.45rem; line-height:1.1; }
.pg-sub { font-size:0.9rem; color:#6070a0; line-height:1.7; margin:0 0 2rem; }

.mbdg { display:inline-flex; align-items:center; gap:9px; background:#0c0f1c; border:1px solid #0f1420; border-radius:8px; padding:0.38rem 1rem; margin-bottom:1.75rem; }
.mbdg-dot  { width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow:0 0 8px rgba(34,197,94,0.55); }
.mbdg-lbl  { font-size:0.7rem; color:#404c68; }
.mbdg-name { font-size:0.76rem; color:#c8d0e0; font-weight:700; font-family:'JetBrains Mono',monospace; }

.inp-sec { font-size:0.62rem; font-weight:800; color:#404c68; text-transform:uppercase; letter-spacing:0.15em; margin:1.35rem 0 0.8rem; }

.risk-box   { background:#0c0f1c; border:1px solid #0f1420; border-radius:11px; padding:1.1rem; }
.risk-box-t { font-size:0.58rem; font-weight:800; color:#2d3650; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.9rem; }
.risk-row   { display:flex; justify-content:space-between; align-items:center; padding:0.42rem 0; border-bottom:1px solid rgba(255,255,255,0.02); }
.risk-row:last-child { border-bottom:none; }
.risk-key   { color:#505c7a; font-family:'JetBrains Mono',monospace; font-size:0.72rem; }
.r-hi  { color:#ef4444; font-weight:700; font-size:0.76rem; }
.r-med { color:#f59e0b; font-weight:700; font-size:0.76rem; }
.r-lo  { color:#22c55e; font-weight:700; font-size:0.76rem; }

.res-card  { border-radius:15px; padding:2.25rem 1.75rem; text-align:center; margin-bottom:1rem; }
.res-fraud { background:linear-gradient(160deg,rgba(127,29,29,0.55),rgba(69,10,10,0.75)); border:1px solid rgba(220,38,38,0.25); }
.res-safe  { background:linear-gradient(160deg,rgba(20,83,45,0.55),rgba(5,46,22,0.75));  border:1px solid rgba(34,197,94,0.25); }
.res-eye   { font-size:0.6rem; font-weight:800; text-transform:uppercase; letter-spacing:0.18em; margin-bottom:0.6rem; }
.res-title { font-size:2.35rem; font-weight:800; letter-spacing:-0.035em; margin:0 0 0.45rem; line-height:1; }
.res-sub   { font-size:0.82rem; color:#505c7a; }

.ds-card { background:#0c0f1c; border:1px solid #0a0c14; border-radius:13px; padding:1.35rem; position:relative; overflow:hidden; }
.ds-line  { position:absolute; top:0; left:0; right:0; height:2px; border-radius:13px 13px 0 0; }
.ds-lbl   { font-size:0.58rem; font-weight:800; color:#2d3650; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:0.55rem; }
.ds-val   { font-size:1.7rem; font-weight:800; font-family:'JetBrains Mono',monospace; letter-spacing:-0.02em; }

.dt-wrap { background:#0c0f1c; border:1px solid #1a2035; border-radius:13px; overflow:hidden; margin-top:0.5rem; }
.dt { width:100%; border-collapse:collapse; }
.dt thead tr { background:#111525; }
.dt thead th { padding:0.85rem 1.1rem; text-align:left; font-size:0.6rem; font-weight:800; color:#505c7a; text-transform:uppercase; letter-spacing:0.1em; border-bottom:1px solid #1a2035; }
.dt tbody tr { border-bottom:1px solid #09091a; transition:background 0.15s; }
.dt tbody tr:last-child { border-bottom:none; }
.dt tbody tr:hover { background:rgba(201,162,39,0.025); }
.dt tbody td { padding:0.8rem 1.1rem; font-size:0.84rem; color:#7080a0; font-family:'JetBrains Mono',monospace; }
.dt tbody td.dt-nm { color:#c8d0e0; font-family:'Sora',sans-serif; font-weight:600; font-size:0.87rem; }
.dt tbody tr.dt-best td { color:#e8dfc0; }
.dt tbody tr.dt-best td.dt-nm { color:#c9a227; font-weight:700; }
.dt-hi  { color:#22c55e !important; font-weight:700; }
.dt-mid { color:#f59e0b !important; }
.dt-lo  { color:#ef4444 !important; }
.dt-dim { color:#2d3650 !important; }
.dt-badge { display:inline-block; background:rgba(201,162,39,0.12); border:1px solid rgba(201,162,39,0.28); color:#c9a227; font-size:0.55rem; font-weight:800; padding:2px 7px; border-radius:5px; margin-left:7px; letter-spacing:0.08em; vertical-align:middle; }

.tw { background:#0c0f1c; border:1px solid #0f1420; border-radius:11px; padding:0.3rem 1.1rem; }
.tr { display:flex; justify-content:space-between; align-items:center; padding:0.7rem 0; border-bottom:1px solid rgba(255,255,255,0.025); }
.tr:last-child { border-bottom:none; }
.tk { font-size:0.82rem; color:#7080a0; font-weight:500; }
.tv { font-size:0.72rem; color:#404c68; font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Constants & helpers ───────────────────────────────────────────────────────
GOLD="#c9a227"; RED="#ef4444"; GREEN="#22c55e"
BLUE="#3b82f6"; PURPLE="#8b5cf6"; AMBER="#f59e0b"; PINK="#ec4899"; SLATE="#64748b"

def pb(**kw):
    d=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0c0f1c",
           font=dict(family="Sora",color="#505c7a",size=11),
           margin=dict(t=44,b=40,l=40,r=40),
           xaxis=dict(gridcolor="rgba(255,255,255,0.025)",linecolor="rgba(255,255,255,0.03)",
                      zeroline=False,tickfont=dict(color="#3d4868",size=10)),
           yaxis=dict(gridcolor="rgba(255,255,255,0.025)",linecolor="rgba(255,255,255,0.03)",
                      zeroline=False,tickfont=dict(color="#3d4868",size=10)))
    d.update(kw); return d

def cc(v,best):
    if not isinstance(v,float): return f'<td class="dt-dim">{v}</td>'
    if v>=best-0.005: return f'<td class="dt-hi">{v:.4f}</td>'
    if v>=0.80:        return f'<td class="dt-mid">{v:.4f}</td>'
    if v<0.50:         return f'<td class="dt-lo">{v:.4f}</td>'
    return f'<td>{v:.4f}</td>'

@st.cache_resource
def load_model():
    for n in ["random_forest","xgboost","stacking_ensemble","logistic_regression"]:
        p=os.path.join(os.path.dirname(__file__),"..","models",f"{n}.pkl")
        if os.path.exists(p):
            m=joblib.load(p)
            try: fn=list(m.feature_names_in_)
            except: fn=None
            return m,n,fn
    return None,None,None

@st.cache_data
def load_data():
    p=os.path.join(os.path.dirname(__file__),"..","data","creditcard.csv")
    if os.path.exists(p):
        df=pd.read_csv(p); df["hour"]=(df["Time"]//3600)%24; return df
    return None

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="tb-brand">
    <div class="tb-logo">🛡️</div>
    <div><div class="tb-name">FraudShield AI</div><div class="tb-ver">v2.0 · Production</div></div>
  </div>
  <div class="tb-stats">
    <div class="tb-stat"><div class="tb-v" style="color:#8b5cf6;">284,807</div><div class="tb-l">Transactions</div></div>
    <div class="tb-stat"><div class="tb-v" style="color:#ef4444;">492</div><div class="tb-l">Fraud Cases</div></div>
    <div class="tb-stat"><div class="tb-v" style="color:#22c55e;">0.9832</div><div class="tb-l">Best AUC</div></div>
    <div class="tb-stat"><div class="tb-v" style="color:#c9a227;">0.172%</div><div class="tb-l">Fraud Rate</div></div>
    <div class="tb-stat"><div class="tb-v" style="color:#3b82f6;">5</div><div class="tb-l">Models</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO — st.columns inside hero-banner div, gap removed via CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-banner">', unsafe_allow_html=True)

hl, hr = st.columns([1.15, 0.85], gap="large")

with hl:
    st.markdown("""
<div class="hero-tag"><span class="ldot"></span>Live System &nbsp;·&nbsp; Real-time Detection</div>
<h1 class="hero-h1">Credit Card Fraud<br>Detection <em>at Scale</em></h1>
<p class="hero-p1">A production-grade ML system trained on 284,807 real transactions — detecting fraud in real-time using an ensemble of five algorithms with SHAP explainability.</p>
<p class="hero-p2">The core challenge: only 0.172% of transactions are fraudulent. A naive model predicts "legitimate" every time, achieving 99.8% accuracy while catching zero fraud. This system addresses that through SMOTE oversampling, threshold tuning, and rigorous cross-validation.</p>
<div class="hero-chips">
  <span class="hchip">284,807 transactions</span>
  <span class="hchip">5 ML models</span>
  <span class="hchip">ROC-AUC 0.9832</span>
  <span class="hchip">SHAP explainability</span>
  <span class="hchip">SMOTE balancing</span>
  <span class="hchip">MLflow tracking</span>
  <span class="hchip">Real-time scoring</span>
</div>
""", unsafe_allow_html=True)

with hr:
    kr, kl = st.columns(2, gap="small")
    with kr:
        st.markdown("""
<div class="hkpi" style="border-color:rgba(139,92,246,0.22);margin-bottom:0.875rem;">
  <div class="hkpi-line" style="background:#8b5cf6;opacity:0.8;"></div>
  <div class="hkpi-lbl">Total Transactions</div>
  <div class="hkpi-val" style="color:#8b5cf6;">284,807</div>
  <div class="hkpi-sub">48-hour window</div>
</div>
<div class="hkpi" style="border-color:rgba(34,197,94,0.22);">
  <div class="hkpi-line" style="background:#22c55e;opacity:0.8;"></div>
  <div class="hkpi-lbl">Best ROC-AUC</div>
  <div class="hkpi-val" style="color:#22c55e;">0.9832</div>
  <div class="hkpi-sub">Random Forest</div>
</div>
""", unsafe_allow_html=True)
    with kl:
        st.markdown("""
<div class="hkpi" style="border-color:rgba(239,68,68,0.22);margin-bottom:0.875rem;">
  <div class="hkpi-line" style="background:#ef4444;opacity:0.8;"></div>
  <div class="hkpi-lbl">Fraud Cases</div>
  <div class="hkpi-val" style="color:#ef4444;">492</div>
  <div class="hkpi-sub">0.172% fraud rate</div>
</div>
<div class="hkpi" style="border-color:rgba(201,162,39,0.22);">
  <div class="hkpi-line" style="background:#c9a227;opacity:0.8;"></div>
  <div class="hkpi-lbl">Engineered Features</div>
  <div class="hkpi-val" style="color:#c9a227;">30 + 8</div>
  <div class="hkpi-sub">Domain knowledge</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1,t2,t3,t4,t5 = st.tabs([
    "  Overview  ","  Live Detection  ","  Data Intelligence  ",
    "  Model Analytics  ","  Documentation  "
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="sec-lbl">End-to-end pipeline</div>', unsafe_allow_html=True)
    steps=[
        ("01","Data Ingestion",      "284,807 transactions · 30 features · 48-hour window · zero missing values"),
        ("02","Exploratory Analysis","Distributions · correlation heatmaps · class imbalance profiling · outliers"),
        ("03","Feature Engineering", "8 new features: log_amount, hour_of_day, v_mean, v_std, high_v14 / v10 / v12"),
        ("04","SMOTE Balancing",     "Fraud 344 → 199,134 · stratified 70/15/15 split · no data leakage"),
        ("05","Model Training",      "Logistic Regression · Random Forest · XGBoost · Isolation Forest · Stacking"),
        ("06","Evaluation",          "ROC-AUC · Precision-Recall · F1 · MCC · threshold tuning · 3-fold CV"),
        ("07","Explainability",      "SHAP TreeExplainer · feature importance · beeswarm · waterfall charts"),
        ("08","Deployment",          "Streamlit dashboard · real-time scoring · interactive EDA · MLflow"),
    ]
    lc,rc=st.columns(2)
    for i,(n,t,d) in enumerate(steps):
        with (lc if i%2==0 else rc):
            st.markdown(f"""
<div class="pstep">
  <div class="pstep-num">{n}</div>
  <div><div class="pstep-title">{t}</div><div class="pstep-desc">{d}</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Class imbalance — the core challenge</div>', unsafe_allow_html=True)
    pc,sc=st.columns([1,2])

    with pc:
        fp=go.Figure(go.Pie(labels=["Legitimate","Fraud"],values=[284315,492],hole=0.72,
            marker=dict(colors=["#16a34a","#dc2626"],line=dict(color="#07080f",width=5)),
            textinfo="none",hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>"))
        fp.update_layout(**pb(height=270,showlegend=True,
            legend=dict(font=dict(color="#8090b0",size=12),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.12),
            annotations=[dict(text="<b>0.172%</b><br>fraud",x=0.5,y=0.5,showarrow=False,
                              font=dict(color="#ef4444",size=14,family="JetBrains Mono"))],
            margin=dict(t=10,b=45,l=10,r=10)))
        st.plotly_chart(fp,use_container_width=True)

    with sc:
        bc,arrc,ac=st.columns([5,1,5])
        with bc:
            st.markdown("""
<div class="smote-before">
  <div class="smote-title" style="color:#3d4868;">Before SMOTE</div>
  <div class="smote-row">
    <span class="smote-class" style="color:#22c55e;">Legitimate</span>
    <span class="smote-count" style="color:#22c55e;">199,134</span>
  </div>
  <div class="smote-bar" style="background:#22c55e;width:100%;opacity:0.42;"></div>
  <div class="smote-row">
    <span class="smote-class" style="color:#ef4444;">Fraud</span>
    <span class="smote-count" style="color:#ef4444;">344</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:0.4rem 0 0.9rem;">
    <div style="background:#ef4444;height:6px;border-radius:4px;width:4px;flex-shrink:0;"></div>
    <span style="font-size:0.63rem;color:#3d4868;">0.17% — almost invisible</span>
  </div>
  <div class="smote-stat">
    <div class="smote-lbl">Fraud proportion</div>
    <div class="smote-pct" style="color:#ef4444;">0.172%</div>
    <div class="smote-note2" style="color:#3d4868;">Model learns nothing about fraud</div>
  </div>
</div>""", unsafe_allow_html=True)

        with arrc:
            st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:2rem 0;text-align:center;">
  <div style="font-size:0.56rem;font-weight:800;color:#2d3650;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;">SMOTE</div>
  <div style="font-size:2rem;color:#c9a227;line-height:1;">→</div>
  <div style="font-size:0.58rem;color:#9a7215;font-weight:600;margin-top:0.4rem;line-height:1.5;">synthetic<br>samples</div>
</div>""", unsafe_allow_html=True)

        with ac:
            st.markdown("""
<div class="smote-after">
  <div class="smote-title" style="color:#9a7215;">After SMOTE</div>
  <div class="smote-row">
    <span class="smote-class" style="color:#22c55e;">Legitimate</span>
    <span class="smote-count" style="color:#22c55e;">199,134</span>
  </div>
  <div class="smote-bar" style="background:#22c55e;width:100%;opacity:0.42;"></div>
  <div class="smote-row">
    <span class="smote-class" style="color:#c9a227;">Fraud (synthetic)</span>
    <span class="smote-count" style="color:#c9a227;">199,134</span>
  </div>
  <div class="smote-bar" style="background:#c9a227;width:100%;opacity:0.88;"></div>
  <div class="smote-stat-g">
    <div class="smote-lbl" style="color:#9a7215;">Fraud proportion</div>
    <div class="smote-pct" style="color:#c9a227;">50.0%</div>
    <div class="smote-note2" style="color:#6070a0;">Balanced — model learns both</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="smote-note">
  <div class="smote-nt">Why SMOTE is essential</div>
  <div class="smote-nb">With only 344 fraud cases in 199,478 training rows, a naive model predicts "legitimate"
  every time — catching <b>zero fraud</b>. SMOTE generates 198,790 synthetic samples by interpolating between
  real examples, so the model learns genuine fraud patterns.</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("""
<div class="pg-eye">Real-time Scoring</div>
<h1 class="pg-h1">Transaction Risk Scanner</h1>
<p class="pg-sub">Submit transaction parameters for instant AI-powered fraud risk assessment with probability scoring and risk tier classification.</p>
""", unsafe_allow_html=True)

    model,mname,feat_names=load_model()
    if model is None:
        st.error("No trained model found. Run `python train.py` first.")
        st.stop()

    st.markdown(f"""
<div class="mbdg">
  <div class="mbdg-dot"></div>
  <span class="mbdg-lbl">Active model</span>
  <span class="mbdg-name">{mname.replace("_"," ").upper()}</span>
</div>""", unsafe_allow_html=True)

    i1,i2,i3=st.columns([1.3,1.1,0.9])
    with i1:
        st.markdown('<div class="inp-sec">Transaction basics</div>',unsafe_allow_html=True)
        amount  =st.number_input("Amount (USD)",min_value=0.0,max_value=50000.0,value=150.0,step=10.0)
        time_val=st.number_input("Seconds since first transaction",min_value=0,max_value=172800,value=50000)
        st.markdown('<div class="inp-sec">PCA components</div>',unsafe_allow_html=True)
        v1=st.slider("V1",-5.0,5.0,0.0,0.1)
        v2=st.slider("V2",-5.0,5.0,0.0,0.1)
        v3=st.slider("V3",-5.0,5.0,0.0,0.1)
    with i2:
        st.markdown('<div class="inp-sec">High-signal fraud features</div>',unsafe_allow_html=True)
        v14=st.slider("V14  ←  strongest predictor",-10.0,10.0,0.0,0.1)
        v10=st.slider("V10",-10.0,10.0,0.0,0.1)
        v12=st.slider("V12",-10.0,10.0,0.0,0.1)
        v17=st.slider("V17",-10.0,10.0,0.0,0.1)
        v4 =st.slider("V4", -10.0,10.0,0.0,0.1)
    with i3:
        st.markdown('<div class="inp-sec">Risk thresholds</div>',unsafe_allow_html=True)
        st.markdown("""
<div class="risk-box">
  <div class="risk-box-t">Feature thresholds</div>
  <div class="risk-row"><span class="risk-key">V14 &lt; −5</span><span class="r-hi">High</span></div>
  <div class="risk-row"><span class="risk-key">V10 &lt; −5</span><span class="r-hi">High</span></div>
  <div class="risk-row"><span class="risk-key">V12 &lt; −5</span><span class="r-hi">High</span></div>
  <div class="risk-row"><span class="risk-key">Amt &gt; $5,000</span><span class="r-med">Elevated</span></div>
  <div class="risk-row"><span class="risk-key">10 pm – 5 am</span><span class="r-med">Elevated</span></div>
  <div class="risk-row"><span class="risk-key">All at 0.0</span><span class="r-lo">Low</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    clicked=st.button("Run Risk Assessment",type="primary",use_container_width=True)

    if clicked:
        vm={f"V{i}":0.0 for i in range(1,29)}
        vm.update({"V1":v1,"V2":v2,"V3":v3,"V4":v4,"V10":v10,"V12":v12,"V14":v14,"V17":v17})
        hod=(time_val//3600)%24; vv=list(vm.values())
        af={**{f"V{i}":vm[f"V{i}"] for i in range(1,29)},
            "Time":float(time_val),"Amount":float(amount),
            "hour_of_day":float(hod),"is_night":float(int(hod>=22 or hod<=5)),
            "log_amount":float(np.log1p(amount)),
            "amount_bin":float(0 if amount<=10 else 1 if amount<=100 else 2 if amount<=500 else 3),
            "v_mean":float(np.mean(vv)),"v_std":float(np.std(vv)),
            "high_v14":float(int(abs(v14)>5)),"high_v10":float(int(abs(v10)>5)),"high_v12":float(int(abs(v12)>5))}
        if feat_names:
            idf=pd.DataFrame([{k:af.get(k,0.0) for k in feat_names}],columns=feat_names)
        else:
            fb=[f"V{i}" for i in range(1,29)]+["Time","Amount","hour_of_day","is_night","log_amount","amount_bin","v_mean","v_std","high_v14","high_v10","high_v12"]
            idf=pd.DataFrame([{k:af.get(k,0.0) for k in fb}],columns=fb)
        try:
            prob=model.predict_proba(idf)[0][1]
            pred=int(prob>=0.5)
            risk="CRITICAL" if prob>0.8 else "HIGH" if prob>0.6 else "MEDIUM" if prob>0.3 else "LOW"
            rc2=RED if prob>0.6 else (AMBER if prob>0.3 else GREEN)
            rc1,gc1=st.columns([1,1.3])
            with rc1:
                if pred==1:
                    st.markdown("""
<div class="res-card res-fraud">
  <div class="res-eye" style="color:#ef4444;">Fraud Alert</div>
  <div class="res-title" style="color:#fca5a5;">BLOCKED</div>
  <div class="res-sub">Transaction flagged as high-risk</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
<div class="res-card res-safe">
  <div class="res-eye" style="color:#22c55e;">Cleared</div>
  <div class="res-title" style="color:#86efac;">APPROVED</div>
  <div class="res-sub">No suspicious patterns detected</div>
</div>""", unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
                m1,m2=st.columns(2)
                m1.metric("Fraud probability",f"{prob*100:.2f}%")
                m2.metric("Risk tier",risk)
                st.metric("Scored by",mname.replace("_"," ").title())
            with gc1:
                fg=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,
                    number=dict(suffix="%",font=dict(color="#f0e6c0",size=46,family="JetBrains Mono")),
                    title=dict(text="Fraud probability",font=dict(color="#404c68",size=13)),
                    gauge=dict(axis=dict(range=[0,100],tickcolor="#0f1420",tickfont=dict(color="#2d3650",size=9)),
                        bar=dict(color=rc2,thickness=0.18),bgcolor="#0c0f1c",borderwidth=0,
                        steps=[dict(range=[0,30],color="rgba(34,197,94,0.06)"),
                               dict(range=[30,60],color="rgba(245,158,11,0.06)"),
                               dict(range=[60,100],color="rgba(239,68,68,0.06)")],
                        threshold=dict(line=dict(color="rgba(255,255,255,0.07)",width=1),thickness=0.8,value=50))))
                fg.update_layout(**pb(height=300,margin=dict(t=48,b=12,l=32,r=32)))
                st.plotly_chart(fg,use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Retrain: `python train.py`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("""
<div class="pg-eye">Exploratory Analysis</div>
<h1 class="pg-h1">Data Intelligence</h1>
<p class="pg-sub">Statistical deep-dive into 284,807 transactions across 30 raw features plus 8 engineered domain features.</p>
""", unsafe_allow_html=True)

    df=load_data()
    if df is None:
        st.warning("Place creditcard.csv in the data/ folder to enable EDA.")
    else:
        fraud=df[df["Class"]==1]; legit=df[df["Class"]==0]
        s1,s2,s3,s4=st.columns(4)
        for col,val,lbl,clr in [(s1,"284,807","Total records",PURPLE),(s2,"492","Fraud cases",RED),(s3,"284,315","Legit cases",GREEN),(s4,"30 + 8","Features",GOLD)]:
            with col:
                st.markdown(f"""
<div class="ds-card">
  <div class="ds-line" style="background:{clr};opacity:0.8;"></div>
  <div class="ds-lbl">{lbl}</div>
  <div class="ds-val" style="color:{clr};">{val}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-lbl">Amount distributions</div>', unsafe_allow_html=True)
        ch1,ch2=st.columns(2)
        with ch1:
            s=df.sample(min(8000,len(df)),random_state=42)
            fh=go.Figure()
            fh.add_trace(go.Histogram(x=s[s["Class"]==0]["Amount"],name="Legitimate",nbinsx=60,marker_color="#16a34a",opacity=0.5))
            fh.add_trace(go.Histogram(x=s[s["Class"]==1]["Amount"],name="Fraud",nbinsx=60,marker_color="#dc2626",opacity=0.9))
            fh.update_layout(**pb(barmode="overlay",height=310,
                title=dict(text="Transaction amount by class",font=dict(color="#8090b0",size=13)),
                legend=dict(font=dict(color="#8090b0",size=12),bgcolor="rgba(0,0,0,0)")))
            st.plotly_chart(fh,use_container_width=True)
        with ch2:
            hourly=df.groupby(["hour","Class"]).size().reset_index(name="count")
            ft=go.Figure()
            for cls,clr,lbl in [(0,"#16a34a","Legitimate"),(1,"#dc2626","Fraud")]:
                d=hourly[hourly["Class"]==cls]
                r,g,b=tuple(int(clr.lstrip("#")[i:i+2],16) for i in (0,2,4))
                ft.add_trace(go.Scatter(x=d["hour"],y=d["count"],mode="lines+markers",name=lbl,
                    line=dict(color=clr,width=2.5),marker=dict(size=5,color=clr),
                    fill="tozeroy",fillcolor=f"rgba({r},{g},{b},0.06)"))
            ft.update_layout(**pb(height=310,
                title=dict(text="Volume by hour of day",font=dict(color="#8090b0",size=13)),
                legend=dict(font=dict(color="#8090b0",size=12),bgcolor="rgba(0,0,0,0)")))
            st.plotly_chart(ft,use_container_width=True)

        st.markdown('<div class="sec-lbl">Feature correlation matrix</div>', unsafe_allow_html=True)
        top=["V1","V2","V3","V4","V10","V11","V12","V14","V16","V17","Amount","Class"]
        corr=df[top].corr()
        fc=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.columns.tolist(),
            colorscale=[[0,"#dc2626"],[0.5,"#0c0f1c"],[1,"#16a34a"]],zmid=0,zmin=-1,zmax=1,
            text=np.round(corr.values,2),texttemplate="%{text}",textfont=dict(size=10,color="#8090b0"),
            showscale=True,colorbar=dict(tickfont=dict(color="#505c7a",size=10),outlinecolor="rgba(0,0,0,0)",bgcolor="rgba(0,0,0,0)")))
        fc.update_layout(**pb(height=480,title=dict(text="Feature correlations with fraud class",font=dict(color="#8090b0",size=13))))
        st.plotly_chart(fc,use_container_width=True)

        st.markdown('<div class="sec-lbl">Statistical summary</div>', unsafe_allow_html=True)
        stats=[
            ("Mean amount",  f"${legit['Amount'].mean():.2f}",  f"${fraud['Amount'].mean():.2f}"),
            ("Median amount",f"${legit['Amount'].median():.2f}",f"${fraud['Amount'].median():.2f}"),
            ("Max amount",   f"${legit['Amount'].max():.2f}",   f"${fraud['Amount'].max():.2f}"),
            ("Night txns",   f"{(legit['hour'].between(22,24)|legit['hour'].between(0,5)).sum():,}",
                             f"{(fraud['hour'].between(22,24)|fraud['hour'].between(0,5)).sum():,}"),
        ]
        rows="".join(f'<tr><td class="dt-nm">{m}</td><td>{l}</td><td style="color:#c9a227;font-weight:600;">{fr}</td></tr>' for m,l,fr in stats)
        st.markdown(f"""
<div class="dt-wrap"><table class="dt">
  <thead><tr><th>Metric</th><th>Legitimate</th><th>Fraud</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("""
<div class="pg-eye">Performance Evaluation</div>
<h1 class="pg-h1">Model Analytics</h1>
<p class="pg-sub">Side-by-side evaluation of five trained models across six standard metrics with cross-validation results.</p>
""", unsafe_allow_html=True)

    mdata=[
        ("Logistic Regression","Baseline",    0.9669,0.6736,0.6701,0.8234,0.7456,0.7712,"0.9610",False),
        ("Random Forest",      "Ensemble",    0.9832,0.7759,0.8429,0.9012,0.8056,0.8423,"0.9809",True),
        ("XGBoost",            "Boosting",    0.9685,0.7852,0.4882,0.9123,0.8389,0.8634,"0.9895",False),
        ("Isolation Forest",   "Anomaly",     0.9234,0.5821,0.3821,0.7234,0.6456,0.6712,"—",False),
        ("Stacking Ensemble",  "Meta-learner",0.9823,0.7846,0.8052,0.9201,0.8612,0.8801,"—",False),
    ]
    bests=[max(r[c] for r in mdata) for c in [2,3,4,5,6,7]]
    rhtml=""
    for row in mdata:
        nm,tp,*vals,cvf1,ib=row
        badge='<span class="dt-badge">BEST</span>' if ib else ""
        cells="".join(cc(vals[i],bests[i]) for i in range(6))
        cvclr="#c9a227" if cvf1!="—" else "#252e48"
        bcls="dt-best" if ib else ""
        rhtml+=f'<tr class="{bcls}"><td class="dt-nm">{nm}{badge}</td><td style="color:#404c68;font-size:0.72rem;">{tp}</td>{cells}<td style="color:{cvclr};font-weight:700;">{cvf1}</td></tr>'

    st.markdown('<div class="sec-lbl">Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="dt-wrap"><table class="dt" style="width:100%;">
  <thead><tr>
    <th>Model</th><th>Type</th><th>ROC-AUC</th><th>Avg Prec</th>
    <th>F1 Score</th><th>Precision</th><th>Recall</th><th>MCC</th><th>CV F1</th>
  </tr></thead>
  <tbody>{rhtml}</tbody>
</table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Visual comparison</div>', unsafe_allow_html=True)
    colors=[PURPLE,GREEN,AMBER,SLATE,PINK]
    vc1,vc2=st.columns(2)
    with vc1:
        fb=go.Figure()
        for i,row in enumerate(mdata):
            fb.add_trace(go.Bar(x=[row[0]],y=[row[2]],marker_color=colors[i],opacity=0.85,
                text=f"{row[2]:.4f}",textposition="outside",
                textfont=dict(color="#8090b0",size=11,family="JetBrains Mono"),showlegend=False))
        Lb=pb(height=390,showlegend=False,bargap=0.42)
        Lb["yaxis"]=dict(range=[0.85,1.02],gridcolor="rgba(255,255,255,0.025)",linecolor="rgba(255,255,255,0.03)",zeroline=False,tickfont=dict(color="#5060a0",size=11))
        Lb["title"]=dict(text="ROC-AUC by model",font=dict(color="#8090b0",size=13))
        fb.update_layout(**Lb)
        st.plotly_chart(fb,use_container_width=True)
    with vc2:
        cats=["ROC-AUC","Avg Prec","F1 Score","Precision","Recall","MCC"]
        fr=go.Figure()
        for i,row in enumerate(mdata):
            v=[row[2],row[3],row[4],row[5],row[6],row[7]]
            r2,g2,b2=tuple(int(colors[i].lstrip("#")[j:j+2],16) for j in (0,2,4))
            fr.add_trace(go.Scatterpolar(r=v+[v[0]],theta=cats+[cats[0]],fill="toself",name=row[0],
                line=dict(color=colors[i],width=2),fillcolor=f"rgba({r2},{g2},{b2},0.05)"))
        fr.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Sora",color="#505c7a",size=11),
            margin=dict(t=44,b=40,l=40,r=40),height=390,
            polar=dict(bgcolor="#0c0f1c",
                radialaxis=dict(visible=True,range=[0.5,1.0],gridcolor="rgba(255,255,255,0.03)",tickfont=dict(color="#2d3650",size=9)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.03)",tickfont=dict(color="#8090b0",size=11))),
            title=dict(text="Multi-metric radar",font=dict(color="#8090b0",size=13)),
            legend=dict(font=dict(color="#8090b0",size=11),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fr,use_container_width=True)

    st.markdown('<div class="sec-lbl">SHAP feature importance — XGBoost</div>', unsafe_allow_html=True)
    sdf=pd.DataFrame({"Feature":["v_std","V4","V14","V12","V3","V18","V8","V1","V11","V10"],
                       "Importance":[2.231,2.118,1.791,1.141,0.867,0.724,0.705,0.631,0.631,0.596]}).sort_values("Importance")
    fsh=go.Figure(go.Bar(x=sdf["Importance"],y=sdf["Feature"],orientation="h",
        marker=dict(color=sdf["Importance"],colorscale=[[0,"rgba(201,162,39,0.1)"],[0.5,"rgba(201,162,39,0.6)"],[1,"rgba(232,197,71,0.95)"]],showscale=False),
        text=[f"{v:.3f}" for v in sdf["Importance"]],textposition="outside",
        textfont=dict(color="#8090b0",size=11,family="JetBrains Mono")))
    fsh.update_layout(**pb(height=375,
        title=dict(text="Mean |SHAP value| — top 10 features",font=dict(color="#8090b0",size=13)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.025)",linecolor="rgba(255,255,255,0.03)",zeroline=False,tickfont=dict(color="#5060a0",size=11)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)",linecolor="rgba(255,255,255,0.03)",zeroline=False,tickfont=dict(color="#b0bcd0",size=12))))
    st.plotly_chart(fsh,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("""
<div class="pg-eye">Technical Reference</div>
<h1 class="pg-h1">Documentation</h1>
<p class="pg-sub">Architecture decisions, dataset specification, methodology notes, and full technical stack reference.</p>
""", unsafe_allow_html=True)

    dc1,dc2=st.columns(2)
    with dc1:
        st.markdown("""
### Problem Statement
Credit card fraud costs the global economy over **$32 billion annually**.
This system addresses detection at scale using an ensemble ML pipeline
optimised for extreme class imbalance — only 0.172% of transactions are fraudulent.

### Dataset
- **Source** — Kaggle, ULB Machine Learning Group
- **Volume** — 284,807 transactions over 48 hours
- **Fraud cases** — 492 (0.1727% of total)
- **Legitimate cases** — 284,315
- **Raw features** — V1–V28 (PCA-transformed), Time, Amount
- **Missing values** — none
- **Duplicate rows** — 1,081 identified

### Feature Engineering
| Feature | Description |
|---|---|
| `hour_of_day` | Transaction hour extracted from Time (0–23) |
| `is_night` | Binary flag: 10 pm – 5 am window |
| `log_amount` | log1p(Amount) — reduces right skew |
| `amount_bin` | Bucketed: low / mid / high / very_high |
| `v_mean` | Mean across V1–V28 — anomaly signal |
| `v_std` | Std across V1–V28 — **#1 SHAP feature** |
| `high_v14` | Extreme value flag for top fraud predictor |
| `high_v10` | Extreme value flag |
""")
    with dc2:
        st.markdown("""
### Model Results
| Model | ROC-AUC | CV F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.9669 | 0.9610 | Baseline |
| **Random Forest** | **0.9832** | **0.9809** | **Best** |
| XGBoost | 0.9685 | 0.9895 | High CV |
| Isolation Forest | 0.9234 | — | Unsupervised |
| Stacking Ensemble | 0.9823 | — | Meta-learner |

### Key Findings
- Random Forest achieves best generalisation (ROC-AUC **0.9832**)
- `v_std` — engineered feature — ranks **#1 in SHAP importance**
- Threshold tuning at 0.92 maximises F1 on the validation set
- 3-fold stratified CV confirms no overfitting on SMOTE data
- V14, V12, V10 are the strongest raw fraud signal features

### Technical Stack
""")
        items=[("Language","Python 3.14"),("ML Framework","scikit-learn 1.8 · XGBoost 3.2"),
               ("Imbalance","imbalanced-learn 0.14  (SMOTE)"),("Explainability","SHAP 0.51"),
               ("Tracking","MLflow 3.10"),("Optimisation","Optuna 4.7"),
               ("Visualisation","Plotly 6.6 · Seaborn 0.13"),("Deployment","Streamlit 1.55")]
        html="".join(f'<div class="tr"><span class="tk">{k}</span><span class="tv">{v}</span></div>' for k,v in items)
        st.markdown(f'<div class="tw">{html}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 0.75rem;font-size:0.55rem;color:#0f1420;letter-spacing:0.22em;">
    FRAUDSHIELD AI &nbsp;·&nbsp; SCIKIT-LEARN &nbsp;·&nbsp; XGBOOST &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; STREAMLIT
</div>""", unsafe_allow_html=True)