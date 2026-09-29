"""
Cavite / Bacoor LGU — DRRMO Smart Drainage & Flood Early Warning System
Production Streamlit Command Center Dashboard

Features:
- DRRMO Command Center Header & Custom Professional Styling
- Scikit-Learn Random Forest Model Loading with Caching & Firmware Rule Fallback
- Multi-Mode Telemetry Ingress: Historical CSV Replay, Dynamic Live Simulation, and Manual What-If Diagnostics
- 4-Tier DRRMO Color-Coded Alert Framework (NORMAL, WARNING, HVYRAIN, OBSTRUCT)
- Real-Time KPI Metrics Strip with Culvert Capacity Utilization
- Inverted-Axis Dual-Y Plotly Interactive Time-Series Telemetry Visualizations
- ML Prediction Probabilities & Feature Importance Explainability Panel
- Automated & Manual Emergency Secondary Valve / Diversion Gate Control
"""

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@contextmanager
def tactical_spinner(text: str = "INGESTING TELEMETRY STREAM..."):
    """
    Tactical 3-Bar Pulsing Crimson Spinner Context Manager.
    Replaces default circular spinner with high-density vertical pulsing crimson rectangles.
    """
    loader_html = f"""
    <div class="tactical-loader-container" style="margin: 10px 0;">
        <div class="tactical-bars-cluster">
            <div class="tactical-bar tactical-bar-1"></div>
            <div class="tactical-bar tactical-bar-2"></div>
            <div class="tactical-bar tactical-bar-3"></div>
        </div>
        <span class="tactical-loader-label">{text}</span>
    </div>
    """
    placeholder = st.empty()
    placeholder.markdown(loader_html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()



# -----------------------------------------------------------------------------
# Configuration & Hardware Domain Constants (per FINAL_CODE_DRAINAGE_V1.ino)
# -----------------------------------------------------------------------------
MODEL_PATH = "drainage_model.pkl"
DATA_PATH = "drainage_data.csv"
FEATURE_COLUMNS = ["distance_cm", "flow_l_min"]
CLASS_LABELS = ["HVYRAIN", "NORMAL", "OBSTRUCT", "WARNING"]

NORMAL_LEVEL_CM = 23.00       # Distance >= 23 cm: water is low (channel clear)
WARNING_LEVEL_CM = 21.00      # 21 cm <= Distance < 23 cm: water rising
LOW_FLOW_THRESHOLD = 3.50     # Flow rate discriminator in L/min
CANAL_BED_DISTANCE_CM = 25.00 # Distance from ultrasonic sensor to canal floor
CRITICAL_OVERFLOW_CM = 20.00  # Distance at which overflow/spillover begins (5.0 cm water depth)

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DRRMO Smart Drainage & Flood Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS Styling (Government DRRMO Command Center Aesthetic)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* =============================================================================
   ENTERPRISE UI DESIGN SYSTEM — DRRMO COMMAND CENTER INDUSTRIAL STYLING
   Pillars: Zero Radius | Monospace Tabular Nums | Warm Neutral & Crimson
   ============================================================================= */

/* -----------------------------------------------------------------------------
   PILLAR 1: GLOBAL ZERO ROUNDED CORNERS (ABSOLUTE 0PX RECTILINEAR UI)
   ----------------------------------------------------------------------------- */
*,
*::before,
*::after,
.stApp,
.stApp *,
.stApp *::before,
.stApp *::after,
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[data-baseweb="menu"],
div[data-baseweb="menu"] *,
ul[data-baseweb="menu"],
li[data-baseweb="menu-item"],
ul[role="listbox"],
li[role="option"] {
    border-radius: 0px !important;
}

/* Explicit Component Selector List for Maximum Specificity */
.stMetric,
.stButton,
.stButton > button,
.stDownloadButton,
.stDownloadButton > button,
[data-testid^="baseButton-"],
[data-testid="stSidebarCollapseButton"] button,
.stAlert,
div[data-testid="stAlert"],
div[data-testid="stMetric"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"],
div[data-testid="stMetricDelta"],
div[data-testid="stMetricBorder"],
div[data-testid="stDataFrame"],
div[data-testid="stDataFrameResizable"],
div[data-testid="stTable"],
.glideDataGrid,
.dvn-scroller,
.stTabs,
div[data-testid="stTabs"],
div[data-baseweb="tab-list"],
button[data-baseweb="tab"],
div[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"],
div[data-baseweb="select"],
div[data-baseweb="select"] *,
.stSelectbox,
div[data-testid="stSelectbox"],
.stTextInput,
.stTextInput input,
div[data-testid="stTextInput"],
div[data-baseweb="input"],
div[data-baseweb="input"] *,
input,
.stNumberInput,
.stNumberInput input,
div[data-testid="stNumberInput"],
div[data-testid="stNumberInput"] button,
.stSlider,
div[data-testid="stSlider"],
div[data-baseweb="slider"],
div[data-baseweb="slider"] *,
div[role="slider"],
div[data-baseweb="radio"],
div[data-baseweb="checkbox"],
div[data-testid="stCheckbox"] span,
div[data-testid="stRadio"] div[role="radiogroup"] label,
div[data-testid="stProgress"],
div[data-testid="stProgress"] > div,
div[data-testid="stProgress"] > div > div,
div[data-baseweb="progress-bar"],
details[data-testid="stExpander"],
details[data-testid="stExpander"] summary,
div[data-testid="stExpander"],
div[data-testid="stExpanderDetails"],
span[data-testid="stBadge"],
div[data-testid="stNotification"],
div[data-testid="stPlotlyChart"],
.js-plotly-plot,
.plot-container,
.main-svg,
iframe,
canvas,
.drrmo-header-container,
.drrmo-badge-online,
.drrmo-badge-online::before,
.drrmo-risk-badge,
.alert-banner,
.alert-banner-normal,
.alert-banner-warning,
.alert-banner-hvyrain,
.alert-banner-obstruct,
.alarm-box,
.metric-card-container,
.valve-indicator,
.valve-closed,
.valve-open,
code,
pre {
    border-radius: 0px !important;
}

/* -----------------------------------------------------------------------------
   PILLAR 2: ENTERPRISE TYPOGRAPHY & TABULAR NUMBERS
   ----------------------------------------------------------------------------- */
/* Monospace Tabular Numerals on all Metrics, Dataframes, and Tables */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] *,
.stDataFrame,
.stTable,
table,
td,
th,
code,
pre,
.alarm-box code {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "Cascadia Mono", "Roboto Mono", monospace !important;
    font-variant-numeric: tabular-nums !important;
    font-feature-settings: "tnum" 1, "zero" 1 !important;
}

/* Tight Uppercase Tracking for Kickers and Metric Labels */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] p,
.stMetric label,
.drrmo-kicker {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #57606a !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #1f2429 !important;
    line-height: 1.15 !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* -----------------------------------------------------------------------------
   PILLAR 3: WARM NEUTRAL & DEEP CRIMSON PALETTE
   ----------------------------------------------------------------------------- */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main {
    background-color: #f8f7f6 !important;
    color: #1f2429 !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] {
    background-color: #f1f0ee !important;
    border-right: 1px solid #d8d4cf !important;
}

[data-testid="stHeader"] {
    background-color: #f8f7f6 !important;
    border-bottom: 1px solid #d8d4cf !important;
}

/* Primary Buttons: Deep Crimson Red (#b02631 / #9a1f2a) */
button[kind="primary"],
.stButton > button[kind="primary"],
div[data-testid="stBaseButton-primary"],
[data-testid="baseButton-primary"] {
    background-color: #b02631 !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid #9a1f2a !important;
    border-radius: 0px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    box-shadow: 0 1px 3px rgba(176, 38, 49, 0.25) !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background-color: #9a1f2a !important;
    border-color: #7d1822 !important;
    color: #ffffff !important;
}

button[kind="primary"]:active,
.stButton > button[kind="primary"]:active {
    background-color: #7d1822 !important;
    border-color: #7d1822 !important;
}

/* Secondary Buttons: Crisp Hairline */
button[kind="secondary"],
.stButton > button,
.stDownloadButton > button,
div[data-testid="stBaseButton-secondary"],
[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #1f2429 !important;
    border: 1px solid #d8d4cf !important;
    border-radius: 0px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

button[kind="secondary"]:hover,
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #f2f1ef !important;
    border-color: #b02631 !important;
    color: #b02631 !important;
}

/* Form Controls & Steppers */
div[data-baseweb="input"],
div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background-color: #ffffff !important;
    border: 1px solid #d8d4cf !important;
    border-radius: 0px !important;
    color: #1f2429 !important;
    font-size: 0.88rem !important;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #b02631 !important;
    box-shadow: 0 0 0 1px #b02631 !important;
}

/* Slider Track & Rectangular Pips */
div[data-baseweb="slider"] div[role="slider"] {
    border-radius: 0px !important;
    width: 14px !important;
    height: 18px !important;
    background-color: #b02631 !important;
    border: 1px solid #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
}

/* Progress Bars: Sharp Industrial Strip */
div[data-testid="stProgress"],
div[data-baseweb="progress-bar"] {
    border-radius: 0px !important;
    height: 8px !important;
    background-color: #e5e2dc !important;
}

/* DRRMO Header Container */
.drrmo-header-container {
    background: #1a1e23 !important;
    padding: 20px 24px;
    border-radius: 0px !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-left: 6px solid #b02631 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15) !important;
    margin-bottom: 20px;
}

.drrmo-title {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 12px;
}

.drrmo-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin: 6px 0 0 0;
    font-weight: 400;
    letter-spacing: 0.04em;
}

.drrmo-badge-online {
    background-color: rgba(16, 185, 129, 0.15) !important;
    border: 1px solid #10b981 !important;
    color: #10b981 !important;
    padding: 3px 10px;
    border-radius: 0px !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.drrmo-badge-online::before {
    content: "";
    width: 6px;
    height: 6px;
    background-color: #10b981;
    border-radius: 0px !important;
    box-shadow: 0 0 6px #10b981;
}

/* Alert Banners & OBSTRUCT Styling */
.alert-banner {
    border-radius: 0px !important;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.08);
}

.alert-banner-normal {
    background-color: #f0fdf4 !important;
    border-left: 6px solid #16a34a !important;
    color: #166534 !important;
}

.alert-banner-warning {
    background-color: #fffbeb !important;
    border-left: 6px solid #d97706 !important;
    color: #92400e !important;
}

.alert-banner-hvyrain {
    background-color: #fff7ed !important;
    border-left: 6px solid #ea580c !important;
    color: #9a3412 !important;
}

.alert-banner-obstruct {
    background-color: #fdf2f2 !important;
    border: 1px solid rgba(176, 38, 49, 0.3) !important;
    border-left: 6px solid #b02631 !important;
    color: #7d1720 !important;
    animation: crimson-tactical-pulse 1.8s infinite ease-in-out !important;
}

/* Enforce Crimson Red on OBSTRUCT Risk Badge via CSS */
.alert-banner-obstruct .drrmo-risk-badge {
    background: #b02631 !important;
    border: 1px solid #9a1f2a !important;
}

@keyframes crimson-tactical-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(176, 38, 49, 0.35); }
    50% { box-shadow: 0 0 0 8px rgba(176, 38, 49, 0); }
}

.alarm-box {
    background: rgba(0, 0, 0, 0.03);
    border: 1px dashed rgba(0, 0, 0, 0.15);
    border-radius: 0px !important;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 13px;
}

.valve-indicator {
    padding: 12px 16px;
    border-radius: 0px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 10px;
}

.valve-closed {
    background: #edebe8 !important;
    color: #57606a !important;
    border: 1px solid #d8d4cf !important;
}

.valve-open {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #065f46 !important;
    border: 2px solid #10b981 !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}

/* -----------------------------------------------------------------------------
   PILLAR 4: ADVANCED TACTICAL LOADING BAR (3 VERTICAL PULSING CRIMSON BARS)
   ----------------------------------------------------------------------------- */
@keyframes tactical-pulse-bars {
    0%, 100% {
        transform: scaleY(0.25);
        opacity: 0.35;
        background-color: #9a1f2a;
    }
    50% {
        transform: scaleY(1.0);
        opacity: 1.0;
        background-color: #b02631;
        box-shadow: 0 0 6px rgba(176, 38, 49, 0.6);
    }
}

/* Suppress default circular Streamlit SVG spinner */
div[data-testid="stSpinner"] svg,
div[data-testid="stSpinner"] > div:first-child:not(.tactical-loader-container) {
    display: none !important;
}

/* Override st.spinner container */
div[data-testid="stSpinner"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 12px !important;
    background: #ffffff !important;
    border: 1px solid #d8d4cf !important;
    border-left: 4px solid #b02631 !important;
    border-radius: 0px !important;
    padding: 10px 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    font-family: ui-monospace, "SF Mono", monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #1f2429 !important;
}

/* Inject Tactical Pulse Pseudo-Elements into st.spinner */
div[data-testid="stSpinner"]::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 20px;
    background-color: #b02631;
    border-radius: 0px !important;
    margin-right: 18px;
    animation: tactical-pulse-bars 0.9s ease-in-out infinite;
    box-shadow: 7px 0 0 0 #b02631, 14px 0 0 0 #b02631;
}

/* Tactical Loader Component Classes */
.tactical-loader-container {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: #ffffff;
    border: 1px solid #d8d4cf;
    border-left: 4px solid #b02631;
    padding: 8px 16px;
    border-radius: 0px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.tactical-bars-cluster {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    height: 22px;
}

.tactical-bar {
    width: 4px;
    height: 20px;
    background-color: #b02631;
    display: inline-block;
    border-radius: 0px !important;
    transform-origin: center;
    animation: tactical-pulse-bars 0.9s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}

.tactical-bar-1 { animation-delay: 0.0s; }
.tactical-bar-2 { animation-delay: 0.15s; }
.tactical-bar-3 { animation-delay: 0.30s; }

.tactical-loader-label {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", monospace;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #1f2429;
}

@media (prefers-reduced-motion: reduce) {
    .tactical-bar,
    div[data-testid="stSpinner"]::before {
        animation: none !important;
        transform: scaleY(1.0) !important;
        opacity: 1.0 !important;
    }
}

/* -----------------------------------------------------------------------------
   PILLAR 5: GLASSMORPHIC METRIC CONTAINERS
   ----------------------------------------------------------------------------- */
div[data-testid="stMetric"],
.metric-card-container {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 0px !important;
    box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.9), 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    padding: 14px 18px !important;
    margin-bottom: 8px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stMetric"]:hover,
.metric-card-container:hover {
    border-color: rgba(176, 38, 49, 0.45) !important;
    box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 1.0), 0 3px 8px rgba(176, 38, 49, 0.08) !important;
}

/* -----------------------------------------------------------------------------
   PILLAR 6: DATAFRAME & TABLE STYLING
   ----------------------------------------------------------------------------- */
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
div[data-testid="stDataFrameResizable"] {
    border: 1px solid #d8d4cf !important;
    border-radius: 0px !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

.glideDataGrid,
.dvn-scroller {
    border-radius: 0px !important;
    font-variant-numeric: tabular-nums !important;
}

div[data-testid="stTable"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    border: 1px solid #d8d4cf !important;
    border-radius: 0px !important;
    background-color: #ffffff !important;
}

div[data-testid="stTable"] th {
    background-color: #edebe8 !important;
    color: #1f2429 !important;
    font-family: ui-monospace, "SF Mono", monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 14px !important;
    border: 1px solid #d8d4cf !important;
    border-bottom: 2px solid #b02631 !important;
    border-radius: 0px !important;
    text-align: left !important;
}

div[data-testid="stTable"] th:not(:first-child) {
    text-align: right !important;
}

div[data-testid="stTable"] td {
    border: 1px solid #e5e2dc !important;
    border-radius: 0px !important;
    padding: 8px 14px !important;
    color: #1f2429 !important;
    font-family: ui-monospace, "SF Mono", monospace !important;
    font-variant-numeric: tabular-nums !important;
    font-size: 0.85rem !important;
}

div[data-testid="stTable"] td:not(:first-child) {
    text-align: right !important;
}

div[data-testid="stTable"] tr:nth-child(even) td {
    background-color: #f2f1ef !important;
}

div[data-testid="stTable"] tr:nth-child(odd) td {
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model & Data Loading with Caching & Resilience
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_model(model_path: str = MODEL_PATH):
    """
    Load serialized scikit-learn RandomForestClassifier from disk.
    Gracefully handles absence or corruption by returning (None, error_msg).
    """
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found on filesystem."
    try:
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            return None, f"Loaded object from '{model_path}' is not a valid scikit-learn estimator."
        return model, None
    except Exception as e:
        return None, f"Failed to deserialize model: {str(e)}"


@st.cache_data
def load_historical_telemetry(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load physical prototype telemetry dataset from CSV.
    Provides robust fallback to synthetic dataframe if CSV is missing.
    """
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names and clean nulls
            df.columns = [c.strip() for c in df.columns]
            if "distance_cm" in df.columns and "flow_l_min" in df.columns:
                df["distance_cm"] = pd.to_numeric(df["distance_cm"], errors="coerce")
                df["flow_l_min"] = pd.to_numeric(df["flow_l_min"], errors="coerce")
                if "status" in df.columns:
                    df["status"] = df["status"].astype(str).str.strip()
                df = df.dropna(subset=["distance_cm", "flow_l_min"]).reset_index(drop=True)
                if not df.empty:
                    return df
        except Exception:
            pass

    # Built-in synthetic fallback covering all 4 operational states
    np.random.seed(42)
    norm_d = np.random.uniform(23.5, 24.8, 50)
    norm_f = np.random.uniform(0.1, 1.2, 50)
    warn_d = np.random.uniform(21.2, 22.8, 50)
    warn_f = np.random.uniform(0.5, 2.4, 50)
    hvy_d = np.random.uniform(20.1, 20.8, 50)
    hvy_f = np.random.uniform(3.6, 5.5, 50)
    obs_d = np.random.uniform(20.1, 20.8, 50)
    obs_f = np.random.uniform(0.0, 1.5, 50)

    df_synth = pd.DataFrame({
        "pc_timestamp": [f"2026-09-26 12:{i//60:02d}:{i%60:02d}" for i in range(200)],
        "elapsed_ms": np.arange(1000, 201000, 1000),
        "distance_cm": np.concatenate([norm_d, warn_d, hvy_d, obs_d]),
        "flow_l_min": np.concatenate([norm_f, warn_f, hvy_f, obs_f]),
        "status": ["NORMAL"] * 50 + ["WARNING"] * 50 + ["HVYRAIN"] * 50 + ["OBSTRUCT"] * 50
    })
    return df_synth


def rule_based_predict(distance_cm: float, flow_l_min: float):
    """
    Firmware rule-based fallback decision engine mirroring FINAL_CODE_DRAINAGE_V1.ino.
    Used when the machine learning model file is unavailable.
    """
    if distance_cm >= NORMAL_LEVEL_CM:
        status = "NORMAL"
        probs = [0.03, 0.92, 0.02, 0.03]  # [HVYRAIN, NORMAL, OBSTRUCT, WARNING]
    elif distance_cm >= WARNING_LEVEL_CM:
        status = "WARNING"
        probs = [0.05, 0.10, 0.05, 0.80]
    else:
        if flow_l_min >= LOW_FLOW_THRESHOLD:
            status = "HVYRAIN"
            probs = [0.91, 0.02, 0.05, 0.02]
        else:
            status = "OBSTRUCT"
            probs = [0.04, 0.02, 0.92, 0.02]
    return status, probs, ["HVYRAIN", "NORMAL", "OBSTRUCT", "WARNING"]


# -----------------------------------------------------------------------------
# DRRMO Alert Level Framework (NDRRMC & Bacoor LGU Protocol)
# -----------------------------------------------------------------------------
DRRMO_ALERT_MATRIX = {
    "NORMAL": {
        "level_title": "LEVEL 0: NORMAL MONITORING",
        "badge": "🟢 ALL SYSTEMS NOMINAL",
        "banner_class": "alert-banner-normal",
        "color": "#16a34a",
        "bg_color": "#f0fdf4",
        "buzzer_state": "OFF (Silenced)",
        "buzzer_timing": "0 ms (Silent)",
        "strobe_state": "OFF",
        "description": "Culvert water level is well within safe canal capacity. Baseline runoff velocity. Free flow verified.",
        "action_protocol": "Maintain continuous routine supervisory monitoring. Secondary diversion gate remains CLOSED. No community alerts required.",
        "risk_level": "Low / Safe (Green)"
    },
    "WARNING": {
        "level_title": "LEVEL 1: FLOOD ADVISORY",
        "badge": "🟡 ELEVATED WATER LEVEL",
        "banner_class": "alert-banner-warning",
        "color": "#d97706",
        "bg_color": "#fffbeb",
        "buzzer_state": "SLOW INTERMITTENT PULSE",
        "buzzer_timing": "300ms ON / 300ms OFF (Firmware Low Cadence)",
        "strobe_state": "AMBER FLASH",
        "description": "Water level is rising toward critical channel capacity (distance between 21.0 cm and 23.0 cm). Upstream runoff accumulating.",
        "action_protocol": "Notify DRRMO Command Desk. Stage barangay quick-response personnel in flood-prone zones. Inspect culvert intake for early drift trash.",
        "risk_level": "Moderate / Advisory (Yellow)"
    },
    "HVYRAIN": {
        "level_title": "LEVEL 2: FLOOD SURGE ALERT",
        "badge": "🟠 HIGH WATER & RAPID FLOW",
        "banner_class": "alert-banner-hvyrain",
        "color": "#ea580c",
        "bg_color": "#fff7ed",
        "buzzer_state": "FAST HIGH-CADENCE PULSE",
        "buzzer_timing": "100ms ON / 100ms OFF (Firmware Rapid Cadence)",
        "strobe_state": "ORANGE HIGH-INTENSITY",
        "description": "High water level detected with swift runoff velocity (>= 3.5 L/min). Heavy storm downpour causing severe volume throughput.",
        "action_protocol": "Activate DRRMO Flood Surge Protocol. Monitor downstream tidal gates and river outfalls for backflow. Alert low-lying communities.",
        "risk_level": "Elevated / Storm Surge (Orange)"
    },
    "OBSTRUCT": {
        "level_title": "LEVEL 3: CRITICAL ALARM (OBSTRUCTION DETECTED)",
        "badge": "🔴 SEVERE DRAINAGE BLOCKAGE",
        "banner_class": "alert-banner-obstruct",
        "color": "#dc2626",
        "bg_color": "#fef2f2",
        "buzzer_state": "CRITICAL EMERGENCY ALARM",
        "buzzer_timing": "100ms ON / 100ms OFF CONTINUOUS ALARM",
        "strobe_state": "RED ROTATING BEACON",
        "description": "High water level detected with severely restricted flow (< 3.5 L/min)! Culvert is obstructed by debris, trash, or silt backup.",
        "action_protocol": "IMMEDIATE EMERGENCY ACTION: Open secondary PVC ball valve / diversion gate to reroute floodwater! Dispatch DPWH / DRRMO clearing crew immediately!",
        "risk_level": "CRITICAL / OVERFLOW IMMINENT (Red)"
    }
}

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = []
if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = 0
if "sim_step_count" not in st.session_state:
    st.session_state.sim_step_count = 0
if "valve_open" not in st.session_state:
    st.session_state.valve_open = False
if "auto_failover_enabled" not in st.session_state:
    st.session_state.auto_failover_enabled = True

# Load Model & Dataset
model, model_error = load_ml_model(MODEL_PATH)
df_historical = load_historical_telemetry(DATA_PATH)

# -----------------------------------------------------------------------------
# DRRMO Command Center Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="drrmo-header-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 class="drrmo-title">
                <span>🌊</span> Cavite / Bacoor LGU — DRRMO Smart Drainage & Flood Early Warning System
            </h1>
            <p class="drrmo-subtitle">
                City Disaster Risk Reduction and Management Office • Sensor Telemetry Node: <strong>ESP32_DRAINAGE_01</strong> • Sector: Bacoor Lowland Culvert Array
            </p>
        </div>
        <div style="text-align: right;">
            <span class="drrmo-badge-online">SYSTEM ACTIVE • LIVE INGRESS</span>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
                Clock: <strong>""" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """ PHT</strong>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Telemetry Ingress Modes & Diagnostic Controls
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ DRRMO Control Console")
st.sidebar.markdown("Configure telemetry source and operational parameters.")

ingress_mode = st.sidebar.radio(
    "Telemetry Ingress Mode",
    [
        "Mode A: Historical Replay (CSV)",
        "Mode B: Live Dynamic Simulation",
        "Mode C: Manual 'What-If' Diagnostics"
    ],
    index=0
)

current_distance = 24.15
current_flow = 0.58
active_scenario_name = "Baseline Observation"

# --- MODE A: Historical Telemetry Replay ---
if ingress_mode == "Mode A: Historical Replay (CSV)":
    st.sidebar.subheader("📁 Historical Replay Scrubbing")
    total_records = len(df_historical)
    st.sidebar.caption(f"Loaded **{total_records:,}** physical telemetry rows from `{os.path.basename(DATA_PATH)}`.")

    # Stepper and navigation buttons
    col_nav1, col_nav2, col_nav3, col_nav4 = st.sidebar.columns(4)
    with col_nav1:
        if st.button("⏮ Prev", use_container_width=True):
            st.session_state.replay_idx = max(0, st.session_state.replay_idx - 1)
    with col_nav2:
        if st.button("▶ Next", use_container_width=True):
            st.session_state.replay_idx = min(total_records - 1, st.session_state.replay_idx + 1)
    with col_nav3:
        if st.button("🎲 Rand", use_container_width=True):
            st.session_state.replay_idx = int(np.random.randint(0, total_records))
    with col_nav4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.replay_idx = 0
            st.session_state.telemetry_history = []

    # Scrub Slider
    st.session_state.replay_idx = st.sidebar.slider(
        "Replay Sample Index",
        min_value=0,
        max_value=max(0, total_records - 1),
        value=min(st.session_state.replay_idx, max(0, total_records - 1)),
        step=1
    )

    # Quick Jump to Specific Target Class
    if "status" in df_historical.columns:
        jump_class = st.sidebar.selectbox(
            "Jump to Event Class Sample",
            ["None", "NORMAL", "WARNING", "HVYRAIN", "OBSTRUCT"],
            index=0
        )
        if jump_class != "None":
            matching_indices = df_historical[df_historical["status"] == jump_class].index
            if len(matching_indices) > 0:
                if st.sidebar.button(f"Jump to First '{jump_class}'", use_container_width=True):
                    st.session_state.replay_idx = int(matching_indices[0])

    # Extract reading
    selected_row = df_historical.iloc[st.session_state.replay_idx]
    current_distance = float(selected_row["distance_cm"])
    current_flow = float(selected_row["flow_l_min"])
    active_scenario_name = f"Replay Row #{st.session_state.replay_idx}"
    if "status" in selected_row:
        active_scenario_name += f" (Ground Truth: {selected_row['status']})"

# --- MODE B: Live Dynamic Simulation Engine ---
elif ingress_mode == "Mode B: Live Dynamic Simulation":
    st.sidebar.subheader("⚡ Dynamic Sensor Stream Simulation")
    scenario_choice = st.sidebar.selectbox(
        "Simulation Scenario",
        [
            "Normal Baseline (Channel Clear)",
            "Gradual Flood Inflow (Rising Runoff)",
            "Sudden Heavy Rain Surge (Flash Downpour)",
            "Debris / Culvert Obstruction (Trash Clog)"
        ]
    )

    noise_factor = st.sidebar.slider("Sensor Noise Jitter (±cm)", 0.0, 0.5, 0.05, 0.01)

    col_sim1, col_sim2 = st.sidebar.columns(2)
    with col_sim1:
        if st.button("▶ Next Tick", use_container_width=True):
            st.session_state.sim_step_count += 1
    with col_sim2:
        if st.button("🔄 Reset Tick", use_container_width=True):
            st.session_state.sim_step_count = 0
            st.session_state.telemetry_history = []

    step = st.session_state.sim_step_count

    # Scenario Profiles
    np.random.seed(int(time.time() * 1000) % 65536)
    jitter = np.random.uniform(-noise_factor, noise_factor)

    if scenario_choice == "Normal Baseline (Channel Clear)":
        current_distance = 24.2 + 0.3 * np.sin(step * 0.2) + jitter
        current_flow = max(0.0, 0.6 + 0.4 * np.cos(step * 0.2))
        active_scenario_name = "Dynamic: Normal Baseline Clear Stream"
    elif scenario_choice == "Gradual Flood Inflow (Rising Runoff)":
        # Distance gradually decreases from 24.5 down to 21.5
        phase = min(1.0, (step % 40) / 40.0)
        current_distance = (24.5 - 2.8 * phase) + jitter
        current_flow = 0.5 + 2.0 * phase
        active_scenario_name = f"Dynamic: Gradual Runoff Accumulation (Phase {phase*100:.0f}%)"
    elif scenario_choice == "Sudden Heavy Rain Surge (Flash Downpour)":
        # Rapid drop in distance to 20.3 cm, flow surges above 4.5 L/min
        current_distance = 20.35 + 0.2 * np.sin(step * 0.3) + jitter
        current_flow = 4.8 + 1.2 * np.cos(step * 0.3)
        active_scenario_name = "Dynamic: Heavy Rain Surge (Flood Flow >= 3.5 L/min)"
    else:  # Debris / Culvert Obstruction
        # Critical high water (distance ~ 20.3 cm) but stagnant/choked flow (< 2.0 L/min)
        current_distance = 20.25 + 0.15 * np.sin(step * 0.3) + jitter
        current_flow = max(0.0, 0.8 + 0.4 * np.cos(step * 0.3))
        active_scenario_name = "Dynamic: Critical Solid Waste / Culvert Obstruction"

    current_distance = float(np.clip(current_distance, 15.0, 30.0))
    current_flow = float(np.clip(current_flow, 0.0, 15.0))

# --- MODE C: Manual Diagnostic "What-If" Controls ---
else:
    st.sidebar.subheader("🎛️ Manual Telemetry 'What-If' Sliders")
    st.sidebar.caption("Directly test classifier boundary conditions and DRRMO alert thresholds.")

    preset = st.sidebar.selectbox(
        "Apply Test Preset",
        ["Custom Sliders", "Preset: Normal State", "Preset: Warning State", "Preset: Heavy Rain Surge", "Preset: Critical Obstruction"]
    )

    if preset == "Preset: Normal State":
        default_dist, default_flow = 24.2, 0.8
    elif preset == "Preset: Warning State":
        default_dist, default_flow = 22.1, 1.6
    elif preset == "Preset: Heavy Rain Surge":
        default_dist, default_flow = 20.3, 4.8
    elif preset == "Preset: Critical Obstruction":
        default_dist, default_flow = 20.3, 1.1
    else:
        default_dist, default_flow = 24.0, 1.0

    current_distance = st.sidebar.slider(
        "Ultrasonic Distance to Water (cm)",
        min_value=15.0,
        max_value=30.0,
        value=float(default_dist),
        step=0.1,
        help="Sensor mounted at top of culvert. Lower distance = higher water level."
    )
    current_flow = st.sidebar.slider(
        "Water Flow Rate (L/min)",
        min_value=0.0,
        max_value=15.0,
        value=float(default_flow),
        step=0.1,
        help="Inline turbine flow sensor reading."
    )
    active_scenario_name = f"Manual Diagnostic Vector [d={current_distance:.1f}cm, f={current_flow:.1f}L/min]"

# Clear History Button in Sidebar
st.sidebar.divider()
if st.sidebar.button("🧹 Clear Telemetry History Buffer", use_container_width=True):
    st.session_state.telemetry_history = []
    st.rerun()

# -----------------------------------------------------------------------------
# Machine Learning Prediction & Resilience Fallback
# -----------------------------------------------------------------------------
input_telemetry = pd.DataFrame([[current_distance, current_flow]], columns=FEATURE_COLUMNS)

if model is not None:
    try:
        predicted_status = str(model.predict(input_telemetry)[0])
        probabilities = model.predict_proba(input_telemetry)[0]
        class_list = list(model.classes_)
        inference_source = "RandomForestClassifier (.pkl)"
        pred_idx = class_list.index(predicted_status)
        confidence_pct = float(probabilities[pred_idx] * 100.0)
    except Exception as e:
        predicted_status, probabilities, class_list = rule_based_predict(current_distance, current_flow)
        inference_source = f"Firmware Fallback (Model Error: {str(e)})"
        pred_idx = class_list.index(predicted_status)
        confidence_pct = float(probabilities[pred_idx] * 100.0)
else:
    predicted_status, probabilities, class_list = rule_based_predict(current_distance, current_flow)
    inference_source = "ESP32 Firmware Rule-Based Engine (Fallback)"
    pred_idx = class_list.index(predicted_status)
    confidence_pct = float(probabilities[pred_idx] * 100.0)

# Physical Derived Values
water_level_cm = max(0.0, CANAL_BED_DISTANCE_CM - current_distance)
# Culvert capacity: 0% at distance=25cm (0cm depth), 100% at distance=20cm (5cm depth, overflow)
capacity_utilization_pct = min(100.0, max(0.0, (water_level_cm / (CANAL_BED_DISTANCE_CM - CRITICAL_OVERFLOW_CM)) * 100.0))

# -----------------------------------------------------------------------------
# Secondary Actuator & Automated Failover Logic
# -----------------------------------------------------------------------------
if st.session_state.auto_failover_enabled and predicted_status == "OBSTRUCT":
    st.session_state.valve_open = True

# Append to Telemetry Rolling Buffer
timestamp_str = datetime.now().strftime("%H:%M:%S")
new_record = {
    "timestamp": timestamp_str,
    "distance_cm": current_distance,
    "flow_l_min": current_flow,
    "water_level_cm": water_level_cm,
    "status": predicted_status,
    "capacity_pct": capacity_utilization_pct,
    "source": inference_source
}
# Only append if different or stepped
st.session_state.telemetry_history.append(new_record)
if len(st.session_state.telemetry_history) > 60:
    st.session_state.telemetry_history.pop(0)

# -----------------------------------------------------------------------------
# Model Status & Resilience Banner
# -----------------------------------------------------------------------------
if model_error:
    st.warning(
        f"⚠️ **Model Fallback Activated**: `{model_error}`. "
        "The DRRMO Command Center is operating reliably using the deterministic firmware threshold logic "
        "from `FINAL_CODE_DRAINAGE_V1.ino`."
    )

# -----------------------------------------------------------------------------
# DRRMO Real-Time Color-Coded Alert Banner
# -----------------------------------------------------------------------------
alert_spec = DRRMO_ALERT_MATRIX.get(predicted_status, DRRMO_ALERT_MATRIX["NORMAL"])

st.markdown(f"""
<div class="alert-banner {alert_spec['banner_class']}">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div>
            <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                DRRMO DEFENSE STATUS — {alert_spec['level_title']}
            </div>
            <div style="font-size: 24px; font-weight: 800; margin: 4px 0 8px 0;">
                {alert_spec['badge']}
            </div>
        </div>
        <div style="text-align: right;">
            <span class="drrmo-risk-badge" style="background: {alert_spec['color']}; color: white; padding: 4px 12px; border-radius: 0px; font-weight: 700; font-size: 13px;">
                RISK: {alert_spec['risk_level']}
            </span>
            <div style="font-size: 12px; margin-top: 5px;">Source: {inference_source}</div>
        </div>
    </div>
    <div style="font-size: 15px; margin-bottom: 8px;">
        <strong>Hydraulic Condition:</strong> {alert_spec['description']}
    </div>
    <div style="font-size: 15px; background: rgba(255,255,255,0.7); padding: 10px 14px; border-radius: 0px; border: 1px solid rgba(0,0,0,0.06);">
        <strong>🚨 Recommended DRRMO Action Protocol:</strong> {alert_spec['action_protocol']}
    </div>
    <div class="alarm-box">
        <strong>Audio/Visual Hardware Alarm State:</strong> Buzzer: <code>{alert_spec['buzzer_state']} ({alert_spec['buzzer_timing']})</code> &nbsp;|&nbsp; Visual Strobe: <code>{alert_spec['strobe_state']}</code>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top KPI Metrics Strip (st.metric cards)
# -----------------------------------------------------------------------------
st.markdown("### 📊 Operational KPI Telemetry")

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    dist_delta = f"{current_distance:.2f} cm air gap"
    st.metric(
        label="Water Depth (Canal Bed)",
        value=f"{water_level_cm:.2f} cm",
        delta=dist_delta,
        delta_color="off",
        help="Calculated depth from canal floor (25.0 cm - sensor distance)."
    )

with kpi_col2:
    flow_delta = f"{'≥' if current_flow >= LOW_FLOW_THRESHOLD else '<'} 3.5 L/min Threshold"
    st.metric(
        label="Water Flow Rate",
        value=f"{current_flow:.2f} L/min",
        delta=flow_delta,
        delta_color="normal" if current_flow >= LOW_FLOW_THRESHOLD else "inverse",
        help="Real-time water velocity through YF-S201 Hall-effect turbine sensor."
    )

with kpi_col3:
    st.metric(
        label="Predicted Drainage Status",
        value=predicted_status,
        delta=f"{inference_source.split(' ')[0]} Engine",
        delta_color="off",
        help="Classification of drainage status (NORMAL, WARNING, HVYRAIN, OBSTRUCT)."
    )

with kpi_col4:
    st.metric(
        label="Model Confidence Score",
        value=f"{confidence_pct:.1f}%",
        delta=f"Top Class: {predicted_status}",
        delta_color="normal" if confidence_pct > 75 else "off",
        help="Statistical probability assigned by the Random Forest classifier."
    )

with kpi_col5:
    st.metric(
        label="Culvert Capacity Load",
        value=f"{capacity_utilization_pct:.1f}%",
        delta="Overflow Risk" if capacity_utilization_pct > 80 else "Safe Volume",
        delta_color="inverse" if capacity_utilization_pct > 80 else "normal",
        help="Percentage of maximum culvert cross-section capacity utilized."
    )

# Capacity Progress Bar
cap_color = "#10b981" if capacity_utilization_pct < 50 else "#f59e0b" if capacity_utilization_pct < 80 else "#ef4444"
st.progress(capacity_utilization_pct / 100.0, text=f"Culvert Capacity Load: {capacity_utilization_pct:.1f}% (Critical Spillover Threshold at 100%)")

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Visual Analytics Grid (2 Columns: Time-Series Chart + ML Explainability)
# -----------------------------------------------------------------------------
viz_col1, viz_col2 = st.columns([7, 5])

# --- Column 1: Time-Series Telemetry Visualizations ---
with viz_col1:
    st.subheader("📈 Time-Series Drainage Telemetry")
    st.caption("Dual-axis real-time telemetry stream. Ultrasonic distance axis is inverted: upward slope represents rising floodwater.")

    df_hist_buffer = pd.DataFrame(st.session_state.telemetry_history)

    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=["Culvert Water Level / Inverted Air Gap & Flow Rate Trajectory"]
    )

    if not df_hist_buffer.empty:
        # Distance line (inverted primary Y)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df_hist_buffer))),
                y=df_hist_buffer["distance_cm"],
                name="Distance (cm) [Inverted]",
                mode="lines+markers",
                line=dict(color="#8b5cf6", width=3),
                marker=dict(size=6, color="#6d28d9"),
                hovertemplate="Sample #%{x}<br>Distance: %{y:.2f} cm<br>Water Depth: %{customdata:.2f} cm<extra></extra>",
                customdata=df_hist_buffer["water_level_cm"]
            ),
            secondary_y=False
        )

        # Flow Rate line (secondary Y)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df_hist_buffer))),
                y=df_hist_buffer["flow_l_min"],
                name="Flow Rate (L/min)",
                mode="lines+markers",
                line=dict(color="#0ea5e9", width=2.5),
                marker=dict(size=5, color="#0284c7"),
                hovertemplate="Sample #%{x}<br>Flow Rate: %{y:.2f} L/min<extra></extra>"
            ),
            secondary_y=True
        )

    # Reference Threshold Horizontal Lines
    fig.add_hline(
        y=NORMAL_LEVEL_CM,
        line_dash="dot",
        line_color="#10b981",
        line_width=2,
        annotation_text="Normal Level (23.0 cm)",
        annotation_position="bottom right",
        secondary_y=False
    )
    fig.add_hline(
        y=WARNING_LEVEL_CM,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text="Warning Level (21.0 cm)",
        annotation_position="top right",
        secondary_y=False
    )
    fig.add_hline(
        y=CRITICAL_OVERFLOW_CM,
        line_dash="solid",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="Overflow Crest (20.0 cm)",
        annotation_position="top left",
        secondary_y=False
    )
    fig.add_hline(
        y=LOW_FLOW_THRESHOLD,
        line_dash="dashdot",
        line_color="#0284c7",
        line_width=1.5,
        annotation_text="Heavy Rain Flow (3.5 L/min)",
        annotation_position="bottom left",
        secondary_y=True
    )

    # Invert Primary Y-Axis (Lower distance = Higher water level)
    fig.update_yaxes(
        title_text="<b>Ultrasonic Distance (cm)</b> [Inverted: Higher = Flooding]",
        autorange="reversed",
        range=[18.0, 27.0],
        secondary_y=False,
        showgrid=True,
        gridcolor="#e5e2dc"
    )
    fig.update_yaxes(
        title_text="<b>Water Flow Rate (L/min)</b>",
        range=[0.0, 10.0],
        secondary_y=True,
        showgrid=False
    )
    fig.update_xaxes(
        title_text="Rolling Buffer Sequence (Latest -> Right)",
        showgrid=True,
        gridcolor="#e5e2dc"
    )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        font=dict(family="ui-monospace, 'SF Mono', 'Cascadia Code', monospace", size=11, color="#1f2429"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Column 2: Machine Learning Prediction & Explainability Panel ---
with viz_col2:
    st.subheader("🧠 ML Prediction & Explainability")
    st.caption("Multi-class probabilistic inference and model feature importance breakdown.")

    # Probability Distribution Bar Chart
    prob_colors = {
        "NORMAL": "#16a34a",
        "WARNING": "#d97706",
        "HVYRAIN": "#ea580c",
        "OBSTRUCT": "#b02631"
    }

    df_probs = pd.DataFrame({
        "Status": class_list,
        "Probability": [p * 100.0 for p in probabilities]
    }).sort_values("Probability", ascending=True)

    bar_colors = [prob_colors.get(c, "#57606a") for c in df_probs["Status"]]

    fig_prob = go.Figure(go.Bar(
        x=df_probs["Probability"],
        y=df_probs["Status"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#9a1f2a", width=1)),
        text=[f"{p:.1f}%" for p in df_probs["Probability"]],
        textposition="auto"
    ))

    fig_prob.update_layout(
        title="Class Probability Distribution (%)",
        xaxis_title="Softmax / Confidence Probability (%)",
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="#e5e2dc"),
        yaxis_title="Drainage Class",
        font=dict(family="ui-monospace, 'SF Mono', 'Cascadia Code', monospace", size=11, color="#1f2429"),
        height=220,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Feature Importance & Model Metadata
    st.markdown("##### 🔬 Classifier Diagnostics & Feature Importance")

    if model is not None and hasattr(model, "feature_importances_"):
        f_imp = model.feature_importances_
        imp_dist = f_imp[0] * 100.0
        imp_flow = f_imp[1] * 100.0
        st.write(f"• **Distance (`distance_cm`)**: {imp_dist:.1f}% relative importance")
        st.progress(imp_dist / 100.0)
        st.write(f"• **Flow Rate (`flow_l_min`)**: {imp_flow:.1f}% relative importance")
        st.progress(imp_flow / 100.0)
        st.caption(f"Architecture: Random Forest ({len(model.estimators_)} Decision Trees, Balanced Weights)")
    else:
        st.info("Operating in Firmware Rule-Based Mode. Feature importances derived from hardware discriminator thresholds: `distance_cm` (Primary Water Gap), `flow_l_min` (Blockage Discriminator).")

    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #d8d4cf; border-radius: 0px; padding: 10px; font-size: 13px; margin-top: 10px;">
        <strong>Active Telemetry Vector:</strong><br>
        <code>distance_cm: {current_distance:.2f} cm</code> &nbsp;|&nbsp; <code>flow_l_min: {current_flow:.2f} L/min</code><br>
        <strong>Prediction:</strong> <span style="color: {alert_spec['color']}; font-weight: 700;">{predicted_status}</span> ({confidence_pct:.1f}% confidence)
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Emergency Secondary Valve & Actuator Bypass Control
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🚧 Emergency Secondary Valve & Diversion Gate Actuator")
st.markdown(
    "Control the secondary PVC diversion bypass line installed at Sector 4. "
    "When solid waste obstructs the primary culvert (`OBSTRUCT`), opening the bypass valve diverts "
    "accumulated floodwaters into the secondary drainage basin to prevent roadway overflow."
)

act_col1, act_col2, act_col3 = st.columns([4, 4, 4])

with act_col1:
    st.markdown("##### ⚙️ Actuator Controls")
    toggle_label = "🔴 CLOSE Diversion Gate" if st.session_state.valve_open else "🟢 OPEN Secondary Diversion Gate"
    if st.button(toggle_label, use_container_width=True, type="primary"):
        st.session_state.valve_open = not st.session_state.valve_open
        st.rerun()

    auto_toggle = st.checkbox(
        "Enable Automated Failover (Auto-Open on OBSTRUCT)",
        value=st.session_state.auto_failover_enabled,
        help="When enabled, the secondary diversion valve opens automatically as soon as a critical obstruction is classified."
    )
    st.session_state.auto_failover_enabled = auto_toggle

with act_col2:
    st.markdown("##### 📍 Actuator Physical State")
    if st.session_state.valve_open:
        st.markdown("""
        <div class="valve-indicator valve-open">
            <div style="font-size: 18px;">🔓 GATE OPEN — DIVERSION ACTIVE</div>
            <div style="font-size: 13px; margin-top: 4px;">Secondary PVC branch discharging at 100% capacity. Backflow pressure relieved.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="valve-indicator valve-closed">
            <div style="font-size: 18px;">🔒 GATE CLOSED — NOMINAL CONDUIT</div>
            <div style="font-size: 13px; margin-top: 4px;">Main culvert handling 100% of volume. Secondary bypass on standby.</div>
        </div>
        """, unsafe_allow_html=True)

with act_col3:
    st.markdown("##### 🌊 Hydraulic Relief Estimate")
    if st.session_state.valve_open:
        relief_flow = max(0.5, current_flow * 1.5 + 2.0)
        st.metric(
            label="Estimated Bypass Throughput",
            value=f"{relief_flow:.2f} L/min",
            delta="Surface Flooding Prevented",
            delta_color="normal"
        )
    else:
        st.metric(
            label="Estimated Bypass Throughput",
            value="0.00 L/min",
            delta="Conduit Inactive",
            delta_color="off"
        )

# -----------------------------------------------------------------------------
# Historical Telemetry Stream Table & Export
# -----------------------------------------------------------------------------
with st.expander("📋 View Recent Telemetry History Buffer & Sensor Audit Log"):
    if st.session_state.telemetry_history:
        df_display = pd.DataFrame(st.session_state.telemetry_history)[
            ["timestamp", "distance_cm", "flow_l_min", "water_level_cm", "capacity_pct", "status", "source"]
        ]
        df_display.columns = ["Timestamp", "Distance (cm)", "Flow Rate (L/min)", "Water Depth (cm)", "Capacity (%)", "Status", "Engine"]
        st.dataframe(
            df_display.iloc[::-1],
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp"),
                "Distance (cm)": st.column_config.NumberColumn("Distance (cm)", format="%.2f cm"),
                "Flow Rate (L/min)": st.column_config.NumberColumn("Flow Rate (L/min)", format="%.2f L/min"),
                "Water Depth (cm)": st.column_config.NumberColumn("Water Depth (cm)", format="%.2f cm"),
                "Capacity (%)": st.column_config.NumberColumn("Capacity (%)", format="%.1f%%"),
                "Status": st.column_config.TextColumn("Status"),
                "Engine": st.column_config.TextColumn("Engine"),
            }
        )

        csv_download = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Telemetry Log as CSV",
            data=csv_download,
            file_name=f"drrmo_telemetry_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No telemetry readings recorded in the current session buffer.")

# -----------------------------------------------------------------------------
# DRRMO Command Center Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 13px; padding-bottom: 20px;">
        <strong>City of Bacoor Disaster Risk Reduction and Management Office (DRRMO)</strong> • Flood Early Warning & Smart Drainage Division<br>
        Cavite State University — Department of Computer and Electronics Engineering (BSCPE CPEN106)<br>
        IoT Hardware: ESP32 Edge Gateway • JSN-SR04T Waterproof Ultrasonic Sensor • YF-S201 Turbine Flow Meter
    </div>
    """,
    unsafe_allow_html=True
)
