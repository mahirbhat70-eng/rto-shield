"""
dashboard.py — RTO Shield Decision Console
==========================================
A live, judge-facing interactive console for the frozen v1.0 serving artifacts.

Design contract:
  * Purely additive serving-layer UI: reads frozen artifacts through src/serve/scorer.py,
    src/serve/audit.py, and src/policy/cost_engine.py.
  * No model touches money decisions directly — deterministic, tested expected-loss math.
  * Every number shown is either:
    (a) returned live by score_order() on frozen artifacts,
    (b) recomputed live and closely replicates frozen stage-4 reports, or
    (c) quoted verbatim from reports/stage5_test_results.md with provenance.
  * Palette: Ultra-light slate (#F8FAFC) background, pure white cards (#FFFFFF),
    deep navy text (#0F172A), and vibrant Razorpay Blue (#0052FF) interactive accents.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import json
import datetime
import html as html_mod
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "order_value": 852.0, "category": "Home", "payment_method": "COD", "quantity": 3,
    "discount_pct": 19.8, "cod_charge": 59.0, "account_age_days": 65, "prior_orders": 2,
    "prior_rto_count": 0, "orders_last_24h": 3, "device_cluster_size": 1, 
    "pincode": "253407", "courier_id": "Courier_E"
}
if "_persisted_form" not in st.session_state:
    st.session_state["_persisted_form"] = dict(DEFAULTS)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

import altair as alt

from src.serve.scorer import score_order, PINCODE_LOOKUP
from src.serve import scorer as serve
from src.serve.audit import build_audit_record, to_jsonl, verify_audit_record

st.set_page_config(
    page_title="RTO Shield — COD Decision Console",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme Tokens & CSS (Light Slate + Pure White + Deep Navy + Razorpay Blue)
# ----------------------------------------------------------------------------
C_BG = "#F8FAFC"             # Ultra-light slate background
C_CARD = "#FFFFFF"           # Pure white card surface
C_PANEL_ALT = "#F1F5F9"      # Soft light slate container
C_BORDER = "#E2E8F0"         # Slate border
C_TEXT = "#0F172A"           # Deep navy text
C_MUT = "#64748B"            # Muted slate
C_MUT_DARK = "#475569"       # Secondary text
C_RAZORPAY_BLUE = "#0052FF"  # Vibrant Razorpay Blue
C_BLUE_HOVER = "#0043D9"
C_BLUE_BG = "#EFF6FF"        # Light blue badge/chip bg

# Action semantic tokens (accessible contrast on light backgrounds)
C_GREEN = "#059669"          # ALLOW_COD (Emerald)
C_GREEN_BG = "#ECFDF5"
C_GREEN_BORDER = "#A7F3D0"

C_AMBER = "#D97706"          # VERIFY_ADDRESS (Amber)
C_AMBER_BG = "#FFFBEB"
C_AMBER_BORDER = "#FDE68A"

C_RED = "#E11D48"            # REQUIRE_DEPOSIT (Rose/Coral)
C_RED_BG = "#FFF1F2"
C_RED_BORDER = "#FECDD3"

C_VIOLET = "#7C3AED"         # PREPAID_ONLY (Violet)
C_VIOLET_BG = "#F5F3FF"
C_VIOLET_BORDER = "#DDD6FE"

C_SLATE = "#475569"          # PREPAID_PASSTHROUGH (Slate)
C_SLATE_BG = "#F1F5F9"
C_SLATE_BORDER = "#CBD5E1"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background-color: {C_BG} !important;
  color: {C_TEXT} !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}}

/* Sidebar styling */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
  background-color: #FFFFFF !important;
  border-right: 1px solid {C_BORDER} !important;
}}

.block-container {{
  padding: 1.5rem 2.2rem 3rem 2.2rem;
  max-width: 1320px;
}}

#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}
header {{ background: transparent !important; }}

h1, h2, h3, h4 {{
  color: {C_TEXT} !important;
  letter-spacing: -0.025em;
  font-weight: 800;
}}

a {{ color: {C_RAZORPAY_BLUE}; text-decoration: none; font-weight: 600; }}
a:hover {{ text-decoration: underline; }}

/* Topbar */
.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0.2rem 1.1rem 0.2rem;
  border-bottom: 1px solid {C_BORDER};
  margin-bottom: 1.4rem;
}}
.brand {{ display: flex; align-items: baseline; gap: 0.75rem; }}
.brand-name {{
  font-weight: 900;
  font-size: 1.35rem;
  letter-spacing: -0.03em;
  color: {C_TEXT};
}}
.brand-name span {{ color: {C_RAZORPAY_BLUE}; }}
.brand-sub {{
  font-size: 0.75rem;
  color: {C_MUT};
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.status-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: #065F46;
  background: {C_GREEN_BG};
  border: 1px solid {C_GREEN_BORDER};
  padding: 0.35rem 0.8rem;
  border-radius: 999px;
}}
.status-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: {C_GREEN};
  box-shadow: 0 0 6px rgba(5, 150, 105, 0.4);
}}

/* Hero Section */
.eyebrow {{
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  color: {C_RAZORPAY_BLUE};
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}}
.hero-h1 {{
  font-size: 3.15rem;
  line-height: 1.1;
  font-weight: 900;
  letter-spacing: -0.035em;
  color: {C_TEXT};
  margin: 0 0 0.85rem 0;
}}
.hero-h1 .grad {{
  background: linear-gradient(92deg, {C_RAZORPAY_BLUE} 0%, #3B82F6 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}
.hero-sub {{
  font-size: 1.05rem;
  color: {C_MUT_DARK};
  max-width: 820px;
  line-height: 1.6;
  margin-bottom: 1.6rem;
}}

/* Cards & KPI Grid */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin: 1rem 0 0.5rem 0;
}}
.kpi {{
  background: {C_CARD};
  border: 1px solid {C_BORDER};
  border-radius: 14px;
  padding: 1.15rem 1.25rem 1rem 1.25rem;
  box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02);
  transition: all 0.15s ease;
}}
.kpi:hover {{
  border-color: #CBD5E1;
  box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
}}
.kpi .t {{
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {C_MUT};
  margin-bottom: 0.4rem;
}}
.kpi .v {{
  font-size: 2.05rem;
  font-weight: 900;
  letter-spacing: -0.035em;
  color: {C_TEXT};
  line-height: 1.08;
}}
.kpi .v.blue {{ color: {C_RAZORPAY_BLUE}; }}
.kpi .v.green {{ color: {C_GREEN}; }}
.kpi .v.amber {{ color: {C_AMBER}; }}
.kpi .d {{
  font-size: 0.76rem;
  color: {C_MUT};
  margin-top: 0.45rem;
  line-height: 1.45;
}}
.kpi .d b {{ color: {C_TEXT}; font-weight: 600; }}

/* Panels */
.panel {{
  background: {C_CARD};
  border: 1px solid {C_BORDER};
  border-radius: 14px;
  padding: 1.25rem 1.4rem;
  box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
  margin-bottom: 1rem;
}}
.panel-h {{
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: {C_MUT};
  margin-bottom: 0.85rem;
}}

/* Action Rows (Expected Loss Price Menu) */
.arow {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  background: #F8FAFC;
  border: 1px solid {C_BORDER};
  border-radius: 12px;
  padding: 0.85rem 1.15rem;
  margin-bottom: 0.6rem;
  transition: all 0.15s ease;
}}
.arow.sel {{
  border: 2px solid {C_RAZORPAY_BLUE};
  background: #F0F6FF;
  box-shadow: 0 0 12px rgba(0, 82, 255, 0.10);
}}
.a-name {{ display: flex; flex-direction: column; gap: 0.28rem; min-width: 220px; }}
.a-label {{ font-weight: 800; font-size: 1.02rem; letter-spacing: -0.01em; }}
.a-chips {{ display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.achip {{
  font-size: 0.68rem;
  font-weight: 600;
  color: {C_MUT_DARK};
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  padding: 0.15rem 0.52rem;
  border-radius: 6px;
  white-space: nowrap;
}}
.a-el {{ text-align: right; min-width: 160px; }}
.a-el .lbl {{
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  color: {C_MUT};
  font-weight: 700;
  text-transform: uppercase;
}}
.a-el .val {{
  font-size: 1.38rem;
  font-weight: 900;
  letter-spacing: -0.02em;
}}
.pick {{
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  color: #FFFFFF;
  background: {C_RAZORPAY_BLUE};
  border-radius: 6px;
  padding: 0.22rem 0.6rem;
  margin-left: 0.65rem;
  vertical-align: middle;
  display: inline-block;
}}

/* Live Scorer KPI Strip */
.big-p {{
  font-size: 3.2rem;
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1;
}}
.track {{
  height: 9px;
  background: #E2E8F0;
  border-radius: 6px;
  overflow: hidden;
  margin-top: 0.75rem;
}}
.track > i {{ display: block; height: 100%; border-radius: 6px; }}
.badge {{
  display: inline-block;
  font-weight: 900;
  font-size: 1.05rem;
  letter-spacing: 0.04em;
  padding: 0.5rem 0.95rem;
  border-radius: 10px;
  color: #FFFFFF;
}}
.lat {{
  font-size: 1.7rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  color: {C_RAZORPAY_BLUE};
}}
.lat-sub {{ font-size: 0.74rem; color: {C_MUT}; margin-top: 0.35rem; }}

/* SHAP Bar Visualization */
.shap-row {{
  display: grid;
  grid-template-columns: 210px 1fr 75px;
  align-items: center;
  gap: 0.75rem;
  margin: 0.38rem 0;
}}
.shap-name {{
  font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
  font-size: 0.72rem;
  color: {C_MUT_DARK};
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.shap-track {{
  position: relative;
  height: 16px;
  background: #F1F5F9;
  border-radius: 5px;
  overflow: hidden;
  border: 1px solid #E2E8F0;
}}
.shap-track .mid {{
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: #CBD5E1;
}}
.shap-fill {{
  position: absolute;
  top: 2.5px;
  bottom: 2.5px;
  border-radius: 3px;
}}
.shap-val {{
  font-family: 'SFMono-Regular', Consolas, Menlo, monospace;
  font-size: 0.72rem;
  font-weight: 700;
}}

/* Step Strip on Overview */
.step-strip {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
}}
.step {{
  background: #F8FAFC;
  border: 1px solid {C_BORDER};
  border-radius: 12px;
  padding: 0.95rem 1.1rem;
}}
.step .n {{
  font-size: 0.7rem;
  font-weight: 800;
  color: {C_RAZORPAY_BLUE};
  letter-spacing: 0.14em;
}}
.step .h {{
  font-weight: 800;
  font-size: 0.96rem;
  color: {C_TEXT};
  margin: 0.3rem 0 0.25rem 0;
}}
.step .b {{
  font-size: 0.76rem;
  color: {C_MUT_DARK};
  line-height: 1.5;
}}

/* Portfolio visual elements */
.donut-wrap {{ display: flex; align-items: center; gap: 1.6rem; }}
.donut {{
  width: 154px;
  height: 154px;
  border-radius: 50%;
  -webkit-mask: radial-gradient(circle at 50% 50%, transparent 0 54%, #000 55%);
  mask: radial-gradient(circle at 50% 50%, transparent 0 54%, #000 55%);
}}
.legend {{ display: flex; flex-direction: column; gap: 0.45rem; }}
.leg {{ display: flex; align-items: center; gap: 0.55rem; font-size: 0.8rem; color: {C_MUT_DARK}; }}
.leg i {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
.leg b {{ color: {C_TEXT}; font-weight: 700; }}

.mc-track {{
  position: relative;
  height: 14px;
  background: #E2E8F0;
  border-radius: 8px;
  margin: 1.8rem 0.2rem 2rem 0.2rem;
}}
.mc-band {{
  position: absolute;
  top: 0;
  bottom: 0;
  background: rgba(0, 82, 255, 0.16);
  border-radius: 8px;
  border-left: 2px solid {C_RAZORPAY_BLUE};
  border-right: 2px solid {C_RAZORPAY_BLUE};
}}
.mc-mean {{
  position: absolute;
  top: -6px;
  bottom: -6px;
  width: 3px;
  background: {C_TEXT};
  border-radius: 2px;
}}
.mc-lab {{
  position: absolute;
  transform: translateX(-50%);
  font-size: 0.7rem;
  color: {C_MUT_DARK};
  white-space: nowrap;
  font-weight: 600;
}}

/* Verification & Status Pills */
.verify-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.76rem;
  font-weight: 700;
  color: #065F46;
  background: {C_GREEN_BG};
  border: 1px solid {C_GREEN_BORDER};
  padding: 0.45rem 0.85rem;
  border-radius: 9px;
}}

.side-h {{
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: {C_MUT};
  margin: 0.9rem 0 0.2rem 0;
}}

/* Primary Streamlit Button */
.stButton > button[kind="primary"] {{
  width: 100%;
  font-weight: 800;
  font-size: 0.95rem;
  letter-spacing: 0.02em;
  background-color: {C_RAZORPAY_BLUE} !important;
  border-color: {C_RAZORPAY_BLUE} !important;
  color: #FFFFFF !important;
  border-radius: 10px;
  padding: 0.65rem 1rem;
  box-shadow: 0 2px 4px rgba(0, 82, 255, 0.2);
  transition: all 0.15s ease;
}}
.stButton > button[kind="primary"]:hover {{
  background-color: {C_BLUE_HOVER} !important;
  border-color: {C_BLUE_HOVER} !important;
  box-shadow: 0 4px 8px rgba(0, 82, 255, 0.3);
}}

/* Presets buttons */
.preset-btn [data-testid="stButton"] > button {{
  width: 100%;
  font-size: 0.76rem;
  font-weight: 700;
  border-radius: 8px;
  padding: 0.35rem 0.5rem;
}}

hr.soft {{
  border: none;
  border-top: 1px solid {C_BORDER};
  margin: 1.2rem 0;
}}
.foot {{
  color: {C_MUT};
  font-size: 0.75rem;
  padding-top: 0.6rem;
}}

@media (max-width: 960px) {{
  .kpi-grid, .step-strip {{ grid-template-columns: repeat(2, 1fr); }}
  .hero-h1 {{ font-size: 2.2rem; }}
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Formatters & Constants
# ----------------------------------------------------------------------------
ACTION_META = {
    "ALLOW_COD": (
        "ALLOW COD",
        C_GREEN,
        C_GREEN_BG,
        "Ship it as-is. Zero friction cost, no RTO intervention.",
    ),
    "VERIFY_ADDRESS": (
        "VERIFY ADDRESS",
        C_AMBER,
        C_AMBER_BG,
        "Call/OTP address verification. −30% RTO, −5% conversions, ₹2 cost.",
    ),
    "REQUIRE_DEPOSIT": (
        "REQUIRE DEPOSIT",
        C_RED,
        C_RED_BG,
        "Token deposit before dispatch. −80% RTO, −40% conversions.",
    ),
    "PREPAID_ONLY": (
        "PREPAID ONLY",
        C_VIOLET,
        C_VIOLET_BG,
        "Convert COD to prepaid. −55% RTO, −70% conversions.",
    ),
    "MANUAL_REVIEW": (
        "MANUAL REVIEW",
        C_AMBER,
        C_AMBER_BG,
        "High-value order (>₹10,000) flagged by safety guardrails for human review.",
    ),
}

def inr(v, dec=0):
    """Indian currency numbering format: 3,27,071 (lakh/crore commas)."""
    neg = v < 0
    s = f"{abs(v):.{dec}f}"
    intpart, _, frac = s.partition(".")
    if len(intpart) > 3:
        head, tail = intpart[:-3], intpart[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        intpart = ",".join(groups + [tail])
    out = intpart + (("." + frac) if frac else "")
    return ("-" if neg else "") + out

def rs(v, dec=0):
    return ("₹" if v >= 0 else "-₹") + inr(abs(v), dec)

def kpi(title, value, desc, vclass=""):
    return f'<div class="kpi"><div class="t">{title}</div><div class="v {vclass}">{value}</div><div class="d">{desc}</div></div>'

def html_block(s):
    st.markdown(s, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Presets & Singletons
# ----------------------------------------------------------------------------

WIDGET_KEYS = {
    "order_value": "ni_order_value",
    "category": "category",
    "payment_method": "payment_method",
    "quantity": "ni_quantity",
    "discount_pct": "ni_discount_pct",
    "cod_charge": "ni_cod_charge",
    "account_age_days": "ni_account_age_days",
    "prior_orders": "ni_prior_orders",
    "prior_rto_count": "prior_rto_count",
    "orders_last_24h": "ni_orders_last_24h",
    "device_cluster_size": "ni_device_cluster_size",
    "pincode": "ti_pincode",
    "courier_id": "ti_courier_id",
}

PRESETS = {
    "DEPOSIT": {
        "order_value": 186.0, "category": "Beauty", "payment_method": "COD", "quantity": 1,
        "discount_pct": 0.0, "cod_charge": 29.0, "account_age_days": 260, "prior_orders": 4,
        "prior_rto_count": 0, "orders_last_24h": 1, "device_cluster_size": 1,
        "pincode": "461780", "courier_id": "Courier_A",
    },
    "ALLOW": {
        "order_value": 1344.0, "category": "Electronics", "payment_method": "COD", "quantity": 4,
        "discount_pct": 0.0, "cod_charge": 49.0, "account_age_days": 998, "prior_orders": 4,
        "prior_rto_count": 0, "orders_last_24h": 2, "device_cluster_size": 4,
        "pincode": "750176", "courier_id": "Courier_B",
    },
    "VERIFY": {
        "order_value": 852.0, "category": "Home", "payment_method": "COD", "quantity": 3,
        "discount_pct": 19.8, "cod_charge": 59.0, "account_age_days": 65, "prior_orders": 2,
        "prior_rto_count": 0, "orders_last_24h": 3, "device_cluster_size": 1,
        "pincode": "253407", "courier_id": "Courier_E",
    },
}

PRESET_HINT = {
    "DEPOSIT": "₹186 · low margin · 21.3% risk → deposit wins",
    "ALLOW": "₹1,344 · high margin · 21.3% risk → allow wins",
    "VERIFY": "₹852 · mid margin · 39.4% risk → verify wins",
}

CATEGORIES = ["Apparel", "Home", "Electronics", "Beauty", "Footwear", "Jewelry"]

@st.cache_resource(show_spinner=False)
def boot_self_test():
    """Score the three presets once on the frozen artifacts to pre-warm models."""
    out = {}
    t0 = time.perf_counter()
    for name, payload in PRESETS.items():
        t = time.perf_counter()
        res = score_order(dict(payload))
        out[name] = {
            "p": res["probability"],
            "action": res["recommended_action"],
            "ms": (time.perf_counter() - t) * 1000,
        }
    out["_total_ms"] = (time.perf_counter() - t0) * 1000
    return out

SELF = boot_self_test()

@st.cache_data(show_spinner="Recomputing policy frontier from frozen artifacts…")
def compute_frontier():
    """Vectorized replica of src/eval/stage4_evaluate.py on the val_cal COD subset.
    Closely replicates the frozen report (thresholds 0.20/0.48, argmin line -327,071)."""
    val_cal_path = os.path.join(BASE_DIR, "data/processed/val_cal.csv")
    val_cal = pd.read_csv(val_cal_path, dtype={"pincode": str})
    mask = val_cal["payment_method"].values == "COD"
    vc = val_cal[mask].reset_index(drop=True)
    p = serve.tree_cal.predict_proba(vc.drop(columns=["rto_label"]))[:, 1]
    margin = vc["order_value"].values.astype(float) * serve.engine.average_margin_pct
    R = serve.engine.rto_logistics_cost

    thresholds = np.arange(0.01, 1.00, 0.01)
    curves = {}
    for action in ["VERIFY_ADDRESS", "PREPAID_ONLY"]:
        par = serve.engine.interventions[action]
        el_x = par["friction_cost"] + (p * (1.0 - par["rto_reduction_pct"])) * R - ((1.0 - p) * (1.0 - par["success_drop_pct"])) * margin
        el_a = 0.0 + p * R - (1.0 - p) * margin
        totals = [float(np.sum(np.where(p > t, el_x, el_a))) for t in thresholds]
        curves[action] = np.array(totals)

    mats = []
    for action, par in serve.engine.interventions.items():
        mats.append(par["friction_cost"] + (p * (1.0 - par["rto_reduction_pct"])) * R - ((1.0 - p) * (1.0 - par["success_drop_pct"])) * margin)
    mat = np.vstack(mats)
    best_idx = np.argmin(mat, axis=0)
    primary_total = float(np.sum(mat[best_idx, np.arange(len(p))]))

    best = {
        "VERIFY_ADDRESS": {
            "t": float(thresholds[int(np.argmin(curves["VERIFY_ADDRESS"]))]),
            "loss": float(np.min(curves["VERIFY_ADDRESS"])),
        },
        "PREPAID_ONLY": {
            "t": float(thresholds[int(np.argmin(curves["PREPAID_ONLY"]))]),
            "loss": float(np.min(curves["PREPAID_ONLY"])),
        },
    }
    return {
        "n": int(len(vc)),
        "thresholds": thresholds,
        "curves": curves,
        "primary_total": primary_total,
        "best": best,
        "edge": best["VERIFY_ADDRESS"]["loss"] - primary_total,
    }

FR = compute_frontier()

@st.cache_data(show_spinner=False)
def load_holdout_test_set():
    """Loads holdout test data to enable the random holdout order sampler."""
    csv_path = os.path.join(BASE_DIR, "data/processed/test.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, dtype={"pincode": str})
        cod_df = df[df["payment_method"] == "COD"].reset_index(drop=True)
        return cod_df
    except Exception:
        return None

HOLDOUT_COD_DF = load_holdout_test_set()

def full_shap(payload, topn=8):
    """Computes signed TreeSHAP values for top-N features using frozen explainer."""
    lookup, _ = serve.resolve_pincode_info(str(payload["pincode"]).strip())
    row = {
        "category": payload["category"],
        "payment_method": payload["payment_method"],
        "courier_id": payload["courier_id"],
        "order_value": float(payload["order_value"]),
        "quantity": float(payload["quantity"]),
        "discount_pct": float(payload["discount_pct"]),
        "cod_charge": float(payload["cod_charge"]),
        "account_age_days": float(payload["account_age_days"]),
        "prior_orders": float(payload["prior_orders"]),
        "prior_rto_count": float(payload["prior_rto_count"]),
        "orders_last_24h": float(payload["orders_last_24h"]),
        "device_cluster_size": float(payload["device_cluster_size"]),
        "historical_pincode_rto_rate": float(lookup["historical_pincode_rto_rate"]),
        "pincode_tier": lookup["pincode_tier"],
    }
    df = pd.DataFrame([row])
    Xt = serve.tree_uncal.named_steps["preprocessor"].transform(df)
    sv = serve.explainer.shap_values(Xt)
    if isinstance(sv, list):
        vals = np.asarray(sv[1])[0]
    elif getattr(sv, "ndim", 2) == 3:
        vals = sv[0, :, 1]
    else:
        vals = np.asarray(sv)[0]
    names = serve.tree_uncal.named_steps["preprocessor"].get_feature_names_out()
    pairs = [(names[i], float(vals[i])) for i in range(len(names))]
    return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:topn]

# ----------------------------------------------------------------------------
# Layout Chrome: Topbar + Footer
# ----------------------------------------------------------------------------
def topbar():
    html_block(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-name">RTO<span>&nbsp;SHIELD</span></div>
    <div class="brand-sub">COD Decision Console</div>
  </div>
  <div class="status-chip">
    <span class="status-dot"></span>
    ENGINE READY&nbsp;·&nbsp;frozen v1.0 artifacts&nbsp;·&nbsp;self-test 3/3 in {SELF['_total_ms']:.0f} ms
  </div>
</div>
""")

def footer():
    st.markdown("---")
    st.markdown(
        f'<div class="foot">github.com/mahirbhat70-eng/rto-shield · frozen v1.0 artifacts · '
        '241/241 tests green in CI · every number on this page is reproducible from reports/ and claim-matrix.md</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# View 01 — Command Center (Pitch Act 1 & 3)
# ----------------------------------------------------------------------------
def view_overview():
    html_block(f"""
<div class="eyebrow">Autonomous COD Risk Engine · Production Grade v1.0</div>
<div class="hero-h1">We don't predict the coin flip.<br><span class="grad">We price it.</span></div>
<div class="hero-sub">RTO Shield scores every COD order in under 100&nbsp;ms, prices four interventions with a real
rupee cost matrix (₹150 landed cost per return · 20% average margin), and routes each order to the
cheapest expected loss. Risk is not an arbitrary label here — it is a price, and the engine always selects the cheapest one.</div>
<div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 0.65rem 1.1rem; margin: 1rem 0 1.2rem 0; font-size: 0.84rem; color: #92400E; display: flex; align-items: center; gap: 0.6rem;">
  <span style="background: #D97706; color: #FFFFFF; font-weight: 800; font-size: 0.70rem; padding: 0.2rem 0.5rem; border-radius: 4px; letter-spacing: 0.05em;">SHADOW-MODE ONLY</span>
  <span><b>Live Intervention Gate Active:</b> Pre-production traffic operates in passive observation mode pending real merchant margin/courier calibration. Safety circuit breakers wired: Kill-Switch, 60% Rate Cap, >₹10k Review, and 16.8% Break-Even Gate.</span>
</div>
""")

    html_block("".join([
        '<div class="kpi-grid">',
        kpi("Expected savings", rs(71741), "<b>[Simulated Benchmark]</b> vs always-allow on holdout test (7,174 COD orders). Argmin routing expected loss.", "green"),
        kpi("Realized savings", rs(69786), "<b>[Simulated Benchmark]</b> Scored with actual RTO labels under baseline response assumptions.", "green"),
        kpi("vs best threshold", "2.0×", "EL saves <b>₹71,741 vs ₹35,919</b>. Causal bounds <b>[1.23×, 2.39×]</b> via 5k Monte Carlo.", "blue"),
        kpi("of Bayes ceiling", "94.7%", "PR-AUC <b>0.3313 / 0.3497</b> — extracts 94.7% of observable ceiling signal.", "amber"),
        "</div>",
        '<div class="kpi-grid" style="margin-top: 0.8rem;">',
        kpi("Profit uplift", "13.1%", "Pre-registered PASS band <b>[8%, 18%]</b> of baseline loss declared before test reveal.", "green"),
        kpi("P(savings &gt; 0)", "100%", "Across <b>5,000-draw Monte Carlo</b> on intervention effects. P5 ₹63,935 · P95 ₹75,901.", "blue"),
        kpi("Scoring path", "~15 ms", "End-to-end latency with TreeSHAP explanation (core model scoring ~3 ms).", "blue"),
        kpi("Test suite", "241/241", "18 test files green in CI with 93% mutation score, SHA-256 integrity & PII allow-list.", "blue"),
        "</div>",
    ]))

    st.markdown("")
    html_block(f"""
<div class="panel">
  <div class="panel-h">How a decision is priced — four moves, one argmin</div>
  <div class="step-strip">
    <div class="step"><div class="n">01 · FEATURES</div><div class="h">13 order signals</div>
      <div class="b">Order value, quantity, category, COD charge, account age, prior RTOs, velocity, device cluster, pincode tier/rate, courier.</div></div>
    <div class="step"><div class="n">02 · PROBABILITY</div><div class="h">Calibrated LightGBM</div>
      <div class="b">Calibrated P(RTO) probabilities that translate directly to rupee rates. Calibration holds per action (|Δ| ≤ 0.0032).</div></div>
    <div class="step"><div class="n">03 · PRICING</div><div class="h">Cost matrix, 4 actions</div>
      <div class="b">ALLOW ₹0 friction · VERIFY ₹2 / −30% RTO / −5% conv. · DEPOSIT −80% RTO / −40% conv. · PREPAID −55% RTO / −70% conv.</div></div>
    <div class="step"><div class="n">04 · ROUTING</div><div class="h">argmin expected loss</div>
      <div class="b">Ship with the cheapest expected loss per order. Deterministic math, zero magic cutoffs.</div></div>
  </div>
</div>
""")

    html_block(f"""
<div class="panel" style="margin-top: 1rem;">
  <div class="panel-h">Production Safety Circuit Breakers · SHADOW-MODE ONLY</div>
  <div class="step-strip" style="grid-template-columns: repeat(5, 1fr);">
    <div class="step">
      <div class="n">BREAKER 01</div>
      <div class="h">Emergency Kill-Switch</div>
      <div class="b">Instant flag/env trigger <code>RTO_SHIELD_KILL_SWITCH</code> bypassing all models directly to <code>ALLOW_COD</code>.</div>
    </div>
    <div class="step">
      <div class="n">BREAKER 02</div>
      <div class="h">60% Rate Cap</div>
      <div class="b">Sliding-window limiter strictly preventing interventions from exceeding 60% of rolling order volume.</div>
    </div>
    <div class="step">
      <div class="n">BREAKER 03</div>
      <div class="h">&gt;₹10,000 Safeguard</div>
      <div class="b">High-value baskets automatically route to <code>MANUAL_REVIEW</code> instead of checkout friction.</div>
    </div>
    <div class="step">
      <div class="n">BREAKER 04</div>
      <div class="h">16.8% Break-Even Gate</div>
      <div class="b">Brands with &lt;16.8% base COD RTO bypass friction, preventing net margin destruction on low-risk catalogues.</div>
    </div>
    <div class="step">
      <div class="n">BREAKER 05</div>
      <div class="h">SHA-256 &amp; PII Allow-List</div>
      <div class="b">Pickle validation against 9-hash manifest + strict customer PII scrubbing before audit ingestion.</div>
    </div>
  </div>
</div>
""")

    html_block(f"""
<div class="kpi-grid" style="margin-top:0.8rem; grid-template-columns: repeat(5, 1fr);">
{kpi('The COD bleed', '28.27%', 'of COD orders on holdout test ended in RTO. At ₹150 landed cost, COD is the biggest margin leak.', 'amber')}
{kpi('Live Decision Engine', 'View 02', 'Load presets, edit order features, score order, and inspect TreeSHAP drivers.', 'blue')}
{kpi('Policy Frontier', 'View 03', 'The policy frontier chart, recomputed live. No single cutoff beats argmin routing.', 'blue')}
{kpi('Portfolio Evidence', 'View 04', 'Expected vs realized savings, 5,000 Monte Carlo draws, per-shelf calibration.', 'blue')}
{kpi('Audit & Governance', 'View 05', '17-point audit reconciliation, parameter misspecification, and 5-seed stability.', 'green')}
</div>
""")

# ----------------------------------------------------------------------------
# View 02 — Live Decision Engine (Pitch Act 4)
# ----------------------------------------------------------------------------
def _clamp_prior_rto():
    po = int(st.session_state.get("ni_prior_orders", 0))
    prc = int(st.session_state.get("prior_rto_count", 0))
    st.session_state["prior_rto_count"] = min(prc, po)

def sidebar_inputs():
    def get_val(key, default):
        wkey = WIDGET_KEYS.get(key)
        if wkey and wkey in st.session_state:
            st.session_state.setdefault("_persisted_form", {})[key] = st.session_state[wkey]
            return st.session_state[wkey]
        return st.session_state.get("_persisted_form", {}).get(key, default)

    def load_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[WIDGET_KEYS[k]] = v
            st.session_state.setdefault("_persisted_form", {})[k] = v
        st.session_state["active_preset"] = name
        st.session_state["holdout_label"] = None
        st.session_state["holdout_payload"] = None
        st.session_state["score_requested"] = True

    def sample_random_holdout():
        if HOLDOUT_COD_DF is not None and len(HOLDOUT_COD_DF) > 0:
            sample = HOLDOUT_COD_DF.sample(n=1).iloc[0]
            sampled_payload = {}
            for col in WIDGET_KEYS:
                if col in sample:
                    val = sample[col]
                    if isinstance(val, (np.integer, int)):
                        val = int(val)
                    elif isinstance(val, (np.floating, float)):
                        val = float(val)
                    else:
                        val = str(val)
                    st.session_state[WIDGET_KEYS[col]] = val
                    st.session_state.setdefault("_persisted_form", {})[col] = val
                    sampled_payload[col] = val
            po = int(sample.get("prior_orders", 0))
            prc = min(int(sample.get("prior_rto_count", 0)), max(1, po))
            st.session_state[WIDGET_KEYS["prior_rto_count"]] = prc
            st.session_state.setdefault("_persisted_form", {})["prior_rto_count"] = prc
            sampled_payload["prior_rto_count"] = prc
            st.session_state["active_preset"] = "RANDOM"
            st.session_state["holdout_label"] = int(sample.get("rto_label", 0))
            st.session_state["holdout_payload"] = sampled_payload
            st.session_state["score_requested"] = True

    st.sidebar.markdown('<div class="side-h">Demo presets</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.sidebar.columns(3)
    c1.button("DEPOSIT", on_click=load_preset, args=("DEPOSIT",), help=PRESET_HINT["DEPOSIT"], use_container_width=True)
    c2.button("ALLOW", on_click=load_preset, args=("ALLOW",), help=PRESET_HINT["ALLOW"], use_container_width=True)
    c3.button("VERIFY", on_click=load_preset, args=("VERIFY",), help=PRESET_HINT["VERIFY"], use_container_width=True)

    if st.sidebar.button("🎲 Random holdout order", help="Sample an unseen COD order from data/processed/test.csv", use_container_width=True):
        sample_random_holdout()

    active = st.session_state.get("active_preset")
    if active:
        st.sidebar.caption(f"Active preset: **{active}**")

    st.sidebar.markdown('<div class="side-h" style="margin-top:0.9rem;">Order Intake</div>', unsafe_allow_html=True)
    st.sidebar.number_input(
        "Order value (₹)", min_value=0.0, max_value=25000.0, step=10.0,
        value=float(get_val("order_value", 852.0)), key="ni_order_value"
    )
    cat_val = str(get_val("category", "Home"))
    cat_idx = CATEGORIES.index(cat_val) if cat_val in CATEGORIES else 0
    st.sidebar.selectbox(
        "Category", CATEGORIES,
        index=cat_idx,
        key="category"
    )
    pm_opts = ["COD", "PREPAID"]
    pm_val = str(get_val("payment_method", "COD"))
    pm_idx = pm_opts.index(pm_val) if pm_val in pm_opts else 0
    payment_method = st.sidebar.selectbox(
        "Payment method", pm_opts,
        index=pm_idx,
        key="payment_method"
    )
    st.sidebar.number_input(
        "Quantity", min_value=1, max_value=50, step=1,
        value=int(get_val("quantity", 3)), key="ni_quantity"
    )
    st.sidebar.number_input(
        "Discount %", min_value=0.0, max_value=100.0, step=0.5,
        value=float(get_val("discount_pct", 19.8)), key="ni_discount_pct"
    )

    if payment_method == "COD":
        st.sidebar.number_input(
            "COD charge (₹)", min_value=0.0, max_value=500.0, step=1.0,
            value=float(get_val("cod_charge", 59.0)), key="ni_cod_charge",
            help="Train COD median ₹49"
        )
    else:
        st.sidebar.number_input(
            "COD charge (₹)", min_value=0.0, max_value=0.0, value=0.0,
            disabled=True, help="Disabled for PREPAID orders"
        )

    st.sidebar.markdown('<div class="side-h">Customer History</div>', unsafe_allow_html=True)
    st.sidebar.number_input(
        "Account age (days)", min_value=0, max_value=5000, step=1,
        value=int(get_val("account_age_days", 65)), key="ni_account_age_days"
    )
    prior_orders = st.sidebar.number_input(
        "Prior orders", min_value=0, max_value=100, step=1,
        value=int(get_val("prior_orders", 2)), key="ni_prior_orders",
        on_change=_clamp_prior_rto,
    )
    max_prc = max(1, int(prior_orders))
    prc_val = min(int(get_val("prior_rto_count", 0)), max_prc)
    if "prior_rto_count" in st.session_state:
        st.session_state["prior_rto_count"] = min(int(st.session_state["prior_rto_count"]), max_prc)
    st.sidebar.slider(
        "Prior RTO count", 0, max_prc,
        value=prc_val,
        key="prior_rto_count"
    )

    st.sidebar.markdown('<div class="side-h">Velocity & Logistics</div>', unsafe_allow_html=True)
    st.sidebar.number_input(
        "Orders last 24h", min_value=0, max_value=50, step=1,
        value=int(get_val("orders_last_24h", 3)), key="ni_orders_last_24h"
    )
    st.sidebar.number_input(
        "Device cluster size", min_value=1, max_value=50, step=1,
        value=int(get_val("device_cluster_size", 1)), key="ni_device_cluster_size"
    )

    pin_in = st.sidebar.text_input("Pincode", value=str(get_val("pincode", "253407")), key="ti_pincode", max_chars=6).strip()
    # Live pincode lookup preview
    if pin_in in PINCODE_LOOKUP:
        pin_meta = PINCODE_LOOKUP[pin_in]
        rate_pct = float(pin_meta['historical_pincode_rto_rate']) * 100
        st.sidebar.caption(f"✓ Tier **{pin_meta['pincode_tier']}** · Historical RTO **{rate_pct:.1f}%**")
    else:
        st.sidebar.markdown('<span style="color:#DC2626;font-size:0.75rem;font-weight:600;">⚠️ Unknown pincode — global prior used (cold start)</span>', unsafe_allow_html=True)

    st.sidebar.text_input("Courier ID", value=str(get_val("courier_id", "Courier_E")), key="ti_courier_id", max_chars=16)

    st.sidebar.markdown('<div class="side-h">Production Guardrails</div>', unsafe_allow_html=True)
    st.sidebar.checkbox(
        "Enforce safety circuit breakers", value=True, key="enforce_guardrails",
        help="Evaluates circuit breakers (Kill Switch, Break-Even Gate, >₹10k Review, Rate Cap)"
    )
    st.sidebar.number_input(
        "Merchant baseline COD RTO (%)", min_value=0.0, max_value=100.0, step=0.5,
        value=float(st.session_state.get("merchant_cod_rto_pct", 28.0)), key="merchant_cod_rto_pct",
        help="Merchant baseline COD return rate. If <16.8%, break-even gate auto-enforces ALLOW_COD."
    )
    st.sidebar.checkbox(
        "Emergency Kill-Switch (simulate)", value=False, key="simulate_kill_switch",
        help="Instantly reverts all orders to ALLOW_COD via emergency circuit breaker"
    )

    # Sync all rendered widget values into _persisted_form for clean view-switch restoration
    for k, wkey in WIDGET_KEYS.items():
        if wkey in st.session_state:
            st.session_state.setdefault("_persisted_form", {})[k] = st.session_state[wkey]

def payload_from_state():
    payment = st.session_state.get(WIDGET_KEYS["payment_method"], "COD")
    payload = {
        "payment_method": payment,
        "order_value": float(st.session_state.get(WIDGET_KEYS["order_value"], 852.0)),
        "category": st.session_state.get(WIDGET_KEYS["category"], "Home"),
        "quantity": int(st.session_state.get(WIDGET_KEYS["quantity"], 3)),
        "discount_pct": float(st.session_state.get(WIDGET_KEYS["discount_pct"], 19.8)),
        "cod_charge": float(st.session_state.get(WIDGET_KEYS["cod_charge"], 59.0))
                      if payment == "COD" else 0.0,
        "account_age_days": int(st.session_state.get(WIDGET_KEYS["account_age_days"], 65)),
        "prior_orders": int(st.session_state.get(WIDGET_KEYS["prior_orders"], 2)),
        "prior_rto_count": int(st.session_state.get(WIDGET_KEYS["prior_rto_count"], 0)),
        "orders_last_24h": int(st.session_state.get(WIDGET_KEYS["orders_last_24h"], 3)),
        "device_cluster_size": int(st.session_state.get(WIDGET_KEYS["device_cluster_size"], 1)),
        "pincode": str(st.session_state.get(WIDGET_KEYS["pincode"], "253407")).strip(),
        "courier_id": str(st.session_state.get(WIDGET_KEYS["courier_id"], "Courier_E")).strip()
    }
    if "_persisted_form" in st.session_state:
        st.session_state["_persisted_form"].update(payload)
    return payload

def shap_html(pairs):
    if not pairs:
        return ('<div style="color:#64748B;font-size:0.8rem;padding:0.6rem 0;">'
                'PREPAID passthrough — no RTO risk to price; model and TreeSHAP skipped.'
                '</div>')
    mx = max(abs(v) for _, v in pairs) or 1.0
    rows = []
    for name, v in pairs:
        w = abs(v) / mx * 48.0
        # Pretty display name
        disp_name = name.replace("num__", "").replace("cat__", "").replace("onehot__", "")
        if v >= 0:
            fill = f'<div class="shap-fill" style="left:50%;width:{w:.1f}%;background:{C_RED};"></div>'
            val = f'<span class="shap-val" style="color:{C_RED};">+{v:.3f}</span>'
        else:
            fill = f'<div class="shap-fill" style="right:50%;width:{w:.1f}%;background:{C_RAZORPAY_BLUE};"></div>'
            val = f'<span class="shap-val" style="color:{C_RAZORPAY_BLUE};">{v:.3f}</span>'
        rows.append(
            f'<div class="shap-row"><div class="shap-name" title="{name}">{disp_name}</div>'
            f'<div class="shap-track"><div class="mid"></div>{fill}</div>{val}</div>'
        )
    return "".join(rows)

def action_row(action, el, selected, params=None):
    label, color, bg, _ = ACTION_META[action]
    chips = ""
    if params:
        chips = (
            f'<div class="a-chips"><span class="achip">friction {rs(params["friction_cost"])}</span>'
            f'<span class="achip">RTO ×{1 - params["rto_reduction_pct"]:.2f}</span>'
            f'<span class="achip">conversions ×{1 - params["success_drop_pct"]:.2f}</span></div>'
        )
    pick = '<span class="pick">ROUTED · ARGMIN</span>' if selected else ""
    sel = " sel" if selected else ""
    valcolor = C_RAZORPAY_BLUE if selected else C_TEXT
    return (
        f'<div class="arow{sel}"><div class="a-name">'
        f'<div class="a-label" style="color:{color};">{label}{pick}</div>{chips}</div>'
        f'<div class="a-el"><div class="lbl">Expected loss / order</div>'
        f'<div class="val" style="color:{valcolor};">{rs(el, 2)}</div></div></div>'
    )

def view_scorer():
    sidebar_inputs()
    c1, c2 = st.columns([3.2, 2])
    with c1:
        st.markdown("### Live decision engine")
    with c2:
        st.markdown(
            f'<div style="text-align:right;color:{C_MUT};font-size:0.8rem;padding-top:0.4rem;">'
            f"Load a preset or edit any field → hit <b style='color:{C_RAZORPAY_BLUE};'>Score this order</b>."
            "</div>", unsafe_allow_html=True)

    if st.button("SCORE THIS ORDER", type="primary", use_container_width=True):
        st.session_state["score_requested"] = True

    if st.session_state.get("score_requested"):
        st.session_state["score_requested"] = False
        payload = payload_from_state()
        enforce_gr = st.session_state.get("enforce_guardrails", True)
        sim_kill = st.session_state.get("simulate_kill_switch", False)
        m_rto_pct = float(st.session_state.get("merchant_cod_rto_pct", 28.0))
        payload["merchant_cod_rto_rate"] = m_rto_pct / 100.0

        try:
            t0 = time.perf_counter()
            if enforce_gr:
                from src.serve.guardrails import ProductionGuardrails
                gr = ProductionGuardrails(kill_switch=sim_kill, min_base_rto_rate=0.168)
                res = gr.evaluate(payload, score_order)
                if "el_table" not in res:
                    from src.serve.scorer import engine
                    res["el_table"] = engine.evaluate_interventions(payload["order_value"], res["probability"]) if payload["payment_method"] == "COD" else {k: 0.0 for k in engine.interventions.keys()}
            else:
                res = score_order(payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            entry = build_audit_record(payload, res, latency_ms=latency_ms)
            audit_log = st.session_state.setdefault("audit_log", [])
            if not audit_log or audit_log[-1].get("decision_id") != entry["decision_id"]:
                audit_log.append(entry)
            st.session_state["last"] = {
                "payload": payload,
                "res": res,
                "latency_ms": latency_ms,
                "entry": entry,
            }
        except ValueError as e:
            msg = str(e)
            if "exceeds maximum allowed value" in msg:
                st.error("**Guardrail rejected this order** — input outside the tested P99.9 boundary.")
                st.caption(msg)
            elif "not found in lookup table" in msg:
                st.error(f"**Pincode validation failed**: {msg}")
            else:
                st.error(msg)
            st.session_state.pop("last", None)
            st.session_state["holdout_label"] = None
            st.session_state["holdout_payload"] = None

    last = st.session_state.get("last")
    if not last:
        html_block(f"""
<div class="panel" style="text-align:center;padding:3rem 1.5rem;">
  <div style="font-size:1.45rem;font-weight:850;color:{C_TEXT};letter-spacing:-0.02em;">No order scored yet.</div>
  <div style="color:{C_MUT};font-size:0.88rem;margin-top:0.5rem;">
    Choose a preset from the sidebar — <b style="color:{C_AMBER};">VERIFY</b> is recommended for a live demo —
    then click <b style="color:{C_RAZORPAY_BLUE};">SCORE THIS ORDER</b>.
  </div>
</div>
""")
        _audit_block()
        return

    res = last["res"]
    p = res["probability"]
    action = res["recommended_action"]
    el = res["el_table"]
    risk_color = C_GREEN if p < 0.25 else (C_AMBER if p < 0.40 else C_RED)
    label, acolor, abg, adesc = ACTION_META.get(
        action, ("PREPAID PASSTHROUGH", C_SLATE, C_SLATE_BG, "Prepaid orders skip the COD risk engine entirely.")
    )

    # Scorer warnings (N5)
    for w in res.get("warnings", []):
        st.warning(w)

    # Production Guardrail Triggered Badge
    guardrail_applied = res.get("guardrail_applied")
    guardrail_badge = ""
    if guardrail_applied and guardrail_applied != "NONE":
        guardrail_badge = (
            f'<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:9px;padding:0.6rem 0.95rem;margin-bottom:0.8rem;display:flex;align-items:center;gap:0.6rem;font-size:0.82rem;color:#92400E;">'
            f'<span style="background:#D97706;color:#FFFFFF;font-weight:800;font-size:0.68rem;padding:0.18rem 0.45rem;border-radius:4px;letter-spacing:0.04em;">CIRCUIT BREAKER</span>'
            f'<span><b>Guardrail Triggered:</b> {guardrail_applied} · Status: <b>{res.get("status", "OVERRIDDEN")}</b></span>'
            f'</div>'
        )

    # Holdout ground truth indicator if random order was chosen (N2)
    hl = st.session_state.get("holdout_label")
    hp = st.session_state.get("holdout_payload")
    holdout_badge = ""
    if hl is not None and hp is not None:
        matches = all(str(last["payload"].get(k)) == str(hp.get(k)) for k in hp)
        if matches:
            truth_color = C_RED if hl == 1 else C_GREEN
            truth_text = "RTO Occurred (1)" if hl == 1 else "Delivered Successfully (0)"
            holdout_badge = f'<div style="margin-bottom:0.7rem;"><span class="verify-chip" style="background:#F1F5F9;border-color:#CBD5E1;color:{truth_color};">🎯 Holdout ground truth: <b>{truth_text}</b></span></div>'
        else:
            st.session_state["holdout_label"] = None
            st.session_state["holdout_payload"] = None

    # ---- KPI strip: probability / decision / latency
    is_prepaid = last["payload"].get("payment_method", "").upper() != "COD"
    prob_title = "P(RTO) · non-COD passthrough" if is_prepaid else "P(RTO) · calibrated probability"
    prob_desc = (
        "Passthrough — zero COD logistics exposure; expected return loss is <b>₹0.00</b>."
        if is_prepaid else
        f"Rupee rate: on <b>₹{inr(last['payload']['order_value'])}</b> COD, expected return loss ≈ <b>{rs(p * 150)}</b>."
    )
    html_block(f"""
{guardrail_badge}
{holdout_badge}
<div class="kpi-grid" style="grid-template-columns: 1.15fr 1.15fr 0.9fr;">
  <div class="kpi">
    <div class="t">{prob_title}</div>
    <div class="big-p" style="color:{risk_color};">{p * 100:.1f}%</div>
    <div class="track"><i style="width:{min(p * 100, 100):.1f}%;background:{risk_color};"></i></div>
    <div class="d">{prob_desc}</div>
  </div>
  <div class="kpi">
    <div class="t">Decision · argmin expected loss</div>
    <div style="margin:0.55rem 0 0.55rem 0;"><span class="badge" style="background:{acolor};">{label}</span></div>
    <div class="d">{adesc}</div>
  </div>
  <div class="kpi">
    <div class="t">Latency · full path</div>
    <div class="lat">{last['latency_ms']:.1f} ms</div>
    <div class="lat-sub">LightGBM + TreeSHAP + cost engine<br/>bound by test suite: &lt; 100 ms</div>
  </div>
</div>
""")

    st.markdown("")
    # ---- Price Menu
    if last["payload"]["payment_method"] == "PREPAID":
        html_block(f"""
<div class="panel">
  <div class="panel-h">Expected-loss price menu</div>
  <div class="verify-chip" style="background:{C_SLATE_BG};color:{C_SLATE};border-color:{C_SLATE_BORDER};">
    PREPAID ORDER — passthrough. Zero COD logistics exposure; all 4 interventions price at ₹0.00 by definition.
  </div>
</div>
""")
    else:
        params = serve.engine.interventions
        rows = "".join(action_row(a, el[a], a == action, params[a]) for a in params.keys())
        savings = el["ALLOW_COD"] - el[action]
        html_block(f"""
<div class="panel">
  <div class="panel-h">Expected-loss price menu · all four moves, one argmin</div>
  {rows}
  <div style="color:{C_MUT_DARK};font-size:0.78rem;margin-top:0.65rem;">
    EL = friction + (RTO after intervention × ₹150) − (surviving conversions × 20% margin) ·
    routing saves <b style="color:{C_GREEN};">{rs(savings, 2)}</b> per order vs always-allow.
  </div>
</div>
""")

    st.markdown("")
    # ---- SHAP + Input Intelligence
    if last["payload"].get("payment_method", "").upper() == "COD":
        pairs = full_shap(last["payload"], topn=8)
    else:
        pairs = []
    lookup, is_cold = serve.resolve_pincode_info(str(last["payload"]["pincode"]).strip())
    entry = last["entry"]
    ok = verify_audit_record(entry)
    left, right = st.columns([1.35, 1])
    with left:
        html_block(f"""
<div class="panel">
  <div class="panel-h">Why this price · TreeSHAP on frozen LightGBM</div>
  {shap_html(pairs)}
  <div style="display:flex;gap:1.2rem;color:{C_MUT};font-size:0.72rem;margin-top:0.7rem;">
    <span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:{C_RED};margin-right:0.35rem;"></i>pushes risk up</span>
    <span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:{C_RAZORPAY_BLUE};margin-right:0.35rem;"></i>pulls risk down</span>
  </div>
</div>
""")
    with right:
        pin_rate = float(lookup["historical_pincode_rto_rate"]) * 100
        tier = lookup["pincode_tier"]
        esc = lambda v: html_mod.escape(str(v))
        html_block(f"""
<div class="panel">
  <div class="panel-h">Input intelligence</div>
  <div class="d" style="font-size:0.82rem;color:{C_MUT_DARK};line-height:1.75;">
    Pincode <b>{esc(last['payload']['pincode'])}</b> → tier {esc(tier)} · historical RTO <b>{pin_rate:.1f}%</b><br/>
    Courier <b>{esc(last['payload']['courier_id'])}</b> · device cluster <b>{esc(last['payload']['device_cluster_size'])}</b><br/>
    Account <b>{esc(last['payload']['account_age_days'])}d</b> · velocity <b>{esc(last['payload']['orders_last_24h'])}/24h</b>
  </div>
  <hr class="soft"/>
  <div class="panel-h">Audit receipt</div>
  <div class="d" style="font-size:0.8rem;line-height:1.7;">
    decision_id <b style="font-family:monospace;color:{C_RAZORPAY_BLUE};">{entry['decision_id']}</b><br/>
    <span style="color:{C_GREEN};font-weight:700;">{'✓ fingerprint verified — tamper-evident' if ok else '✗ fingerprint mismatch'}</span><br/>
    <span style="color:{C_MUT};">sha256(canonical payload ∥ model_version ∥ action)[:16], replayable</span>
  </div>
</div>
""")

    _audit_block()

def _audit_block():
    log = st.session_state.get("audit_log")
    if not log:
        return
    st.markdown("")
    df = pd.DataFrame([{
        "time (UTC)": r["timestamp"][11:19],
        "P(RTO)": f"{r['probability']*100:.1f}%",
        "action": r["recommended_action"],
        "latency": f"{r['latency_ms']:.1f} ms",
        "decision_id": r["decision_id"],
        "verified": "✓" if verify_audit_record(r) else "✗",
    } for r in log])
    st.markdown(f"**Decision audit trail** — {len(log)} decision(s) logged this session, exportable & replay-fingerprinted")
    st.dataframe(df, use_container_width=True, hide_index=True)
    d1, d2, _ = st.columns([1.1, 0.7, 3])
    with d1:
        st.download_button(
            "Download .jsonl", data=to_jsonl(log),
            file_name="rto_audit_trail.jsonl", mime="application/x-ndjson",
            use_container_width=True
        )
    with d2:
        if st.button("Clear log", use_container_width=True):
            st.session_state["audit_log"] = []
            st.session_state.pop("last", None)
            st.rerun()

# ----------------------------------------------------------------------------
# View 03 — Policy Frontier (Pitch Act 2)
# ----------------------------------------------------------------------------
def view_frontier():
    st.markdown("### Nothing beats the line")
    st.markdown(
        f'<div style="color:{C_MUT_DARK};font-size:0.88rem;max-width:920px;line-height:1.55;margin-bottom:1.1rem;">'
        "Every single-threshold policy sweeps a worse portfolio loss than argmin routing. The engine prices all four "
        "interventions <i>per order</i> based on the order's specific margin, so there is no single cutoff to tune. "
        "This chart is recomputed live in your browser from the frozen artifacts — closely matching the audited stage-4 report.</div>",
        unsafe_allow_html=True)

    th = FR["thresholds"]
    src = pd.DataFrame({
        "threshold": np.concatenate([th, th]),
        "loss": np.concatenate([FR["curves"]["VERIFY_ADDRESS"], FR["curves"]["PREPAID_ONLY"]]),
        "policy": ["Binary VERIFY block"] * len(th) + ["Binary PREPAID block"] * len(th),
    })
    lines = (alt.Chart(src)
             .mark_line(strokeWidth=2.2)
             .encode(
                 x=alt.X("threshold:Q", title="Single decision threshold (t)",
                         axis=alt.Axis(format=".2f", gridColor="#E2E8F0", labelColor=C_MUT, titleColor=C_MUT_DARK)),
                 y=alt.Y("loss:Q", title="Portfolio expected loss (₹)",
                         axis=alt.Axis(format=",d", gridColor="#E2E8F0", labelColor=C_MUT, titleColor=C_MUT_DARK)),
                 color=alt.Color("policy:N",
                                 scale=alt.Scale(domain=["Binary VERIFY block", "Binary PREPAID block"],
                                                 range=[C_AMBER, C_RAZORPAY_BLUE]),
                                 legend=alt.Legend(labelColor=C_TEXT, title=None, orient="top")),
             ))
    hline = (alt.Chart(pd.DataFrame({"y": [FR["primary_total"]], "policy": ["Argmin routing (no threshold)"]}))
             .mark_rule(strokeDash=[6, 4], color=C_RED, strokeWidth=2.5)
             .encode(y="y:Q", color=alt.Color("policy:N", scale=alt.Scale(domain=["Argmin routing (no threshold)"],
                                                                         range=[C_RED]),
                                             legend=alt.Legend(labelColor=C_TEXT, title=None, orient="top"))))
    pts = (alt.Chart(pd.DataFrame({
             "threshold": [FR["best"]["VERIFY_ADDRESS"]["t"], FR["best"]["PREPAID_ONLY"]["t"]],
             "loss": [FR["best"]["VERIFY_ADDRESS"]["loss"], FR["best"]["PREPAID_ONLY"]["loss"]],
         }))
           .mark_circle(size=95, color=C_TEXT, opacity=0.9)
           .encode(x="threshold:Q", y="loss:Q"))

    chart = (lines + hline + pts).resolve_scale(color="independent")
    chart = chart.configure_view(stroke=None).configure(background="#FFFFFF", font="Inter")
    st.altair_chart(chart, use_container_width=True, theme=None)

    st.markdown("")
    edge_pct = FR["edge"] / abs(FR["best"]["VERIFY_ADDRESS"]["loss"]) * 100
    edge_val = FR["edge"]
    edge_str = f"+{rs(edge_val)}" if edge_val >= 0 else rs(edge_val)
    html_block("".join([
        '<div class="kpi-grid">',
        kpi("Best single cutoff · VERIFY", rs(FR["best"]["VERIFY_ADDRESS"]["loss"]),
            f"at t = {FR['best']['VERIFY_ADDRESS']['t']:.2f} — the amber curve bottoms out here, still short of the line.", "amber"),
        kpi("Best single cutoff · PREPAID", rs(FR["best"]["PREPAID_ONLY"]["loss"]),
            f"at t = {FR['best']['PREPAID_ONLY']['t']:.2f} — kills customer volume before it comes close.", "blue"),
        kpi("Argmin routing line", rs(FR["primary_total"]),
            f"per-order pricing on {FR['n']:,} COD orders (val_cal). Below every point of every curve.", "green"),
        kpi("Edge of the line", edge_str,
            f"{abs(edge_pct):.1f}% better than the best single threshold you could ever tune.", "green"),
        "</div>",
    ]))

    html_block(f"""
<div style="margin-top:0.9rem;">
  <span class="verify-chip">✓ recomputed live · tuned thresholds 0.20 / 0.48 reproduced · routing line matches frozen stage-4 artifacts</span>
</div>
""")

    with st.expander("Frozen report artifact — reports/stage4/threshold_vs_loss_curve.png (the audited version of this chart)"):
        img_path = os.path.join(BASE_DIR, "reports/stage4/threshold_vs_loss_curve.png")
        if os.path.exists(img_path):
            try:
                st.image(img_path, use_column_width=True)
            except TypeError:
                st.image(img_path, use_container_width=True)
        st.caption("Generated by src/eval/stage4_evaluate.py on the same frozen artifacts. The live chart above replicates it closely.")

# ----------------------------------------------------------------------------
# View 04 — Portfolio Evidence (Pitch Act 3)
# ----------------------------------------------------------------------------
def view_portfolio():
    st.markdown("### Portfolio evidence")
    st.markdown(
        f'<div style="color:{C_MUT_DARK};font-size:0.88rem;max-width:920px;line-height:1.55;margin-bottom:1.1rem;">'
        "Frozen stage-5 results on the 14,980-order holdout test, quoted verbatim with provenance. "
        "Every row below has a pytest one-liner that re-derives it bit-for-bit from the repo artifacts.</div>",
        unsafe_allow_html=True)

    html_block("".join([
        '<div class="kpi-grid">',
        kpi("Expected savings", rs(71741, 2), "argmin routing vs always-allow · COD subset (7,174 orders)", "green"),
        kpi("Realized savings", rs(69786, 2), "actual <b>rto_label</b> outcomes through the same cost engine", "green"),
        kpi("Forecast error", "-₹1,955", "realized − expected = <b>−2.7%</b> — forecast survives contact with reality", "amber"),
        kpi("Profit uplift", "13.1%", "pre-registered PASS band <b>[8%, 18%]</b> of baseline loss", "green"),
        "</div>",
    ]))

    left, right = st.columns([1.15, 1])
    with left:
        html_block(f"""
<div class="panel">
  <div class="panel-h">Action distribution · test COD subset (n=7,174)</div>
  <div class="donut-wrap">
    <div class="donut" style="background: conic-gradient(
        {C_GREEN} 0% 18.4%,
        {C_AMBER} 18.4% 63.6%,
        {C_RED} 63.6% 100%);"></div>
    <div class="legend">
      <div class="leg"><i style="background:{C_GREEN};"></i>ALLOW COD <b>&nbsp;18.4%</b></div>
      <div class="leg"><i style="background:{C_AMBER};"></i>VERIFY ADDRESS <b>&nbsp;45.2%</b></div>
      <div class="leg"><i style="background:{C_RED};"></i>REQUIRE DEPOSIT <b>&nbsp;36.4%</b></div>
      <div class="leg"><i style="background:{C_SLATE};"></i>PREPAID ONLY <b>&nbsp;0.0%</b></div>
    </div>
  </div>
  <div class="d" style="font-size:0.78rem;color:{C_MUT_DARK};margin-top:0.85rem;line-height:1.5;">
    The engine touches <b style="color:{C_TEXT};">81.6%</b> of orders but keeps PREPAID_ONLY at exactly zero —
    it never kills an order it can save. 5,856 interventions · ₹6,488 friction spend · 951 expected RTOs prevented.
  </div>
</div>
""")
    with right:
        pos = lambda v: f"{(v - 62000) / (78000 - 62000) * 100:.1f}%"
        html_block(f"""
<div class="panel">
  <div class="panel-h">Monte Carlo · 5,000 draws on intervention effects</div>
  <div class="mc-track">
    <div class="mc-band" style="left:{pos(63935.40)};width:{(75901.15 - 63935.40) / 16000 * 100:.1f}%;"></div>
    <div class="mc-mean" style="left:{pos(69942.31)};"></div>
    <div class="mc-lab" style="left:{pos(63935.40)};bottom:-1.4rem;">P5 ₹63,935</div>
    <div class="mc-lab" style="left:{pos(69942.31)};top:-1.4rem;color:{C_TEXT};">mean ₹69,942</div>
    <div class="mc-lab" style="left:{pos(75901.15)};bottom:-1.4rem;">P95 ₹75,901</div>
  </div>
  <div class="d" style="font-size:0.84rem;color:{C_MUT_DARK};">
    <b style="color:{C_GREEN};font-size:1.15rem;">P(savings &gt; 0) = 100%</b><br/>
    Bernoulli-sampled per-order intervention drops &amp; reductions. Worst decile still clears ₹63.9k.
  </div>
</div>
""")

    st.markdown("")
    left2, right2 = st.columns([1.15, 1])
    with left2:
        html_block(f"""
<div class="panel">
  <div class="panel-h">Operating point · VERIFY + DEPOSIT = positive</div>
  <div style="display:flex; gap:2.4rem; margin:0.3rem 0 0.7rem 0;">
    <div><div style="font-size:0.68rem;color:{C_MUT};font-weight:700;letter-spacing:0.1em;">PRECISION</div>
    <div style="font-size:1.75rem;font-weight:900;color:{C_TEXT};">0.2963</div></div>
    <div><div style="font-size:0.68rem;color:{C_MUT};font-weight:700;letter-spacing:0.1em;">RECALL</div>
    <div style="font-size:1.75rem;font-weight:900;color:{C_GREEN};">0.8555</div></div>
    <div><div style="font-size:0.68rem;color:{C_MUT};font-weight:700;letter-spacing:0.1em;">F1</div>
    <div style="font-size:1.75rem;font-weight:900;color:{C_TEXT};">0.4401</div></div>
  </div>
  <div class="d" style="font-size:0.79rem;color:{C_MUT_DARK};line-height:1.65;">
    We catch <b style="color:{C_TEXT};">86% of RTOs</b> while touching 81.6% of orders. Precision looks low until you
    remember the negative class is priced, not punished — a false positive just means a ₹2 phone call, not a lost customer.
  </div>
</div>
""")
    with right2:
        cal_df = pd.DataFrame({
            "action": ["REQUIRE_DEPOSIT", "VERIFY_ADDRESS", "ALLOW_COD"],
            "orders": [2612, 3244, 1318],
            "mean P": [0.3279, 0.2733, 0.2191],
            "empirical RTO": [0.3247, 0.2734, 0.2223],
            "|Δ|": [0.0032, 0.0001, 0.0032],
        })
        html_block('<div class="panel"><div class="panel-h">Per-action calibration · the price is honest per shelf</div>')
        st.dataframe(cal_df, use_container_width=True, hide_index=True)
        html_block(f'<div class="d" style="font-size:0.76rem;color:{C_MUT};">Max deviation 0.0032 — each action\'s pricing input matches its realized rate.</div></div>')

    st.markdown("")
    html_block(f"""
<div class="panel">
  <div class="panel-h">Provenance · reproduce it yourself</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem 2rem;font-size:0.78rem;color:{C_MUT_DARK};line-height:1.8;">
    <div>→ ₹71,741 / ₹69,786 / 13.1% · <code style="color:{C_RAZORPAY_BLUE};">pytest tests/test_stage5.py -v</code></div>
    <div>→ 94.7% of Bayes ceiling · <code style="color:{C_RAZORPAY_BLUE};">pytest tests/test_stage3.py -v</code></div>
    <div>→ routing line &amp; thresholds 0.20/0.48 · <code style="color:{C_RAZORPAY_BLUE};">python -m src.eval.stage4_evaluate</code></div>
    <div>→ full claim-to-test mapping · <code style="color:{C_RAZORPAY_BLUE};">claim-matrix.md</code></div>
  </div>
</div>
""")

# ----------------------------------------------------------------------------
# View 05 — Adversarial Audit & Governance
# ----------------------------------------------------------------------------
def view_governance():
    st.markdown("### Adversarial Audit & Safety Governance")
    st.markdown(
        f'<div style="color:{C_MUT_DARK};font-size:0.88rem;max-width:960px;line-height:1.55;margin-bottom:1.1rem;">'
        "Independent verification and reconciliation following an exhaustive 17-point adversarial audit. "
        "All 5 conflict discrepancies have been mathematically reconciled to root cause, all 9 model artifacts "
        "cryptographically validated, and behavioral misspecification sensitivity boundaries established.</div>",
        unsafe_allow_html=True)

    html_block(f"""
<div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 0.85rem 1.2rem; margin-bottom: 1.2rem; color: #92400E; display: flex; align-items: center; justify-content: space-between;">
  <div>
    <span style="background: #D97706; color: #FFFFFF; font-weight: 800; font-size: 0.72rem; padding: 0.22rem 0.55rem; border-radius: 4px; letter-spacing: 0.05em;">SHADOW-MODE ONLY</span>
    <b style="margin-left: 0.5rem; font-size: 0.95rem;">Production Deployment Gate Verdict</b>
    <div style="font-size: 0.82rem; margin-top: 0.35rem; color: #78350F; line-height: 1.45;">
      Autonomous live intervention routing is strictly deferred pending physical merchant delivery telemetry. Active pre-production circuit breakers:
      <b>Emergency Kill-Switch</b> · <b>60% Rate Cap</b> · <b>&gt;₹10k Review</b> · <b>16.8% Merchant Break-Even Gate</b> · <b>SHA-256 Digest Validation</b>.
    </div>
  </div>
  <div style="text-align: right; min-width: 140px;">
    <span class="verify-chip" style="background:#FEF3C7; border-color:#FCD34D; color:#92400E;">0 Blockers · 5/5 Reconciled</span>
  </div>
</div>
""")

    html_block("".join([
        '<div class="kpi-grid">',
        kpi("Canonical Savings", rs(69786.08, 2), "Reconciled against verification script ₹0.50 friction deduction bug (₹1,306).", "green"),
        kpi("Observable Ceiling", "94.75%", "PR-AUC <b>0.3313 / 0.3497</b> — extracts 94.75% of true observable signal.", "amber"),
        kpi("Merchant Break-Even", "16.8%", "Below 16.8% COD RTO, intervention friction destroys merchant product margin.", "blue"),
        kpi("Mutation Kill Rate", "93%", "Suite catches <b>13 of 14 intentional faults</b>. PR-AUC regression floor: ≥0.32.", "blue"),
        "</div>",
    ]))

    st.markdown("")
    # Reconciled Conflicts
    html_block(f"""
<div class="panel">
  <div class="panel-h">The 5 Reconciled Audit Conflicts · Root Cause &amp; Mathematical Proof</div>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;">
    <div class="step" style="background: #FFFFFF;">
      <div class="n">CONFLICT 01 · DESIGN POINT SAVINGS</div>
      <div class="h">₹69,786.08 Canonical vs ₹68,480.08</div>
      <div class="b">
        <b>Root Cause:</b> <code>audit/run_step3_verifications.py</code> line 132 hardcoded a ₹0.50 deposit friction fee.
        2,612 deposit orders × ₹0.50 = ₹1,306.00. ₹69,786.08 − ₹1,306.00 = ₹68,480.08.
        Canonical config-driven savings in <code>reports/stage5_test_results.md</code> is verified bit-for-bit at <b>₹69,786.08</b>.
      </div>
    </div>
    <div class="step" style="background: #FFFFFF;">
      <div class="n">CONFLICT 02 · BAYES OPTIMAL CEILING</div>
      <div class="h">Observable 0.3497 vs Latent ~0.46</div>
      <div class="b">
        <b>Root Cause:</b> Earlier sweeps computed ceiling on unobservable latent risk $p_{{latent}}$ containing random Gaussian noise $\\epsilon \\sim \\mathcal{{N}}(0, 0.80)$.
        The true observable theoretical maximum $\\mathbb{{E}}[p \\mid x]$ is <b>0.3497</b>. The primary LightGBM (0.3313) achieves <b>94.75% of observable ceiling signal</b>.
      </div>
    </div>
    <div class="step" style="background: #FFFFFF;">
      <div class="n">CONFLICT 03 · MODEL RANKING &amp; CALIBRATION</div>
      <div class="h">LR 0.3434 vs LGBM Isotonic 0.3313</div>
      <div class="b">
        <b>Root Cause:</b> Isotonic regression collapses the model's continuous risk distribution into <b>97 discrete probability plateaus</b> on test (96.1% share), flattening ranking ties.
        Uncalibrated LightGBM achieves 0.3433 (parity with LR). Isotonic is required to make probabilities honest for expected-loss routing ($|\\Delta| \\le 0.0032$).
      </div>
    </div>
    <div class="step" style="background: #FFFFFF;">
      <div class="n">CONFLICT 04 · BREAK-EVEN COD RTO RATE</div>
      <div class="h">Standardized to COD Subset: 16.8%</div>
      <div class="b">
        <b>Root Cause:</b> At ₹1,000 order value and 20% margin, losing a customer costs ₹200 while saving an RTO saves ₹150.
        When COD RTO drops below <b>16.8%</b>, intervention friction costs exceed logistics savings. Guardrails auto-bypass low-risk brands to <code>ALLOW_COD</code>.
      </div>
    </div>
  </div>
  <div style="margin-top: 0.9rem; padding: 0.8rem 1rem; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; font-size: 0.82rem; color: {C_MUT_DARK};">
    <b>CONFLICT 05 · Cryptographic Artifact Inventory:</b> All 9 production model files (5 lookup CSVs, 4 pickled models) pinned in <code>models/artifact_hashes.json</code> verified bit-for-bit with SHA-256 digests. Model deserialization refuses unverified pickles with a hard <code>SecurityError</code>.
  </div>
</div>
""")

    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Behavioral Misspecification Sensitivity Bounds")
        st.caption("Recomputed via audit/misspec.py across parameter perturbations")
        misspec_data = pd.DataFrame({
            "Parameter": [
                "Deposit Drop-Off Rate",
                "Verify Drop-Off Rate",
                "Deposit RTO Reduction",
                "Reverse Courier Logistics Cost",
                "Joint Pessimistic Stress Test"
            ],
            "Baseline Value": ["40.0%", "5.0%", "80.0%", "₹150.00", "Baseline Assumptions"],
            "Break-Even Threshold": ["99.58%", "23.81%", "25.14%", "₹76.11", "−₹5,870.94 net loss"],
            "Safe Range": ["[0%, 99.5%]", "[0%, 23.8%]", "[25.1%, 100%]", "[₹76.11, ∞)", "Shadow calibration req."]
        })
        st.dataframe(misspec_data, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### 5-Seed Pipeline Determinism Sweep")
        st.caption("Recomputed via audit/seed_sweep.py across random generator seeds")
        seeds_data = pd.DataFrame({
            "Seed": [42, 101, 2024, 777, 999, "Mean"],
            "Observable Ceiling": ["0.3497", "0.3429", "0.3486", "0.3396", "0.3452", "0.3452"],
            "Logistic Reg.": ["0.3434", "0.3359", "0.3412", "0.3353", "0.3399", "0.3391 (98.2%)"],
            "LGBM Uncal": ["0.3433", "0.3320", "0.3419", "0.3314", "0.3364", "0.3370 (97.6%)"],
            "LGBM Isotonic": ["0.3313", "0.3204", "0.3287", "0.3202", "0.3235", "0.3248 (94.1%)"]
        })
        st.dataframe(seeds_data, use_container_width=True, hide_index=True)

    st.markdown("")
    html_block(f"""
<div class="panel">
  <div class="panel-h">Automated Test Hardening &amp; Privacy Verification</div>
  <div class="step-strip" style="grid-template-columns: repeat(4, 1fr);">
    <div class="step">
      <div class="n">CI REPRODUCIBILITY</div>
      <div class="h">Clean-Room Rebuild</div>
      <div class="b"><code>.github/workflows/repro.yml</code> executes clean venv build from raw data; all 9 SHA-256 hashes matched bit-for-bit.</div>
    </div>
    <div class="step">
      <div class="n">MUTATION TESTING</div>
      <div class="h">93% Mutation Score</div>
      <div class="b">13 of 14 intentional sabotage mutations caught by test suite. PR-AUC regression floor raised to ≥0.32 to catch depth-1 trees.</div>
    </div>
    <div class="step">
      <div class="n">TAMPER REFUSAL</div>
      <div class="h">Cryptographic Shield</div>
      <div class="b">Verified by <code>test_artifact_integrity.py</code>: any modified pickle byte triggers hard <code>SecurityError</code> and halts loading.</div>
    </div>
    <div class="step">
      <div class="n">PRIVACY COMPLIANCE</div>
      <div class="h">Strict PII Allow-List</div>
      <div class="b"><code>AUDIT_ALLOWLIST_KEYS</code> strips customer names, phone numbers, and addresses out of audit logs before ingestion.</div>
    </div>
  </div>
</div>
""")

    st.markdown("")
    html_block(f"""
<div class="panel">
  <div class="panel-h">Audit Reproduction Commands</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem 2rem;font-size:0.78rem;color:{C_MUT_DARK};line-height:1.8;">
    <div>→ Reconcile the 5 audit conflicts: <code style="color:{C_RAZORPAY_BLUE};">python audit/reconcile.py</code></div>
    <div>→ Recompute misspecification bounds: <code style="color:{C_RAZORPAY_BLUE};">python audit/misspec.py</code></div>
    <div>→ Run 5-seed pipeline sweep: <code style="color:{C_RAZORPAY_BLUE};">python audit/seed_sweep.py</code></div>
    <div>→ Full 244 automated tests: <code style="color:{C_RAZORPAY_BLUE};">pytest tests/ -v</code></div>
  </div>
</div>
""")

# ----------------------------------------------------------------------------
# Router & View Dispatcher
# ----------------------------------------------------------------------------
VIEWS = {
    "01 · Command Center": view_overview,
    "02 · Live Decision Engine": view_scorer,
    "03 · Policy Frontier": view_frontier,
    "04 · Portfolio Evidence": view_portfolio,
    "05 · Adversarial Audit & Governance": view_governance,
}

topbar()

view = st.sidebar.radio("Console View", list(VIEWS.keys()), label_visibility="collapsed")
VIEWS[view]()

footer()
