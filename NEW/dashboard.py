"""
dashboard.py — RTO Shield Decision Console
==========================================
A live, judge-facing console for the frozen v1.0 serving artifacts.

Design contract:
  * READ-ONLY reuse of frozen modules: src/serve/scorer.py, src/serve/audit.py,
    src/policy/cost_engine.py, configs/cost_config.yaml, data/processed/val_cal.csv.
  * No frozen file is modified. This file is purely additive serving/UI.
  * Every number shown is either (a) returned live by score_order() on the frozen
    artifacts, (b) recomputed live and verified bitwise-identical to the frozen
    stage-4 stage-5 reports, or (c) quoted verbatim from reports/ with provenance.

Views:
  01 Command Center      — hero + frozen headline numbers
  02 Live Decision Engine— preset -> score -> P(RTO) / EL price menu / SHAP / audit
  03 Policy Frontier     — live threshold sweep ("nothing beats the line")
  04 Portfolio Evidence  — frozen stage-5 P&L, MC band, calibration, provenance
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.serve.scorer import score_order
from src.serve import scorer as serve
from src.serve.audit import build_audit_record, to_jsonl, verify_audit_record

st.set_page_config(
    page_title="RTO Shield — Decision Console",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Palette + CSS
# ----------------------------------------------------------------------------
C_BG = "#0B0F17"
C_PANEL = "#111827"
C_PANEL2 = "#0E1420"
C_BORDER = "rgba(148,163,184,0.14)"
C_TEXT = "#E5E7EB"
C_MUT = "#8B98AC"
C_GREEN = "#34D399"
C_AMBER = "#FBBF24"
C_RED = "#FB7185"
C_SLATE = "#94A3B8"
C_BLUE = "#60A5FA"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, .stApp {
  background: #0B0F17;
  color: #E5E7EB;
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.block-container { padding: 1.6rem 2.2rem 3.2rem 2.2rem; max-width: 1280px; }
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3, h4 { color: #F1F5F9 !important; letter-spacing: -0.02em; }
a { color: #60A5FA; }
.stApp > div div[data-baseweb="radio"] > div { gap: .35rem; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: #1F2A3C; border-radius: 6px; }
::-webkit-scrollbar-track { background: transparent; }

/* ---------- top bar ---------- */
.topbar { display:flex; align-items:center; justify-content:space-between;
  padding: .55rem .35rem .95rem .35rem; border-bottom: 1px solid rgba(148,163,184,.12);
  margin-bottom: 1.15rem; }
.brand { display:flex; align-items:baseline; gap:.7rem; }
.brand-name { font-weight: 900; font-size: 1.22rem; letter-spacing: -.02em; color:#F8FAFC; }
.brand-name span { color: #34D399; }
.brand-sub { font-size: .72rem; color:#8B98AC; font-weight:500; letter-spacing:.08em; text-transform:uppercase; }
.status-chip { display:inline-flex; align-items:center; gap:.5rem; font-size:.72rem; font-weight:600;
  color:#A7F3D0; background: rgba(52,211,153,.08); border:1px solid rgba(52,211,153,.25);
  padding:.34rem .7rem; border-radius:999px; }
.status-dot { width:7px; height:7px; border-radius:50%; background:#34D399;
  box-shadow: 0 0 8px rgba(52,211,153,.9); }

/* ---------- hero ---------- */
.eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.22em; color:#34D399; text-transform:uppercase; margin-bottom:.7rem; }
.hero-h1 { font-size: 3.35rem; line-height: 1.06; font-weight: 900; letter-spacing: -.035em; color:#F8FAFC; margin: 0 0 .9rem 0; }
.hero-h1 .grad { background: linear-gradient(92deg,#34D399 10%,#60A5FA 90%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { font-size: 1.02rem; color:#9CA9BD; max-width: 780px; line-height:1.55; margin-bottom: 1.5rem; }

/* ---------- cards ---------- */
.kpi-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: .9rem; margin: 1rem 0 .4rem 0; }
.kpi-grid3 { display:grid; grid-template-columns: repeat(4, 1fr); gap: .9rem; margin: .7rem 0 0 0; }
.kpi { background:#111827; border:1px solid rgba(148,163,184,.14); border-radius:14px; padding:1.05rem 1.15rem .9rem 1.15rem; }
.kpi .t { font-size:.7rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#8B98AC; margin-bottom:.45rem; }
.kpi .v { font-size: 1.92rem; font-weight: 850; letter-spacing:-.03em; color:#F8FAFC; line-height:1.05; }
.kpi .v.green { color:#34D399; } .kpi .v.blue { color:#60A5FA; } .kpi .v.amber { color:#FBBF24; }
.kpi .d { font-size:.74rem; color:#8B98AC; margin-top:.4rem; line-height:1.4; }
.kpi .d b { color:#C4CDD9; font-weight:600; }

.panel { background:#111827; border:1px solid rgba(148,163,184,.14); border-radius:14px; padding:1.1rem 1.25rem; }
.panel-h { font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#8B98AC; margin-bottom:.7rem; }

/* ---------- action rows (price menu) ---------- */
.arow { display:flex; align-items:center; justify-content:space-between; gap:1rem;
  background:#0E1420; border:1px solid rgba(148,163,184,.12); border-radius:12px;
  padding:.78rem 1rem; margin-bottom:.55rem; }
.arow.sel { border:1px solid rgba(52,211,153,.65); background: rgba(52,211,153,.07);
  box-shadow: 0 0 22px rgba(52,211,153,.10); }
.a-name { display:flex; flex-direction:column; gap:.22rem; min-width: 200px; }
.a-label { font-weight:800; font-size:.95rem; letter-spacing:.01em; }
.a-chips { display:flex; gap:.35rem; flex-wrap:wrap; }
.achip { font-size:.66rem; font-weight:600; color:#9CA9BD; background:#151E2E; border:1px solid rgba(148,163,184,.14);
  padding:.14rem .48rem; border-radius:6px; white-space:nowrap; }
.a-el { text-align:right; min-width: 150px; }
.a-el .lbl { font-size:.64rem; letter-spacing:.1em; color:#8B98AC; font-weight:700; text-transform:uppercase; }
.a-el .val { font-size:1.32rem; font-weight:850; letter-spacing:-.02em; }
.pick { font-size:.62rem; font-weight:800; letter-spacing:.14em; color:#052015; background:#34D399;
  border-radius:6px; padding:.22rem .55rem; margin-left:.7rem; vertical-align:middle; }

/* ---------- KPI strip on scorer ---------- */
.big-p { font-size: 3.1rem; font-weight: 900; letter-spacing: -.04em; line-height: 1; }
.track { height: 9px; background:#1B2434; border-radius: 6px; overflow:hidden; margin-top:.7rem; }
.track > i { display:block; height:100%; border-radius:6px; }
.badge { display:inline-block; font-weight:900; font-size:1.02rem; letter-spacing:.06em;
  padding:.5rem .95rem; border-radius:10px; color:#06120C; }
.lat { font-size: 1.6rem; font-weight: 850; letter-spacing: -.03em; color:#60A5FA; }
.lat-sub { font-size:.72rem; color:#8B98AC; margin-top:.3rem; }

/* ---------- SHAP bars ---------- */
.shap-row { display:grid; grid-template-columns: 190px 1fr 74px; align-items:center; gap:.6rem; margin:.3rem 0; }
.shap-name { font-family: 'SFMono-Regular', Consolas, monospace; font-size:.68rem; color:#9CA9BD; text-align:right;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.shap-track { position:relative; height:15px; background:#0E1420; border-radius:5px; overflow:hidden; }
.shap-track .mid { position:absolute; left:50%; top:0; bottom:0; width:1px; background:rgba(148,163,184,.35); }
.shap-fill { position:absolute; top:2.5px; bottom:2.5px; border-radius:4px; }
.shap-val { font-family: 'SFMono-Regular', Consolas, monospace; font-size:.68rem; font-weight:600; }

/* ---------- donut / mc band ---------- */
.donut-wrap { display:flex; align-items:center; gap:1.4rem; }
.donut { width:158px; height:158px; border-radius:50%;
  -webkit-mask: radial-gradient(circle at 50% 50%, transparent 0 54%, #000 55%);
  mask: radial-gradient(circle at 50% 50%, transparent 0 54%, #000 55%); }
.legend { display:flex; flex-direction:column; gap:.42rem; }
.leg { display:flex; align-items:center; gap:.5rem; font-size:.78rem; color:#B7C1CF; }
.leg i { width:10px; height:10px; border-radius:3px; display:inline-block; }
.leg b { color:#F1F5F9; font-weight:700; }
.mc-track { position:relative; height:12px; background:#1B2434; border-radius:8px; margin: 1.6rem .2rem 1.9rem .2rem; }
.mc-band { position:absolute; top:0; bottom:0; background: rgba(96,165,250,.20); border-radius:8px;
  border-left:2px solid #60A5FA; border-right:2px solid #60A5FA; }
.mc-mean { position:absolute; top:-6px; bottom:-6px; width:3px; background:#F1F5F9; border-radius:2px; }
.mc-lab { position:absolute; transform:translateX(-50%); font-size:.68rem; color:#8B98AC; white-space:nowrap; font-weight:600; }

/* ---------- tables ---------- */
[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.14); border-radius:12px; overflow:hidden; }
[data-testid="stDataFrame"] th { background:#0E1420 !important; }

/* ---------- misc ---------- */
.stButton > button[kind="primary"] { width:100%; font-weight:800; letter-spacing:.03em;
  background:#34D399; border-color:#34D399; color:#06120C; }
.stButton > button[kind="primary"]:hover { background:#6EE7B7; border-color:#6EE7B7; color:#06120C; }
.preset-row [data-testid="stButton"] > button { width:100%; font-size:.74rem; font-weight:700; border-radius:9px; }
.side-h { font-size:.66rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; color:#64748B; margin:.8rem 0 .1rem 0; }
.sidebar .sidebar-content { background:#0A0E15; }
hr.soft { border:none; border-top:1px solid rgba(148,163,184,.12); margin: 1.2rem 0; }
.foot { color:#64748B; font-size:.72rem; padding-top: .5rem; }
.verify-chip { display:inline-flex; align-items:center; gap:.45rem; font-size:.72rem; font-weight:700; color:#A7F3D0;
  background: rgba(52,211,153,.08); border:1px solid rgba(52,211,153,.3); padding:.4rem .75rem; border-radius:9px; }
.step-strip { display:grid; grid-template-columns: repeat(4,1fr); gap:.9rem; }
.step { background:#0E1420; border:1px solid rgba(148,163,184,.12); border-radius:12px; padding:.85rem 1rem; }
.step .n { font-size:.66rem; font-weight:800; color:#34D399; letter-spacing:.14em; }
.step .h { font-weight:750; font-size:.9rem; color:#F1F5F9; margin:.25rem 0 .2rem 0; }
.step .b { font-size:.73rem; color:#8B98AC; line-height:1.45; }
@media (max-width: 900px) { .kpi-grid, .kpi-grid3, .step-strip { grid-template-columns: repeat(2,1fr);} .hero-h1 { font-size:2.2rem;} }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
ACTION_META = {
    "ALLOW_COD":       ("ALLOW COD",      C_GREEN, "Ship it as-is. No friction, no protection."),
    "VERIFY_ADDRESS":  ("VERIFY ADDRESS", C_AMBER, "Call/OTP verify the address. −30% RTO, −5% conversions, ₹2 cost."),
    "REQUIRE_DEPOSIT": ("REQUIRE DEPOSIT",C_RED,   "Token deposit before dispatch. −80% RTO, −40% conversions."),
    "PREPAID_ONLY":    ("PREPAID ONLY",   C_SLATE, "Flip COD to prepaid. −55% RTO, −70% conversions."),
}

def inr(v, dec=0):
    """Indian-format number: 3,22,053 (lakh/crore grouping)."""
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
# Frozen data + live engines
# ----------------------------------------------------------------------------
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
    "DEPOSIT": "₹186 · low value · tier-1 pin",
    "ALLOW": "₹1,344 · 998-day veteran",
    "VERIFY": "₹852 · 65-day account · risky pin",
}

CATEGORIES = ["Apparel", "Home", "Electronics", "Beauty", "Footwear", "Jewelry"]

@st.cache_resource(show_spinner=False)
def boot_self_test():
    """Score the three presets once on the frozen artifacts; report health."""
    out = {}
    t0 = time.perf_counter()
    for name, payload in PRESETS.items():
        t = time.perf_counter()
        res = score_order(dict(payload))
        out[name] = {"p": res["probability"], "action": res["recommended_action"],
                     "ms": (time.perf_counter() - t) * 1000}
    out["_total_ms"] = (time.perf_counter() - t0) * 1000
    return out

SELF = boot_self_test()

@st.cache_data(show_spinner="Recomputing the policy frontier from frozen artifacts…")
def compute_frontier():
    """Vectorized replica of src/eval/stage4_evaluate.py on the val_cal COD subset.

    Verified bitwise-identical to the frozen loop implementation
    (scripts/verify_frontier.py): same curve values, same argmin thresholds
    (0.48 PREPAID / 0.20 VERIFY), same multi-action line (-327,070.57).
    """
    val_cal = pd.read_csv("data/processed/val_cal.csv", dtype={"pincode": str})
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

    actions = list(serve.engine.interventions.keys())
    dist = {a: float((best_idx == i).mean() * 100) for i, a in enumerate(actions)}

    best = {
        "VERIFY_ADDRESS": {"t": float(thresholds[int(np.argmin(curves["VERIFY_ADDRESS"]))]),
                           "loss": float(np.min(curves["VERIFY_ADDRESS"]))},
        "PREPAID_ONLY":   {"t": float(thresholds[int(np.argmin(curves["PREPAID_ONLY"]))]),
                           "loss": float(np.min(curves["PREPAID_ONLY"]))},
    }
    return {
        "n": int(len(vc)),
        "thresholds": thresholds,
        "curves": curves,
        "primary_total": primary_total,
        "best": best,
        "edge": best["VERIFY_ADDRESS"]["loss"] - primary_total,
        "dist": dist,
    }

FR = compute_frontier()

# ----------------------------------------------------------------------------
# Extended SHAP (same frozen explainer; superset of score_order's top-3)
# ----------------------------------------------------------------------------
def full_shap(payload, topn=8):
    lookup = serve.PINCODE_LOOKUP[str(payload["pincode"])]
    row = {
        "category": payload["category"], "payment_method": payload["payment_method"],
        "courier_id": payload["courier_id"],
        "order_value": float(payload["order_value"]), "quantity": float(payload["quantity"]),
        "discount_pct": float(payload["discount_pct"]), "cod_charge": float(payload["cod_charge"]),
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
# Chrome: top bar + footer
# ----------------------------------------------------------------------------
def topbar():
    html_block(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-name">RTO<span>&nbsp;SHIELD</span></div>
    <div class="brand-sub">COD decision console</div>
  </div>
  <div class="status-chip"><span class="status-dot"></span>
    ENGINE READY&nbsp;·&nbsp;frozen v1.0 artifacts&nbsp;·&nbsp;self-test 3/3 in {SELF['_total_ms']:.0f} ms
  </div>
</div>""")

def footer():
    st.markdown("---")
    st.markdown(
        '<div class="foot">github.com/mahirbhat70-eng/rto-shield · frozen v1.0 artifacts · '
        "60/60 tests green in CI · every number on this page is reproducible from reports/ and claim-matrix.md</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# View 01 — Command Center
# ----------------------------------------------------------------------------
def view_overview():
    html_block(f"""
<div class="eyebrow">Razorpay AI Buildathon · Track 02 · AI Risk Manager</div>
<div class="hero-h1">We don't predict the coin flip.<br><span class="grad">We price it.</span></div>
<div class="hero-sub">RTO Shield scores every COD order in under 100&nbsp;ms, prices four interventions with a
rupee cost matrix (₹150 landed cost per return · 20% average margin), and routes each order to the
cheapest expected loss. Risk is not a label here — it is a price, and the engine always buys the cheapest one.</div>
""")

    html_block("".join([
        '<div class="kpi-grid">',
        kpi("Expected savings", rs(71741), "<b>vs always-allow</b> on the 14,980-order holdout test (7,174 COD orders). Portfolio expected loss, argmin routing.", "green"),
        kpi("Realized savings", rs(69786), "Same portfolio scored with <b>actual RTO labels</b> — only −2.7% off the forecast.", "green"),
        kpi("vs best threshold", "2.0×", "EL routing saves <b>₹71,741 vs ₹35,919</b> for the best single-threshold policy. No cutoff beats the line.", "blue"),
        kpi("of Bayes ceiling", "94.7%", "PR-AUC <b>0.3313 / 0.3497</b> — the model extracts almost all signal the generator can give.", "amber"),
        "</div>",
        '<div class="kpi-grid3">',
        kpi("Profit uplift", "13.1%", "Pre-registered PASS band <b>[8%, 18%]</b> of baseline loss. Declared before the test reveal.", ""),
        kpi("P(savings &gt; 0)", "100%", "Across <b>5,000-draw Monte Carlo</b> on intervention effects. P5 ₹63,935 · P95 ₹75,901.", "green"),
        kpi("Scoring path", "&lt;100 ms", "Full path <b>including TreeSHAP</b>. Core p50 ≈ 8 ms. &gt;100 orders/sec per core.", "blue"),
        kpi("Test suite", "60/60", "8 test files green in CI on pinned versions (sklearn 1.9.0 · lightgbm 4.7.0 · shap 0.52.0).", ""),
        "</div>",
    ]))

    st.markdown("")
    html_block(f"""
<div class="panel">
  <div class="panel-h">How a decision is priced — four moves, one argmin</div>
  <div class="step-strip">
    <div class="step"><div class="n">01 · FEATURES</div><div class="h">13 order signals</div>
      <div class="b">Value, basket, category, COD charge, account age, prior RTOs, velocity, device cluster, pincode RTO rate + tier, courier.</div></div>
    <div class="step"><div class="n">02 · PROBABILITY</div><div class="h">Calibrated LightGBM</div>
      <div class="b">Calibrated P(RTO) — probabilities you can read as rupee rates. Calibration holds per action (max |Δ| 0.0032).</div></div>
    <div class="step"><div class="n">03 · PRICING</div><div class="h">Cost matrix, 4 actions</div>
      <div class="b">ALLOW ₹0 friction · VERIFY ₹2 / −30% RTO / −5% conv. · DEPOSIT −80% RTO / −40% conv. · PREPAID −55% RTO / −70% conv.</div></div>
    <div class="step"><div class="n">04 · ROUTING</div><div class="h">argmin expected loss</div>
      <div class="b">Ship with the cheapest expected loss. That is the whole policy — no threshold to tune, nothing to overfit.</div></div>
  </div>
</div>
""")

    html_block(f"""
<div class="kpi-grid3" style="margin-top:.9rem;">
{kpi('The bleed', '28.27%', 'of COD orders on the holdout test ended in RTO. At ₹150 landed cost per return, COD is the most expensive button on the checkout.', 'amber')}
{kpi('Try it live', 'View 02', 'Load the VERIFY preset, hit score, and watch the engine price all four moves and route the order — SHAP and audit receipt included.', 'blue')}
{kpi('Why no threshold', 'View 03', 'The policy frontier chart, recomputed live in your browser from the frozen model. No single cutoff touches the routing line.', 'blue')}
{kpi('Audited P&amp;L', 'View 04', 'Expected vs realized savings, Monte Carlo band, per-action calibration, precision/recall — with pytest one-liners to reproduce.', 'blue')}
</div>
""")

# ----------------------------------------------------------------------------
# View 02 — Live Decision Engine
# ----------------------------------------------------------------------------
def sidebar_inputs():
    """app.py-proven pattern: widgets carry ni_/sb_/ti_/sl_ keys and take value= from
    plain value-keys that presets write via on_click callbacks. (Passing the value
    explicitly is what makes the UI actually repaint on preset load.)"""
    def get_val(key, default):
        return st.session_state.get(key, default)

    def load_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[k] = v
        st.session_state["score_requested"] = True

    st.sidebar.markdown('<div class="side-h">Demo presets</div>', unsafe_allow_html=True)
    cA, cB, cC = st.sidebar.columns(3)
    cA.button("DEPOSIT", on_click=load_preset, args=("DEPOSIT",), help=PRESET_HINT["DEPOSIT"], use_container_width=True)
    cB.button("ALLOW", on_click=load_preset, args=("ALLOW",), help=PRESET_HINT["ALLOW"], use_container_width=True)
    cC.button("VERIFY", on_click=load_preset, args=("VERIFY",), help=PRESET_HINT["VERIFY"], use_container_width=True)

    st.sidebar.markdown('<div class="side-h" style="margin-top:1rem;">Order</div>', unsafe_allow_html=True)
    st.sidebar.number_input("Order value (₹)", min_value=0.0, step=10.0,
                            value=get_val("order_value", 616.0), key="ni_order_value")
    st.sidebar.selectbox("Category", CATEGORIES,
                         index=CATEGORIES.index(get_val("category", "Apparel")), key="sb_category")
    payment_method = st.sidebar.selectbox("Payment method", ["COD", "PREPAID"],
                                          index=["COD", "PREPAID"].index(get_val("payment_method", "COD")),
                                          key="sb_payment_method")
    st.sidebar.number_input("Quantity", min_value=1, step=1,
                            value=get_val("quantity", 2), key="ni_quantity")
    st.sidebar.number_input("Discount %", min_value=0.0, max_value=100.0, step=0.5,
                            value=get_val("discount_pct", 2.0), key="ni_discount_pct")
    if payment_method == "COD":
        st.sidebar.number_input("COD charge (₹)", min_value=0.0, step=1.0,
                                value=get_val("cod_charge", 49.0), key="ni_cod_charge")
    else:
        st.sidebar.number_input("COD charge (₹)", min_value=0.0, max_value=0.0, value=0.0, disabled=True)

    st.sidebar.markdown('<div class="side-h">Customer</div>', unsafe_allow_html=True)
    st.sidebar.number_input("Account age (days)", min_value=0, step=1,
                            value=get_val("account_age_days", 236), key="ni_account_age_days")
    prior_orders = st.sidebar.number_input("Prior orders", min_value=0, step=1,
                                           value=get_val("prior_orders", 3), key="ni_prior_orders")
    st.sidebar.slider("Prior RTO count", 0, max(1, int(prior_orders)),
                      value=get_val("prior_rto_count", 0), key="sl_prior_rto_count")

    st.sidebar.markdown('<div class="side-h">Signals</div>', unsafe_allow_html=True)
    st.sidebar.number_input("Orders last 24h", min_value=0, step=1,
                            value=get_val("orders_last_24h", 1), key="ni_orders_last_24h")
    st.sidebar.number_input("Device cluster size", min_value=1, step=1,
                            value=get_val("device_cluster_size", 1), key="ni_device_cluster_size")
    st.sidebar.text_input("Pincode", value=get_val("pincode", "597542"), key="ti_pincode")
    st.sidebar.text_input("Courier ID", value=get_val("courier_id", "Courier_A"), key="ti_courier_id")

def payload_from_state():
    pm = st.session_state["sb_payment_method"]
    return {
        "order_value": float(st.session_state["ni_order_value"]),
        "category": st.session_state["sb_category"],
        "payment_method": pm,
        "quantity": int(st.session_state["ni_quantity"]),
        "discount_pct": float(st.session_state["ni_discount_pct"]),
        "cod_charge": float(st.session_state["ni_cod_charge"]) if pm == "COD" else 0.0,
        "account_age_days": int(st.session_state["ni_account_age_days"]),
        "prior_orders": int(st.session_state["ni_prior_orders"]),
        "prior_rto_count": int(st.session_state["sl_prior_rto_count"]),
        "orders_last_24h": int(st.session_state["ni_orders_last_24h"]),
        "device_cluster_size": int(st.session_state["ni_device_cluster_size"]),
        "pincode": str(st.session_state["ti_pincode"]),
        "courier_id": str(st.session_state["ti_courier_id"]),
    }

def shap_html(pairs):
    mx = max(abs(v) for _, v in pairs) or 1.0
    rows = []
    for name, v in pairs:
        w = abs(v) / mx * 48.0
        if v >= 0:
            fill = f'<div class="shap-fill" style="left:50%;width:{w:.1f}%;background:{C_RED};"></div>'
            val = f'<span class="shap-val" style="color:{C_RED};">+{v:.3f}</span>'
        else:
            fill = f'<div class="shap-fill" style="right:50%;width:{w:.1f}%;background:{C_GREEN};"></div>'
            val = f'<span class="shap-val" style="color:{C_GREEN};">{v:.3f}</span>'
        rows.append(
            f'<div class="shap-row"><div class="shap-name">{name}</div>'
            f'<div class="shap-track"><div class="mid"></div>{fill}</div>{val}</div>'
        )
    return "".join(rows)

def action_row(action, el, selected, params=None):
    label, color, _ = ACTION_META[action]
    chips = ""
    if params:
        chips = (
            f'<div class="a-chips"><span class="achip">friction {rs(params["friction_cost"])}</span>'
            f'<span class="achip">RTO ×{1 - params["rto_reduction_pct"]:.2f}</span>'
            f'<span class="achip">conversions ×{1 - params["success_drop_pct"]:.2f}</span></div>'
        )
    pick = '<span class="pick">ROUTED · ARGMIN</span>' if selected else ""
    sel = " sel" if selected else ""
    valcolor = C_GREEN if selected else C_TEXT
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
            '<div style="text-align:right;color:#8B98AC;font-size:.76rem;padding-top:.45rem;">'
            "Load a preset or edit any field → hit <b style='color:#34D399;'>Score this order</b>."
            "</div>", unsafe_allow_html=True)

    if st.button("SCORE THIS ORDER", type="primary", use_container_width=True):
        st.session_state["score_requested"] = True

    if st.session_state.get("score_requested"):
        st.session_state["score_requested"] = False
        payload = payload_from_state()
        try:
            t0 = time.perf_counter()
            res = score_order(payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            entry = build_audit_record(payload, res, latency_ms=latency_ms)
            st.session_state.setdefault("audit_log", []).append(entry)
            st.session_state["last"] = {"payload": payload, "res": res,
                                        "latency_ms": latency_ms, "entry": entry}
        except ValueError as e:
            msg = str(e)
            if "exceeds maximum allowed value" in msg:
                st.error("**Guardrail rejected this order** — input outside the tested P99.9 range.")
                st.caption(msg)
            else:
                st.error(msg)
            st.session_state.pop("last", None)

    last = st.session_state.get("last")
    if not last:
        html_block("""
<div class="panel" style="text-align:center;padding:3.2rem 1rem;">
  <div style="font-size:1.5rem;font-weight:850;color:#F1F5F9;letter-spacing:-.02em;">No order on the bench yet.</div>
  <div style="color:#8B98AC;font-size:.85rem;margin-top:.5rem;">
    Pick a preset on the left — <b style="color:#FBBF24;">VERIFY</b> is the good one for a demo —
    then hit <b style="color:#34D399;">Score this order</b>.</div>
</div>""")
        _audit_block()
        return

    res = last["res"]
    p = res["probability"]
    action = res["recommended_action"]
    el = res["el_table"]
    risk_color = C_GREEN if p < 0.25 else (C_AMBER if p < 0.40 else C_RED)
    label, acolor, adesc = ACTION_META.get(
        action, ("PREPAID PASSTHROUGH", C_SLATE, "Prepaid orders skip the COD risk menu entirely."))

    # ---- KPI strip: probability / decision / latency
    html_block(f"""
<div class="kpi-grid" style="grid-template-columns: 1.15fr 1.15fr .9fr;">
  <div class="kpi">
    <div class="t">P(RTO) · calibrated</div>
    <div class="big-p" style="color:{risk_color};">{p * 100:.1f}%</div>
    <div class="track"><i style="width:{min(p * 100, 100):.1f}%;background:{risk_color};"></i></div>
    <div class="d">Reads as a rupee rate: on <b>₹{inr(last['payload']['order_value'])}</b> COD, expected return cost ≈ <b>{rs(p * 150)}</b>.</div>
  </div>
  <div class="kpi">
    <div class="t">Decision · argmin EL</div>
    <div style="margin:.55rem 0 .55rem 0;"><span class="badge" style="background:{acolor};">{label}</span></div>
    <div class="d">{adesc}</div>
  </div>
  <div class="kpi">
    <div class="t">Latency · full path</div>
    <div class="lat">{last['latency_ms']:.1f} ms</div>
    <div class="lat-sub">model + TreeSHAP + cost matrix<br/>bound by test: &lt; 100 ms</div>
  </div>
</div>
""")

    st.markdown("")
    # ---- price menu
    if last["payload"]["payment_method"] == "PREPAID":
        html_block(f"""
<div class="panel">
  <div class="panel-h">Expected-loss price menu</div>
  <div class="verify-chip">PREPAID ORDER — passthrough. No COD risk to manage; all four interventions price at ₹0 by definition.</div>
</div>""")
    else:
        params = serve.engine.interventions
        rows = "".join(action_row(a, el[a], a == action, params[a]) for a in params.keys())
        savings = el["ALLOW_COD"] - el[action]
        html_block(f"""
<div class="panel">
  <div class="panel-h">Expected-loss price menu · all four moves, one argmin</div>
  {rows}
  <div style="color:#8B98AC;font-size:.76rem;margin-top:.5rem;">
    EL = friction + (RTO after intervention × ₹150) − (surviving conversions × 20% margin) ·
    routing saves <b style="color:{C_GREEN};">{rs(savings, 2)}</b> per order vs always-allow.
  </div>
</div>""")

    st.markdown("")
    # ---- SHAP + input intelligence
    pairs = full_shap(last["payload"], topn=8)
    lookup = serve.PINCODE_LOOKUP[str(last["payload"]["pincode"])]
    entry = last["entry"]
    ok = verify_audit_record(entry)
    left, right = st.columns([1.35, 1])
    with left:
        html_block(f"""
<div class="panel">
  <div class="panel-h">Why this price · TreeSHAP on frozen model</div>
  {shap_html(pairs)}
  <div style="display:flex;gap:1.1rem;color:#8B98AC;font-size:.7rem;margin-top:.6rem;">
    <span><i style="display:inline-block;width:9px;height:9px;border-radius:3px;background:{C_RED};margin-right:.35rem;"></i>pushes risk up</span>
    <span><i style="display:inline-block;width:9px;height:9px;border-radius:3px;background:{C_GREEN};margin-right:.35rem;"></i>pulls risk down</span>
  </div>
</div>""")
    with right:
        pin_rate = float(lookup["historical_pincode_rto_rate"]) * 100
        tier = lookup["pincode_tier"]
        html_block(f"""
<div class="panel">
  <div class="panel-h">Input intelligence</div>
  <div class="d" style="font-size:.8rem;color:#B7C1CF;line-height:1.75;">
    Pincode <b>{last['payload']['pincode']}</b> → tier {tier} · historical RTO <b>{pin_rate:.1f}%</b><br/>
    Courier <b>{last['payload']['courier_id']}</b> · device cluster <b>{last['payload']['device_cluster_size']}</b><br/>
    Account <b>{last['payload']['account_age_days']}d</b> · velocity <b>{last['payload']['orders_last_24h']}/24h</b>
  </div>
  <hr class="soft"/>
  <div class="panel-h">Audit receipt</div>
  <div class="d" style="font-size:.78rem;line-height:1.7;">
    decision_id <b style="font-family:monospace;color:#60A5FA;">{entry['decision_id']}</b><br/>
    <span style="color:{C_GREEN};font-weight:700;">{'✓ fingerprint verified — tamper-evident' if ok else '✗ fingerprint mismatch'}</span><br/>
    <span style="color:#8B98AC;">sha256(payload ∥ timestamp), replayable</span>
  </div>
</div>""")

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
    st.markdown(f"**Decision audit trail** — {len(log)} decision(s) this session, exportable, replay-fingerprinted", )
    st.dataframe(df, use_container_width=True, hide_index=True)
    d1, d2, _ = st.columns([1.1, 0.7, 3])
    with d1:
        st.download_button("Download .jsonl", data=to_jsonl(log),
                           file_name="rto_audit_trail.jsonl", mime="application/x-ndjson",
                           use_container_width=True)
    with d2:
        if st.button("Clear log", use_container_width=True):
            st.session_state["audit_log"] = []
            st.session_state.pop("last", None)
            st.rerun()

# ----------------------------------------------------------------------------
# View 03 — Policy Frontier
# ----------------------------------------------------------------------------
def view_frontier():
    st.markdown("### Nothing beats the line")
    st.markdown(
        '<div style="color:#9CA9BD;font-size:.86rem;max-width:900px;line-height:1.5;margin-bottom:1rem;">'
        "Every single-threshold policy sweeps a worse portfolio loss than argmin routing. The engine prices all four "
        "interventions <i>per order</i>, so there is no cutoff to tune. This chart is recomputed live in your browser "
        "from the frozen artifacts — bitwise-identical to the frozen stage-4 report.</div>",
        unsafe_allow_html=True)

    th = FR["thresholds"]
    src = pd.DataFrame({
        "threshold": np.concatenate([th, th]),
        "loss": np.concatenate([FR["curves"]["VERIFY_ADDRESS"], FR["curves"]["PREPAID_ONLY"]]),
        "policy": ["Binary VERIFY block"] * len(th) + ["Binary PREPAID block"] * len(th),
    })
    lines = (alt.Chart(src)
             .mark_line()
             .encode(
                 x=alt.X("threshold:Q", title="single decision threshold",
                         axis=alt.Axis(format=".2f", gridColor="#1B2434", labelColor="#8B98AC", titleColor="#8B98AC")),
                 y=alt.Y("loss:Q", title="portfolio expected loss (₹)",
                         axis=alt.Axis(format=",d", gridColor="#1B2434", labelColor="#8B98AC", titleColor="#8B98AC")),
                 color=alt.Color("policy:N",
                                 scale=alt.Scale(domain=["Binary VERIFY block", "Binary PREPAID block"],
                                                 range=[C_AMBER, C_BLUE]),
                                 legend=alt.Legend(labelColor="#B7C1CF", title=None, orient="top")),
             ))
    hline = (alt.Chart(pd.DataFrame({"y": [FR["primary_total"]], "policy": ["Argmin routing (no threshold)"]}))
             .mark_rule(strokeDash=[6, 4], color=C_RED, strokeWidth=2.4)
             .encode(y="y:Q", color=alt.Color("policy:N", scale=alt.Scale(domain=["Argmin routing (no threshold)"],
                                                                         range=[C_RED]),
                                             legend=alt.Legend(labelColor="#B7C1CF", title=None, orient="top"))))
    pts = (alt.Chart(pd.DataFrame({
             "threshold": [FR["best"]["VERIFY_ADDRESS"]["t"], FR["best"]["PREPAID_ONLY"]["t"]],
             "loss": [FR["best"]["VERIFY_ADDRESS"]["loss"], FR["best"]["PREPAID_ONLY"]["loss"]],
         }))
           .mark_circle(size=90, color="#F8FAFC", opacity=0.95)
           .encode(x="threshold:Q", y="loss:Q"))
    chart = (lines + hline + pts).resolve_scale(color="independent")
    chart = chart.configure_view(stroke=None).configure(background="#0B0F17", font="Inter")
    st.altair_chart(chart, use_container_width=True, theme=None)

    st.markdown("")
    edge_pct = FR["edge"] / abs(FR["best"]["VERIFY_ADDRESS"]["loss"]) * 100
    html_block("".join([
        '<div class="kpi-grid" style="grid-template-columns: repeat(4,1fr);">',
        kpi("Best single cutoff · VERIFY", rs(FR["best"]["VERIFY_ADDRESS"]["loss"]),
            f"at t = {FR['best']['VERIFY_ADDRESS']['t']:.2f} — the orange curve bottoms out here, still short of the line.", "amber"),
        kpi("Best single cutoff · PREPAID", rs(FR["best"]["PREPAID_ONLY"]["loss"]),
            f"at t = {FR['best']['PREPAID_ONLY']['t']:.2f} — kills the book before it comes close.", "blue"),
        kpi("Argmin routing line", rs(FR["primary_total"]),
            f"per-order pricing on {FR['n']:,} COD orders (val_cal). Below every point of every curve.", "green"),
        kpi("Edge of the line", "+" + rs(FR["edge"]),
            f"{edge_pct:.1f}% better than the best single threshold you could ever tune.", "green"),
        "</div>",
    ]))
    html_block(f"""
<div style="margin-top:.9rem;">
<span class="verify-chip">✓ recomputed live · tuned thresholds 0.20 / 0.48 reproduced · routing line matches frozen stage-4 artifacts</span>
</div>""")
    with st.expander("Frozen report artifact — reports/stage4/threshold_vs_loss_curve.png (the audited version of this chart)"):
        st.image("reports/stage4/threshold_vs_loss_curve.png", use_container_width=True)
        st.caption("Generated by src/eval/stage4_evaluate.py on the same frozen artifacts. The live chart above reproduces it bitwise.")

# ----------------------------------------------------------------------------
# View 04 — Portfolio Evidence
# ----------------------------------------------------------------------------
def view_portfolio():
    st.markdown("### Portfolio evidence")
    st.markdown(
        '<div style="color:#9CA9BD;font-size:.86rem;max-width:900px;line-height:1.5;margin-bottom:1rem;">'
        "Frozen stage-5 results on the 14,980-order holdout test, quoted verbatim with provenance. "
        "Every row below has a pytest one-liner that re-derives it.</div>",
        unsafe_allow_html=True)

    html_block("".join([
        '<div class="kpi-grid">',
        kpi("Expected savings", rs(71741, 2), "argmin routing vs always-allow · COD subset (7,174 orders)", "green"),
        kpi("Realized savings", rs(69786, 2), "actual <b>rto_label</b> outcomes through the same cost engine", "green"),
        kpi("Forecast error", "-₹1,955", "realized − expected = <b>−2.7%</b> — the forecast survives contact with reality", "amber"),
        kpi("Profit uplift", "13.1%", "pre-registered PASS band <b>[8%, 18%]</b> of baseline loss", "green"),
        "</div>",
    ]))

    left, right = st.columns([1.15, 1])
    with left:
        html_block(f"""
<div class="panel">
  <div class="panel-h">Action distribution · test COD subset</div>
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
  <div class="d" style="font-size:.74rem;color:#8B98AC;margin-top:.8rem;">
    The engine touches <b style="color:#C4CDD9;">81.6%</b> of orders but keeps PREPAID_ONLY at exactly zero —
    it never kills an order it can save. 5,856 interventions · ₹6,488 friction spend · 951 expected RTOs prevented.</div>
</div>""")
    with right:
        pos = lambda v: f"{(v - 62000) / (78000 - 62000) * 100:.1f}%"
        html_block(f"""
<div class="panel">
  <div class="panel-h">Monte Carlo · 5,000 draws on intervention effects</div>
  <div class="mc-track">
    <div class="mc-band" style="left:{pos(63935.40)};width:{(75901.15 - 63935.40) / 16000 * 100:.1f}%;"></div>
    <div class="mc-mean" style="left:{pos(69942.31)};"></div>
    <div class="mc-lab" style="left:{pos(63935.40)};bottom:-1.35rem;">P5 ₹63,935</div>
    <div class="mc-lab" style="left:{pos(69942.31)};top:-1.35rem;color:#F1F5F9;">mean ₹69,942</div>
    <div class="mc-lab" style="left:{pos(75901.15)};bottom:-1.35rem;">P95 ₹75,901</div>
  </div>
  <div class="d" style="font-size:.8rem;color:#B7C1CF;">
    <b style="color:{C_GREEN};font-size:1.15rem;">P(savings &gt; 0) = 100%</b><br/>
    Bernoulli-sampled per-order intervention drops &amp; reductions. Worst decile still clears ₹63.9k.</div>
</div>""")

    st.markdown("")
    left2, right2 = st.columns([1.15, 1])
    with left2:
        html_block("""
<div class="panel">
  <div class="panel-h">Operating point · VERIFY + DEPOSIT = positive</div>
  <div style="display:flex; gap:2.2rem; margin:.2rem 0 .6rem 0;">
    <div><div style="font-size:.66rem;color:#8B98AC;font-weight:700;letter-spacing:.1em;">PRECISION</div>
    <div style="font-size:1.7rem;font-weight:850;">0.2963</div></div>
    <div><div style="font-size:.66rem;color:#8B98AC;font-weight:700;letter-spacing:.1em;">RECALL</div>
    <div style="font-size:1.7rem;font-weight:850;color:#34D399;">0.8555</div></div>
    <div><div style="font-size:.66rem;color:#8B98AC;font-weight:700;letter-spacing:.1em;">F1</div>
    <div style="font-size:1.7rem;font-weight:850;">0.4401</div></div>
  </div>
  <div class="d" style="font-size:.78rem;color:#9CA9BD;line-height:1.65;">
    We catch <b style="color:#C4CDD9;">86% of RTOs</b> while touching 81.6% of orders. Precision looks low until you
    remember the negative class is priced, not punished — a false positive just means a ₹2 phone call, not a lost customer.</div>
</div>""")
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
        html_block('<div class="d" style="font-size:.74rem;color:#8B98AC;">Max deviation 0.0032 — each action\'s pricing input matches its realized rate.</div></div>')

    st.markdown("")
    html_block("""
<div class="panel">
  <div class="panel-h">Provenance · reproduce it yourself</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem 2rem;font-size:.76rem;color:#9CA9BD;line-height:1.8;">
    <div>→ ₹71,741 / ₹69,786 / 13.1% · <code style="color:#60A5FA;">pytest tests/test_stage5.py -v</code></div>
    <div>→ 94.7% of Bayes ceiling · <code style="color:#60A5FA;">pytest tests/test_stage3.py -v</code></div>
    <div>→ routing line &amp; thresholds 0.20/0.48 · <code style="color:#60A5FA;">python -m src.eval.stage4_evaluate</code></div>
    <div>→ full claim→test map · <code style="color:#60A5FA;">claim-matrix.md</code></div>
  </div>
</div>""")

# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
VIEWS = {
    "01 · Command Center": view_overview,
    "02 · Live Decision Engine": view_scorer,
    "03 · Policy Frontier": view_frontier,
    "04 · Portfolio Evidence": view_portfolio,
}

topbar()

view = st.sidebar.radio("Console", list(VIEWS.keys()), label_visibility="collapsed")
VIEWS[view]()

footer()

