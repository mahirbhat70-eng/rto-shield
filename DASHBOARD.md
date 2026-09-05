# RTO Shield — Live Decision Console

An interactive, video-ready decision console built for the **Razorpay AI Buildathon (Track 02: AI Risk Manager)**.

---

## 1. Overview & Architecture

RTO Shield addresses the cash-on-delivery (COD) dilemma for Indian e-commerce merchants: **28% of COD orders return to origin**, costing merchants **~₹150 in lost logistics per return with zero revenue**. 

Rather than treating risk as a classification label or relying on rigid threshold cutoffs, RTO Shield is a **two-layer decision engine**:
1. **Layer 1 (Risk Estimation)**: Calibrated LightGBM model estimates $P(\text{RTO})$ per order.
2. **Layer 2 (Expected-Loss Engine)**: Deterministic, tested financial math prices 4 possible interventions per order:
   - `ALLOW_COD` (Ship as-is, zero friction)
   - `VERIFY_ADDRESS` (Call/OTP verification, ₹2 friction, −30% RTO, −5% conversion)
   - `REQUIRE_DEPOSIT` (Token deposit, −80% RTO, −40% conversion)
   - `PREPAID_ONLY` (Convert to prepaid, −55% RTO, −70% conversion)

The engine chooses the intervention that minimizes expected loss ($\operatorname{argmin} \text{EL}$):
$$\text{EL} = \text{friction} + \big(P(\text{RTO}) \times (1 - \text{reduction}) \times ₹150\big) - \big((1 - P(\text{RTO})) \times (1 - \text{drop}) \times 20\% \times \text{order\_value}\big)$$

---

## 2. Design System

Built to adhere to the requested Razorpay aesthetic:
- **Background**: Ultra-light slate (`#F8FAFC`)
- **Cards & Surfaces**: Pure white (`#FFFFFF`) with subtle slate borders (`#E2E8F0`) and soft elevation shadows
- **Typography**: Deep navy (`#0F172A`), muted slate (`#64748B`), and secondary slate (`#475569`)
- **Interactive Accents & Primary CTAs**: Vibrant Razorpay Blue (`#0052FF`) with hover transitions
- **Action Semantic Colors**:
  - `ALLOW_COD`: Emerald (`#059669`)
  - `VERIFY_ADDRESS`: Amber (`#D97706`)
  - `REQUIRE_DEPOSIT`: Rose/Coral (`#E11D48`)
  - `PREPAID_ONLY`: Violet (`#7C3AED`)
  - `PREPAID_PASSTHROUGH`: Slate (`#475569`)

---

## 3. Four Console Views (Aligned to Video Pitch Script)

| View | Pitch Act | Live Functionality |
|---|---|---|
| **01 · Command Center** | Act 1 & Act 3 (0:00–1:00) | Hero statement (*"We don't predict the coin flip. We price it."*), 8 headline KPI cards (₹71,741 expected savings, ₹69,786 realized, 2.0× threshold beat, 94.7% Bayes ceiling, 13.1% profit uplift, 100% Monte Carlo confidence), and 4-step pricing workflow. |
| **02 · Live Decision Engine** | Act 4 (1:30–3:00) | Instant presets (`VERIFY`, `ALLOW`, `DEPOSIT`, and `Random holdout order`), live pincode intelligence preview, `SCORE THIS ORDER` live timer (<100 ms), calibrated $P(\text{RTO})$ meter, **Expected-Loss Price Menu with ROUTED · ARGMIN highlight**, top-8 signed TreeSHAP bars, input intelligence, tamper-evident SHA-256 decision fingerprinting, and session `.jsonl` audit trail export. |
| **03 · Policy Frontier** | Act 2 (3:00–3:30) | Live recomputation on 3,673 COD orders (`val_cal.csv`): interactive Altair visualization comparing Binary VERIFY and Binary PREPAID curves against the horizontal Argmin Routing line. Displays best single cutoffs ($t=0.20$ and $t=0.48$) and the edge of the line (+₹18,331 / 5.9%). |
| **04 · Portfolio Evidence** | Act 3 (3:30–4:00) | Frozen Stage-5 results on 14,980 test orders: Expected vs. Realized savings (−2.7% error), Action distribution donut chart (81.6% touched, 0% PREPAID_ONLY), Monte Carlo 5,000-draw confidence band, Precision/Recall (0.2963 / 0.8555), Per-action calibration table ($|\Delta| \le 0.0032$), and copy-paste reproduction commands. |

---

## 4. Acceptance Gates & Verification Contract

Run the headless verification suite:
```bash
python scripts/acceptance_dashboard.py
```

Result:
```text
============================================================
RTO-SHIELD DASHBOARD ACCEPTANCE GATES
============================================================
[1/7] Testing VERIFY preset truth gate...
      PASS: P(RTO)=39.45%, action=VERIFY_ADDRESS, EL=-54.60
[2/7] Testing PREPAID passthrough...
      PASS: PREPAID_PASSTHROUGH confirmed with zeroed EL table.
[3/7] Testing guardrail rejection on out-of-distribution values...
      PASS: Guardrail rejected order_value=30000 as expected.
[4/7] Testing unknown pincode rejection...
      PASS: Unknown pincode cleanly rejected.
[5/7] Testing audit record fingerprinting & tamper detection...
      PASS: Audit fingerprint determinism & tamper-evidence verified.
[6/7] Testing dashboard.py compilation...
      PASS: dashboard.py compiled cleanly with zero syntax errors.
[7/7] Testing dashboard.py code hygiene and import integrity...
      PASS: Source imports frozen pipeline without staging or logic forking.
============================================================
ALL-PASS: All 7 acceptance gates passed successfully!
============================================================
```

And full CI test suite:
```bash
python -m pytest tests/ -q
# 60 passed
```

---

## 5. Deployment Guide (Streamlit Community Cloud)

1. Ensure changes are pushed to branch `dashboard-v2`.
2. Go to **share.streamlit.io** → Select repository `mahirbhat70-eng/rto-shield`.
3. Set **Branch**: `dashboard-v2`.
4. Set **Main file path**: `dashboard.py`.
5. Under **Advanced settings**: Select Python 3.12.
6. Click **Deploy**.
