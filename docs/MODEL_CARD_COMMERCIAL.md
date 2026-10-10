# Commercial Model Card: FlipPrice AI Decision Engine

> **Version:** 2.0 (Commercial Release)  
> **Primary Risk Estimator:** Calibrated Logistic Regression (D3 1-SE Rule Winner)  
> **Decision Architecture:** Pointwise Minimum Expected Loss Router (`ALLOW`, `VERIFY`, `REQUIRE_DEPOSIT`, `FORCE_PREPAID`)  
> **Regulatory Standard:** Digital Personal Data Protection (DPDP) Act 2023  

---

## 1. Model Overview

FlipPrice AI is a per-order decision-pricing engine designed specifically for Indian cash-on-delivery (COD) e-commerce. Rather than predicting binary fraud or applying blunt pincode blacklists, FlipPrice estimates a calibrated return probability $p(x)$ and computes the expected financial loss across four distinct interventions, selecting the action that maximizes net portfolio profit.

- **Primary Classifier:** Logistic Regression ($L_2$ regularization) selected over gradient boosted trees via the D3 1-Standard-Error simplicity hierarchy on additive data.
- **Calibration Layer:** Cross-validated Platt scaling on log-odds logits (`LogisticRegressionCV(Cs=10, cv=5)`).
- **Inference Latency:** 10.0 ms p50 with SHAP TreeExplainer explanations; 5.1 ms without SHAP.

---

## 2. Intended Use & Operational Boundary

### Target Deployment
- **Customer Profile:** Indian D2C brands processing ₹10 Lakh to ₹1 Crore monthly GMV with 25% to 60% COD share.
- **Platforms:** Shopify storefronts + courier tracking via Shiprocket, Delhivery, or Bluedart.
- **Workflow:** Real-time pre-dispatch routing on the `orders/create` webhook (tagging orders for automated WhatsApp verification, deposit collection, or frictionless fulfillment).

### Out-of-Scope Uses
- Not intended for offline brick-and-mortar retail or purely prepaid transactions.
- Not a replacement for merchant fraud investigation on orders exceeding ₹25,000 (such orders are routed to human review via safety guardrails).

---

## 3. Feature Contract & Provenance

The model evaluates 13 operational features per transaction:

| Feature | Type | Valid Bounds | Provenance & Handling |
| :--- | :--- | :--- | :--- |
| `order_value` | Float | ₹1.0 – ₹25,000.0 | Shopify cart total |
| `quantity` | Integer | 1 – 50 items | Line-item sum |
| `category` | Categorical | 6 categories | Inferred from product taxonomy |
| `discount_pct` | Float | 0.0% – 100.0% | Discount amount relative to subtotal |
| `payment_method` | Categorical | `COD` vs `Prepaid` | Financial status / payment gateway |
| `cod_charge` | Float | ₹0.0 – ₹500.0 | Extra shipping fee charged for COD |
| `account_age_days` | Integer | 0 – 5,000 days | Days since customer first seen |
| `prior_orders` | Integer | 0 – 100 | Historical order count |
| `prior_rto_count` | Integer | 0 – 100 | Historical return count |
| `pincode` | String | 6 digits | Normalized Indian postal code |
| `courier_id` | Categorical | Courier A–E | Allocated delivery partner |
| `orders_last_24h` | Integer | 0 – 50 | Short-term velocity |
| `device_cluster_size` | Integer | 1 – 50 | Syndicate detection proxy |

---

## 4. Cold-Start & Missing Value Policy

1. **Unknown Pincodes:** If a delivery pincode does not appear in the historical rate lookup table, the engine dynamically falls back to the modal tier (Tier 2) and the national mean historical rate (`0.2827`).
2. **New Customers (Zero History):** `prior_orders`, `prior_rto_count`, and `account_age_days` safely default to `0`. There is zero lookahead bias or speculative extrapolation.
3. **Missing Categoricals:** Out-of-vocabulary product types default to modal `Apparel` and generate an audit log warning.

---

## 5. Per-Merchant Recalibration Protocol

To ensure local validity across distinct merchant catalogs:
- **Sample Threshold ($n \ge 300$):** If a merchant uploads $\ge 300$ historical orders with settled delivery/RTO outcomes, a merchant-specific Platt calibration layer is fitted on held-out folds.
- **Below Threshold ($n < 300$):** Model remains pinned to robust pre-trained priors, with explicit written disclosure in the generated Bleed Audit report.

---

## 6. Privacy & DPDP Act 2023 Compliance

- **Zero Plaintext PII Ingestion:** Customer phone numbers and emails are salted and hashed using `SHA-256` before feature engineering. Raw phone numbers are discarded from engine memory immediately after webhook processing.
- **No Household Profiling:** Risk is estimated using postal division aggregates and temporal velocity—never individual street addresses or demographics.
- **Tenant Isolation:** Merchant databases are strictly separated; data from Merchant A is never shared, leaked, or used to fine-tune models for Merchant B.

---

## 7. Known Limitations & Caveats

1. **Synthetic Pre-Training Baseline:** The baseline benchmark of ₹55,058 realized savings is measured on a rigorously verified synthetic cohort ($N=7,174$). Real merchant savings are calculated individually via the free Bleed Audit.
2. **Behavioral Adaptation:** Customers confronted with deposit requests may experience drop-off (modeled conservatively at 40% in baseline config, 35.3% in learned P2 policy).
3. **Festive Drift:** During high-volume festive sale events (e.g. Diwali sales), base RTO rates spike by 5–10%. FlipPrice deploys weekly Population Stability Index (PSI) monitoring to alert merchants to recalibrate.
