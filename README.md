# RTO Shield -- AI Risk Manager for Merchant COD RTO Risk

RTO Shield is an enterprise AI risk management platform designed to predict, quantify, and minimize Cash-on-Delivery (COD) Return-to-Origin (RTO) financial losses for e-commerce merchants.

---

## Executive Summary: Final Stage 5 (Test Set) Results
The multi-action policy earns **2.0× the savings** of the best single-threshold policy on a fully held-out test window, achieving a **13.1% portfolio profit uplift** (₹71,741 on the test COD subset).

### Policy Evaluation (Test COD Subset, N=7,174)
- **COD-Subset RTO Rate:** 28.27%
- **Mean Calibrated P:** 0.2832
- **Order Value (COD):** Mean ₹826.89 / Median ₹607.57

| Strategy | Savings vs Baseline | Action Dist (ALLOW / VERIFY / DEPOSIT / PREPAID) | Orders Touched | Friction Spend | Expected RTOs Prevented | Expected Good-Customer Drops |
|----------|---------------------|--------------------------------------------------|----------------|----------------|-------------------------|------------------------------|
| 1. Baseline | ₹0 | 100.0% / 0.0% / 0.0% / 0.0% | 0 | ₹0 | - | - |
| 2. Binary PREPAID | ₹930 | 99.3% / 0.0% / 0.0% / 0.7% | 49 | ₹0 | 17.02 | 12.63 |
| 3. Binary VERIFY | ₹35,919 | 17.0% / 83.0% / 0.0% / 0.0% | 5,954 | ₹11,908 | 546.48 | 206.62 |
| **4. Primary (Cal)** | **₹71,741** | **18.4% / 45.2% / 36.4% / 0.0%** | **5,856** | **₹6,488** | **951.16** | **820.08** |

*All results verified on a strict temporal hold-out test set with 11/11 pre-registered checks passing, confirming robustness against probability noise (+2.7% delta) and drop-rate sensitivity.*

For full details, see the [Stage 5 Test Results](reports/stage5_test_results.md).

---

## Current Status: Stage 2 -- EDA + Temporal Split + Baseline Models

Stage 1 (data contract + synthetic generator) is complete and frozen.
Stage 2 adds temporal splitting, exploratory analysis, and two baseline models.

**This is Stage 2 only** -- no XGBoost/LightGBM, SHAP, calibration, cost engine, API, dashboard, or agents yet.

---

## Synthetic Dataset Notes

- The dataset (`data/raw/synthetic_orders.csv`) is fully synthetic, for development/evaluation only -- not real merchant data.
- Labels are generated via a stochastic latent-risk logistic-Bernoulli process, not deterministic rules.
- `rto_label` is a post-delivery outcome, not available at order creation.
- **Oracle-Grade Feature Warning**: `historical_pincode_rto_rate` is a synthetic pre-order proxy from a pincode-level latent prior (`Beta(2,8)`). Because it is generated from this fixed latent prior and *not* by aggregating future rows in the dataset, it is strictly backward-looking in terms of temporal leakage. However, it is **oracle-grade** because it is the ground-truth risk parameter. A real production system must estimate this from finite history (adding noise). This fidelity advantage makes every evaluation metric an optimistic upper bound.
- `prior_orders`, `prior_rto_count`, `orders_last_24h` are latent per-row attributes (Stage 1 simplification).

---

## Stage 2: Temporal Split

Split methodology: **by timestamp (calendar time), not by row count or random shuffle**.

| Split | Rows | Date Range | RTO Rate |
|-------|------|------------|----------|
| Train | 69,883 | 2026-03-01 00:00:48 -> 2026-07-09 21:33:15 | 19.9% |
| Val | 15,137 | 2026-07-09 21:35:03 -> 2026-08-06 22:46:27 | 20.1% |
| Test | 14,980 | 2026-08-06 22:48:28 -> 2026-09-03 23:58:17 | 19.8% |

- Time range split: first 70% / next 15% / final 15% of the total time range.
- No temporal leakage: Strict inequality holds between splits (e.g. `max(train) = 21:33:15 < min(val) = 21:35:03`).
- **Note on timestamps and drift**: Timestamps are future-dated to 2026 by design (synthetic). RTO rates are flat across splits (~20%), which means the split methodology is drift-ready, but the generator is drift-neutral (no temporal drift was injected).

---

## Stage 2: Rule Baseline

A simple deterministic if/else policy (not a learned model):

```
Predict RTO=1 if:
  historical_pincode_rto_rate > 0.2676
  OR prior_rto_count >= 2
  OR (payment_method == "COD" AND account_age_days < 113)
```

**Thresholds derived from TRAIN set only** (top quartile / percentiles). Not tuned on val or test.

---

## Stage 2: Logistic Regression Baseline

- **Preprocessing**: StandardScaler on 10 numerical features; OneHotEncoder on 4 low-cardinality categoricals (category, payment_method, courier_id, pincode_tier).
- **Pincode encoding decision**: `pincode` (~1000 distinct values) is DROPPED from the feature set. Its predictive signal is already captured by `historical_pincode_rto_rate` (continuous) and `pincode_tier` (categorical). Including 1000 one-hot columns would add noise and increase overfitting risk.
- **Regularization**: Default C=1.0 (not tuned this stage).
- **Fitted on train only**; scaler/encoder statistics come from train only.
- Model saved to `models/logistic_baseline.pkl`.

---

## Stage 2: Results (Default 0.5 Threshold)

> **Note on Logistic Regression's low recall (0.027):**
> At 20% prevalence, a calibrated model at 0.5 should flag ~20% of orders, but LR flags ~1%. The model converged cleanly (`n_iter_=19`) and max probabilities reach ~0.71, but the distribution is highly compressed (99% of scores are < 0.5). It remains conservative at the default threshold; the operating threshold will be deferred to the cost-matrix stage. This sets up why a non-linear model (like GBM) is expected to extract more signal in Stage 3.
> 
> **Rule Baseline vs Logistic Regression:**
> The Rule Baseline flags ~36% of all orders to catch ~49% of RTOs. This is operationally absurd as a real policy (you would gate a third of your volume). LR at 0.5 flags ~1%. F1 score between a 36%-flag-rate policy and a 1%-flag-rate policy is meaningless. The fair comparison is ranking quality. At the ranking level, LR (PR-AUC ~0.34) comfortably beats the rule baseline floor (PR-AUC ~0.23). 

| Model | Split | Flag Rate | Precision | Recall | F1 | PR-AUC | PR-AUC Provenance | ROC-AUC | Prec @ Rule Recall |
|-------|-------|-----------|-----------|--------|----|--------|-------------------|---------|-------------------|
| Rule Baseline | val | 0.3628 | 0.2638 | 0.4751 | 0.3393 | 0.2311 | implied (closed-form, binary scorer) | N/A | N/A |
| Rule Baseline | test | 0.3557 | 0.2717 | 0.4887 | 0.3493 | 0.2339 | implied (closed-form, binary scorer) | N/A | N/A |
| Logistic Regression | val | 0.0100 | 0.5497 | 0.0272 | 0.0519 | 0.3406 | sklearn | 0.6756 | 0.3299 |
| Logistic Regression | test | 0.0097 | 0.5793 | 0.0283 | 0.0541 | 0.3434 | sklearn | 0.6854 | 0.3408 |

### Production-Realism Stress Test (Noise Injection)
Because `historical_pincode_rto_rate` is an oracle-grade latent prior, we stress-tested the Logistic Regression pipeline by adding empirical estimation noise ($\sigma \approx 0.04$) to this feature on the validation and test sets (simulating a ~100-order moving average estimate):
- **Val Set Clean PR-AUC**: 0.3406 $\rightarrow$ **Noisy PR-AUC**: 0.3379 ($\Delta = -0.0027$)
- **Test Set Clean PR-AUC**: 0.3434 $\rightarrow$ **Noisy PR-AUC**: 0.3422 ($\Delta = -0.0012$)

**Conclusion:** The metrics barely move. While the baseline metrics represent an optimistic upper bound, this stress test significantly shrinks the caveat.

---

## Stage 3.5: Bayes Ceiling & Diagnostic Audit

### Bayes Ceiling Proxy
We quantified the theoretical maximum PR-AUC possible given the dataset's stochastic noise by reconstructing the deterministic latent-risk probability (p) implied by the generator.

- **Bayes Ceiling PR-AUC:** **0.3462**
- **Binning Validation:** Max deviation of 0.0252 sits within the expected ~2-sigma statistical tolerance for max-of-10 bins at n=750, with no systematic sign pattern.

| Model | PR-AUC | % of Bayes Ceiling (0.3462) |
|-------|--------|-----------------------------|
| Uncalibrated Tree (LGBM) | 0.3409 | 98.5% |
| Logistic Regression | 0.3395 | 98.1% |
| Calibrated Tree (LGBM) | 0.3276 | 94.6% |

### Feature Family SHAP Aggregation (Corrected)
- **All-Other-Family (Residual):** 41.11% (Top 3: prior_rto_count, category, account_age_days)
- **COD-Family:** 34.63%
- **Pincode-Family:** 24.25%
- **Noise-Columns:** 0.00% (No dedicated noise columns were generated; quantity and device_cluster_size served as low-signal proxies.)

---

## Stage 3 Expectations (Tripwires)

As we transition to Stage 3 (GBM implementation), the expectations are set:
- **Expected GBM PR-AUC:** 0.45 - 0.60.
- **Below 0.40:** The synthetic signal is too diffuse vs the noise columns. We must check noise-column importance and lift magnitudes before declaring a successful floor beat.
- **Above 0.65:** Suspect oracle-feature dominance or near-deterministic lifts. We must check `historical_pincode_rto_rate`'s feature importance share.
- **Calibration:** LR's compressed probabilities predict poor mid-range calibration. GBM should sit closer to the diagonal.

---

## Top Logistic Regression Coefficients

| Feature | Coefficient | Expected Sign | Match? |
|---------|------------|---------------|--------|
| payment_method_COD | +0.6162 | + (COD increases risk) | Yes |
| pincode_tier_1 | -0.5383 | - (Tier 1 = lower risk) | Yes |
| payment_method_Debit Card | -0.4679 | - (prepaid = lower risk) | Yes |
| payment_method_UPI | -0.3865 | - (prepaid = lower risk) | Yes |
| payment_method_Credit Card | -0.3665 | - (prepaid = lower risk) | Yes |
| payment_method_Net Banking | -0.3591 | - (prepaid = lower risk) | Yes |
| category_Home | -0.3449 | - (lower risk category) | Yes |
| pincode_tier_2 | -0.2933 | - (Tier 2 < Tier 3 risk) | Yes |
| historical_pincode_rto_rate | +0.2677 | + (high hist rate = risk) | Yes |
| courier_id_Courier_C | -0.2635 | - (courier variation) | Yes |

All coefficient signs match the expected risk directions from Stage 1 data dictionary.

---

## Project Structure

```text
rto-shield/
|-- README.md
|-- requirements.txt
|-- configs/
|   +-- data_config.yaml
|-- docs/
|   +-- data_dictionary.md
|-- src/
|   |-- data/
|   |   |-- generator.py          # Stage 1 (frozen)
|   |   +-- split.py              # Stage 2: temporal split
|   |-- eda/
|   |   +-- report.py             # Stage 2: EDA script
|   |-- models/
|   |   |-- rule_baseline.py      # Stage 2: rule baseline
|   |   +-- logistic_baseline.py  # Stage 2: logistic regression
|   +-- eval/
|       +-- evaluate.py           # Stage 2: evaluation
|-- tests/
|   |-- conftest.py
|   |-- test_generator.py         # Stage 1 tests
|   +-- test_stage2.py            # Stage 2 tests
|-- models/
|   +-- logistic_baseline.pkl     # Trained LR pipeline
|-- data/
|   |-- raw/
|   |   +-- synthetic_orders.csv
|   +-- processed/
|       |-- train.csv
|       |-- val.csv
|       +-- test.csv
+-- reports/
    |-- stage2_baseline_results.md
    +-- eda/
        |-- 01_distributions.png
        |-- 02_rto_by_categorical.png
        |-- 03_rto_vs_hist_pincode_rate.png
        +-- 04_correlation_matrix.png
```

---

## Verification Commands

```bash
# Stage 1
python src/data/generator.py
python -m pytest tests/test_generator.py -q

# Stage 2
python src/data/split.py
python src/eda/report.py
python src/eval/evaluate.py
python -m pytest tests/test_stage2.py -q
```
