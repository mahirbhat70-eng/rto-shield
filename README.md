# RTO Shield — AI Risk Manager for Merchant COD RTO Risk

RTO Shield converts COD order risk prediction into financially optimal intervention decisions for Indian e-commerce merchants. Every claim in this README traces to a frozen artifact in `reports/`, backed by 40 passing automated tests and 11/11 pre-registered checks on a strictly held-out test set.

## Executive Summary — Final Results (Held-Out Test Set)

The multi-action policy earns **2.0× the savings** of the best single-threshold policy on a fully held-out test window it never influenced, achieving a **13.1% portfolio profit uplift** (₹71,741 on the test COD subset of 7,174 orders).

**Model quality**: the primary model scores 94.7% of the measured Bayes ceiling (0.3313 vs theoretical maximum 0.3497) on test. Because the ceiling proves no model family can do meaningfully better on this problem, the value lives in the decision layer — which is where this project built its edge.

### Policy Evaluation (Test COD Subset, N=7,174)
- **COD-Subset RTO Rate:** 28.27%
- **Mean Calibrated P:** 0.2832
- **Order Value (COD):** Mean ₹826.89 / Median ₹607.57

| Strategy | Total Loss (INR) | Savings vs Baseline | Action Dist (ALLOW/VERIFY/DEPOSIT/PREPAID) | Orders Touched | Friction Spend | RTOs Prevented | Good-Customer Drops |
|----------|------------------|---------------------|--------------------------------------------|----------------|----------------|----------------|---------------------|
| 1. Baseline (Always Allow) | −₹547,766 | ₹0 | 100/0/0/0 | 0 | ₹0 | – | – |
| 2. Binary PREPAID (thr 0.48) | −₹548,696 | ₹930 | 99.3/0/0/0.7 | 49 | ₹0 | 17.0 | 12.6 |
| 3. Binary VERIFY (thr 0.20) | −₹583,686 | ₹35,919 | 17.0/83.0/0/0 | 5,954 | ₹11,908 | 546.5 | 206.6 |
| **4. Primary Multi-Action (Calibrated)** | **−₹619,508** | **₹71,741** | **18.4/45.2/36.4/0.0** | **5,856** | **₹6,488** | **951.2** | **820.1** |
| 5. Sensitivity: Uncalibrated router | −₹630,346 | ₹82,580 | 18.7/46.5/34.8/0.0 | 5,832 | ₹6,672 | – | – |
| 6. Sensitivity: Tail-clipped [0.02, 0.85] | −₹619,648 | ₹71,881 | 18.4/45.2/36.4/0.0 | 5,856 | ₹6,488 | – | – |

*Rows 5–6 are pre-registered sensitivity disclosures, not model changes: the primary model was frozen before the test reveal. Row 5 quantifies what isotonic calibration costs in policy savings (₹10,839 on test); row 6 shows the calibrator's single overconfident tail output is financially immaterial (Δ₹140).*

**Robustness**: savings move +2.7% under N(0, 0.04) probability noise; hold across 25–50% DEPOSIT drop-rate assumptions (₹49.9k / ₹36.3k / ₹31.8k). The router's decision-band structure transferred essentially unchanged between validation and test windows (VERIFY mean P 0.274→0.273; DEPOSIT 0.326→0.328).

**The policy defends unit economics, not volume.** DEPOSIT share by order-value quartile on validation: 91.1% → 43.7% → 12.5% → 0.7% — deposits demanded on cheap risky orders (where ₹150 reverse logistics dwarfs the margin), while high-value orders stay frictionless.

Full numbers: `reports/stage5_test_results.md`, `reports/stage4_financial_results.md`, threshold curve: `reports/stage4/threshold_vs_loss_curve.png`

---

## Why a Cost Engine? (The Core Idea)

Predicting P(RTO) is not the product — deciding what to do about it is. For every COD order, RTO Shield scores four interventions with an expected-loss model and routes to the argmin:

`EL(action) = friction_cost + P_rto_after × ₹150 − P_success × margin`
`margin = order_value × 20%`

---

### Oracle-Grade Feature Warning

`historical_pincode_rto_rate` is a synthetic pre-order proxy drawn from a pincode-level latent prior (Beta(2,8)). It is strictly backward-looking (no temporal leakage) — but it is **oracle-grade**: the generator's own ground-truth risk parameter. A production system would estimate it from finite history (adding noise). This fidelity advantage makes every reported metric an optimistic upper bound, and we quantified the bound: adding σ≈0.04 estimation noise moves LR PR-AUC by only −0.0027 (val) / −0.0012 (test). All metrics should be read as upper bounds with a small, measured optimism.

---

## Stage Records (Historical)

### Stage 1 — Data Contract + Generator (frozen)
19-column schema, edge-case injection (good customer/bad pincode and vice versa), consistency constraints (`prior_rto_count ≤ prior_orders`), 21 validation tests.
Documented deviation: no dedicated pure-noise columns were generated; `quantity` and `device_cluster_size` served as low-signal proxies (SHAP share ~1%).

### Stage 2 — Temporal Split + Baselines

Split by calendar time (strict inequality verified, exact timestamps recorded):

| Split | Rows | Date Range (2026) | RTO Rate |
|-------|------|-------------------|----------|
| Train | 69,883 | Mar 1 → Jul 9 | 19.9% |
| Val   | 15,137 | Jul 9 → Aug 6 | 20.1% |
| Test  | 14,980 | Aug 6 → Sep 3 | 19.8% |

**Rule baseline** (deterministic, thresholds from train only):
`hist_rate > 0.2676 OR prior_rto_count ≥ 2 OR (COD AND account_age < 113)`

**Logistic Regression:** StandardScaler + OneHot; pincode (~1000 values) dropped — its signal is captured by hist_rate + pincode_tier. All 10 top coefficients match documented risk directions (COD +0.616, Tier-1 −0.538, hist_rate +0.268, …).

**The F1 comparison trap (documented so you don't fall for it):** the rule baseline flags ~36% of orders; LR at 0.5 flags ~1%. F1 between them is meaningless. The fair comparison is ranking quality: LR PR-AUC 0.34 vs rule-implied 0.23, and LR precision at the rule's recall (0.33 vs 0.26). LR's low recall (0.027) at the default threshold is a compression artifact — max prob ≈ 0.71, converged cleanly; the operating threshold was deferred to the cost stage by design.

| Model | Split | Flag Rate | Precision | Recall | F1 | PR-AUC | Provenance |
|-------|-------|-----------|-----------|--------|----|--------|------------|
| Rule Baseline | val | 0.3628 | 0.2638 | 0.4751 | 0.3393 | 0.2311 | implied (closed-form) |
| Rule Baseline | test | 0.3557 | 0.2717 | 0.4887 | 0.3493 | 0.2339 | implied (closed-form) |
| Logistic Reg. | val | 0.0100 | 0.5497 | 0.0272 | 0.0519 | 0.3406 | sklearn |
| Logistic Reg. | test | 0.0097 | 0.5793 | 0.0283 | 0.0541 | 0.3434 | sklearn |

---

## SHAP — What Drives Risk (Corrected Family Aggregation)

- **COD-Family** (`cod_charge` + `payment_method_COD`): 34.6%
- **Pincode-Family** (`historical_pincode_rto_rate` + `pincode_tier`): 24.3%
- **Residual** (behavioral): 41.1% — top: `prior_rto_count` 10.0%, `category` 7.4%, `account_age_days` 7.3%
- Lowest-signal proxies: ~1% combined

*(Note: `cod_charge` is the COD indicator in continuous form — family aggregation prevents it from splitting credit with its one-hot duplicate and confusing the importance story.)*

---

## Project Structure

```
rto-shield/
├── README.md
├── requirements.txt
├── configs/
│   ├── data_config.yaml # feature groups, paths (Stage 1)
│   └── cost_config.yaml # interventions, costs, provenance (Stage 4)
├── docs/
│   └── data_dictionary.md # schema, ranges, evidence tiers, oracle warning
├── src/
│   ├── data/
│   │   ├── generator.py # Stage 1 (frozen)
│   │   ├── split.py # Stage 2: temporal split + val_cal/val_rep (Stage 3)
│   │   └── verify_splits.py # boundary + leakage verification
│   ├── eda/
│   │   └── report.py # Stage 2 EDA (4 diagnostic plots)
│   ├── models/
│   │   ├── rule_baseline.py # Stage 2
│   │   ├── logistic_baseline.py # Stage 2
│   │   └── tree_model.py # Stage 3 LightGBM (grid on val_cal only)
│   ├── policy/
│   │   └── cost_engine.py # Stage 4 expected loss + router
│   └── eval/
│       ├── evaluate.py # Stage 2 comparison (closed-form rule AP, matched-recall)
│       ├── calibration.py # Stage 3 reliability + isotonic (val_cal)
│       ├── explainability.py # Stage 3 SHAP summary + waterfalls
│       ├── bayes_ceiling.py # Stage 3.5 theoretical maximum
│       ├── stage3_evaluate.py # continuous results table
│       ├── stage4_evaluate.py # 6-strategy portfolio + noise sensitivity
│       ├── stage5_test_reveal.py # one-shot held-out evaluation
│       ├── verify_calibration.py # bin-MAE artifact investigation
│       └── stress_test_noise.py # oracle-feature σ=0.04 stress test
├── tests/ # 40 tests across 5 stage files
├── reports/
│   ├── stage2_baseline_results.md
│   ├── stage3_results.md # incl. bin-MAE decomposition footnote
│   ├── stage4_financial_results.md
│   ├── stage5_test_results.md # transfer table + pre-registered checks
│   ├── stage4/threshold_vs_loss_curve.png
│   └── eda/ stage3/ # plots
└── models/ # gitignored; reproducible via commands below
```

Model artifacts and datasets are gitignored by design; all final numbers live in `reports/`, and the full chain below reproduces every artifact deterministically.

---

## Reproduction

```bash
pip install -r requirements.txt

# Stage 1 — generate + validate data
python src/data/generator.py
python -m pytest tests/test_generator.py -q

# Stage 2 — split, EDA, baselines
python src/data/split.py && python src/eda/report.py
python src/models/rule_baseline.py && python src/models/logistic_baseline.py
python src/eval/evaluate.py

# Stage 3 — GBM, val_cal/val_rep split, calibration, SHAP, ceiling
python src/models/tree_model.py
python src/eval/calibration.py && python src/eval/explainability.py
python src/eval/bayes_ceiling.py && python src/eval/stage3_evaluate.py

# Stage 4 — cost engine, policy router, sensitivity
python src/eval/stage4_evaluate.py

# Stage 5 — one-shot held-out test reveal (frozen artifacts)
python src/eval/stage5_test_reveal.py

# Full test suite (40 tests)
python -m pytest tests/ -q
```

## Honest Limitations
- **Synthetic data.** Metrics are upper bounds on a known generator; real-world drift, labeling noise, and adversarial adaptation are not simulated (and stated where it matters).
- **Oracle feature (disclosed above):** `hist_rate` is ground truth, not an estimate — optimism measured at Δ PR-AUC ≈ 0.003.
- **One-order cost model.** The expected-loss model prices a single order; customer lifetime value of dropped good customers is not modeled.
- **No live serving layer.** This repository is the measured analytics core; the decision logic is tested and deterministic, but there is no production API.
- **Isotonic tradeoff is real:** the calibrated primary gives up ₹10.8k of test savings vs the uncalibrated router (disclosed, pre-registered).

## License
MIT
