# RTO Shield — AI Risk Manager for Merchant COD RTO Risk

[![CI](https://github.com/mahirbhat70-eng/rto-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/mahirbhat70-eng/rto-shield/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://rto-shield-nlthpydtndpkupyfgfl3yy.streamlit.app)

[![Decision Architecture](docs/decision_architecture.png)](https://htmlpreview.github.io/?https://github.com/mahirbhat70-eng/rto-shield/blob/main/docs/decision_architecture.html)

> 📄 **[Executive Brief (PDF)](docs/executive_brief.pdf)** · **[Interactive HTML](https://htmlpreview.github.io/?https://github.com/mahirbhat70-eng/rto-shield/blob/main/docs/executive_brief.html)** — key numbers, architecture narrative, and calibration story in one page.

---

## ⚡ Quickstart — 5 Minutes

| Step | Command / Link |
|------|----------------|
| **1. Try the live demo** | [rto-shield-nlthpydtndpkupyfgfl3yy.streamlit.app](https://rto-shield-nlthpydtndpkupyfgfl3yy.streamlit.app) |
| **2. Clone & run tests** | `git clone https://github.com/mahirbhat70-eng/rto-shield && cd rto-shield && pip install -r requirements.txt && pytest tests/ -v` |
| **3. Benchmark latency** | `python scripts/benchmark_latency.py` |
| **4. Read the evidence** | [`claim-matrix.md`](claim-matrix.md) — every number → its artifact → its command |
| **5. Read failure story** | [`WHAT_BROKE.md`](WHAT_BROKE.md) — 5 bugs found & fixed across the project lifecycle |
| **6. Hard questions** | [`docs/JUDGE_QA.md`](docs/JUDGE_QA.md) — 10 questions with evidence-backed answers |
| **7. Adversarial audit** | [`audit/GO_NO_GO.md`](audit/GO_NO_GO.md) — 17-point audit, 5 reconciled conflicts & misspecification bounds |

All frozen model artifacts and reports are committed. **No data download needed to run tests (244/244 passed).**

---

### Decision Engine Latency (Full score_order() Path: Model + TreeSHAP + Cost Matrix)

| Percentile | End-to-End Latency (with TreeSHAP) | Raw Model Scoring (without SHAP) |
|-----------|-------------------------------------|----------------------------------|
| p50 | ~15.0 ms | ~3.0 ms |
| p95 | ~18.3 ms | ~8.0 ms |
| p99 | ~21.1 ms | ~15.0 ms |

*Measured across 1,000 iterations on standard CPU. Raw scoring runs in ~3ms; per-order TreeSHAP factor explanation adds ~12ms.*  
Reproduce: `python scripts/benchmark_latency.py --n 1000 --warmup 100`

---

RTO Shield converts COD order risk prediction into financially optimal intervention decisions for Indian e-commerce merchants. Every claim in this README traces to a frozen artifact in `reports/`, backed by **241 passing automated tests** (generator contract, split integrity, preprocessing leakage, metric correctness, dominance regression, adversarial robustness, serving-path equivalence, independent P&L checks, UI logic, decision audit trail) and 11/11 pre-registered checks on a strictly held-out test set.

## Operational Deployment Status: **SHADOW-MODE ONLY**

> ⚠️ **Pre-Production Operational Gate:** Autonomous live intervention routing is strictly disabled pending passive shadow calibration against real merchant delivery exports. Live serving is protected by active production safety circuit breakers:
> 1. **Emergency Kill-Switch:** Environment / flag toggle instantly reverting all traffic to `ALLOW_COD`.
> 2. **Intervention Rate Limiter:** Rolling window cap preventing interventions from exceeding 60% of volume.
> 3. **High-Value Basket Protection:** Orders exceeding ₹10,000 are automatically routed to manual review rather than checkout friction.
> 4. **Merchant Break-Even Gate:** Brands with historical COD RTO rates below **16.8%** automatically bypass friction (`ALLOW_COD`), preventing margin loss on healthy delivery profiles.
> 5. **Cryptographic Integrity & Privacy:** Mandatory SHA-256 digest validation before pickle loading and strict allow-list sanitization stripping all customer PII.

## Executive Summary — Final Results (Held-Out Test Set, Simulated Benchmark)

> **Important Disclosure:** All metrics reported below are derived from an offline, forward-chaining synthetic benchmark dataset (N=100,000) generated under a controlled latent-risk logistic process. The reported RTO volume reductions and cash savings represent **counterfactual mathematical expectations** under specified behavioral response priors (e.g. 40% deposit drop-off, 80% RTO mitigation), not live physical merchant experiments.

The multi-action policy achieves an **assumed 46.9% reduction in COD RTO return volume** (expected interception of **951.2 out of 2,028** failed deliveries based on intervention priors) and cuts the platform COD return rate from **28.3% down to 19.9%** (an **8.4 percentage-point drop**). Financially, it earns **2.0× the savings [1.23×, 2.39× bound]** of the best single-threshold policy, achieving a **13.1% portfolio profit uplift** (₹71,741 expected savings / ₹69,786 counterfactual realized cash savings on the test COD subset of 7,174 orders).

**Model quality**: the primary model scores 94.7% of the measured Bayes ceiling (0.3313 vs theoretical maximum 0.3497) on test. Note: Thresholds (such as ~0.20 for verification and ~0.48 for deposits) are **order-value dependent optimal boundaries** ($p^*(V)$), not rigid constants — on a ₹100 order, verification triggers at 6.5% risk, while on a ₹10,000 order it triggers at 70%.

### 📊 Comprehensive Model Evaluation Scorecard (Simulated Benchmark)

| Evaluation Dimension | Metric / Test | Value | Production & Business Meaning |
|---|---|---|---|
| **1. Business Impact** | **RTO Volume Reduction (Simulated)** | **46.9%** | Counterfactual model expectation (951.2 out of 2,028) based on intervention priors. |
| | **COD RTO Rate Drop (Simulated)** | **28.3% → 19.9%** | Simulated 8.4 percentage-point drop in platform return rate. |
| | **Portfolio Profit Uplift (Simulated)**| **+13.1%** | ₹71,741 net simulated savings on 7,174 COD orders after friction costs. |
| | **Realized P&L Savings (Simulated)** | **₹69,786** | Evaluated on actual ground-truth labels against response models (within 2.7% of EL forecast). |
| | **Policy Superiority (Simulated)** | **2.0×** | Multi-action routing earns 2× the savings of single-threshold verify. |
| **2. Statistical Performance** | **PR-AUC (Primary)** | **0.3313** | Evaluated on strictly held-out chronological test set (Logistic Regression baseline: 0.3434). |
| | **Bayes Optimal Ceiling** | **0.3497** | Model achieves **94.7% of theoretical maximum signal** for this data generator. |
| | **Recall @ Operating Point** | **85.6%** | Intercepts 1,735 out of 2,028 true RTO attempts. |
| | **Precision @ Operating Point** | **29.6%** | Balances customer friction against ₹150 logistics penalty. |
| | **Calibration (Brier / Bin-MAE)**| **0.1476 / 0.0086** | Probability accuracy required for cost engine. |
| **3. Data Rigor & Splits** | **Split Strategy** | **Forward-Chaining** | Strict chronological split (Train: 71k, Val: 15k, Test: 15k) with 0 temporal leakage. |
| | **Target Segment** | **7,174 COD Orders** | Evaluated specifically on Cash-on-Delivery checkout flow. |
| **4. Robustness & Stress** | **Noise Sensitivity** | **+2.7% Delta** | Evaluated with σ=0.04 Gaussian feature noise (stable ≤ 10%). |
| | **Cold-Start Pincode Fallback**| **12.0% share** | Automatic empirical Bayesian prior fallback on new locations. |
| | **Monte Carlo (5,000 runs)** | **P(ROI > 0) = 100%** | Mean ₹69,942 (5th %tile: ₹63,935, 95th %tile: ₹75,901) over synthetic test set resampling. |
| **5. Computational Efficiency**| **Latency (p50 / p95 / p99)** | **15ms / 18ms / 21ms** | Full end-to-end serving latency including TreeSHAP factors (3ms without SHAP). |
| | **Model Footprint** | **123 KB** | Ultra-lightweight LightGBM, embeddable in Edge/Lambda functions. |


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

**Realized Performance (Test COD):** Operating-point P/R is 0.2963 / 0.8555, and
per-action calibration is tight (Δ ≤ 0.0032 across all three decision bands —
DEPOSIT 0.3279 vs 0.3247, VERIFY 0.2733 vs 0.2734, ALLOW 0.2191 vs 0.2223).
Realized actual-label savings reached **₹69,786** — within **−2.7%** of the EL
forecast and inside Monte Carlo noise (5,000 draws: P5 ₹63,935, P95 ₹75,901,
P(savings > 0) = 100%). The uncalibrated router tells the other story: its EL
forecast of ₹82,580 collapsed to ₹69,902 against labels (**−15.3%**). In short:
the ₹10.8k "cost" of calibration bought the only forecast that held up.

Full numbers: [reports/stage5_test_results.md](reports/stage5_test_results.md), [reports/stage4_financial_results.md](reports/stage4_financial_results.md), threshold curve: [reports/stage4/threshold_vs_loss_curve.png](reports/stage4/threshold_vs_loss_curve.png)

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

## 🛡️ Adversarial Audit, Robustness & Safety Guardrails

In response to an exhaustive 17-point adversarial audit, the codebase underwent rigorous hardening, reconciliation, and automated safety verification:

### 1. Reconciled Audit Conflicts (Zero Discrepancies)
- **Design Point Savings (₹69,786.08 canonical):** An independent verification script had hardcoded a ₹0.50 per-order deposit friction ($2,612 \times 0.50 = ₹1,306.00$), yielding ₹68,480.08 ($₹69,786.08 - ₹1,306.00$). The config-driven canonical number is **₹69,786.08**.
- **Bayes Optimal Ceiling (0.3497 observable):** The theoretical upper bound derived from the true observable generator expectation $\mathbb{E}[p \mid x]$ is **0.3497**. The primary model (0.3313) achieves **94.75% of observable ceiling signal**. The ~0.46 PR-AUC ceiling reported in earlier sweeps was computed against the unobservable latent risk $p_{\text{latent}}$ containing irreducible Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.80)$.
- **Model Ranking & Isotonic Calibration:** On the full test set ($N=14,980$), Logistic Regression achieves PR-AUC 0.3434, uncalibrated LightGBM achieves 0.3433, and isotonic LightGBM achieves 0.3313. Isotonic regression collapses the model's continuous risk distribution into 97 discrete probability plateaus (a step-function artifact that dampens ranking granularity), while preserving perfect calibration within decision bands ($|\Delta| \le 0.0032$).
- **Merchant Break-Even Gate (16.8% COD RTO):** At average ₹1,000 order value and 20% margin, losing a customer costs ₹200 while intercepting an RTO saves ₹150. If a merchant's baseline COD RTO is below **16.8%**, intervention friction destroys net margin. The system automatically enforces an `ALLOW_COD` bypass for low-risk brands.
- **Cryptographic Artifact Inventory:** All 9 production artifacts (5 CSVs, 4 PKLs) are pinned in `models/artifact_hashes.json`. Deserialization refuses unverified pickles with a hard `SecurityError`.

### 2. Behavioral Misspecification Sensitivity Bounds
Executed via `audit/misspec.py` across full parameter perturbations:
- **Deposit Drop-Off Rate:** Base 40%; profitable up to **99.58%** drop-off.
- **Verify Drop-Off Rate:** Base 5%; profitable up to **23.81%** drop-off.
- **Reverse Courier Logistics Cost:** Base ₹150; remains net-profitable down to **₹76.11**.
- **Joint Pessimistic Scenario:** Under worst-case joint conditions (Deposit Drop 60%, RTO drop 60%, Courier ₹120, Margin 25%), routing yields a net loss of −₹5,870.94, confirming why live deployment requires shadow calibration.

### 3. Clean-Room CI Reproducibility
- Automated GitHub Actions workflow: [`.github/workflows/repro.yml`](.github/workflows/repro.yml) rebuilds all 9 model artifacts from raw data in a clean environment and asserts SHA-256 match.
- Fully documented in [`audit/prompt_12_repro.md`](audit/prompt_12_repro.md).

---

## Project Structure

```
rto-shield/
├── app.py                      # Streamlit UI (Single-Order Scorer)
├── dashboard.py                # Streamlit UI (5-View Decision & Governance Console)
├── run_app.bat                 # Windows launcher
├── README.md
├── requirements.txt
├── requirements-ci.txt         # Pinned CI dependencies
├── .github/workflows/
│   ├── ci.yml                  # Continuous integration test runner
│   └── repro.yml               # Clean-room artifact rebuild & SHA-256 verification
├── audit/                      # Adversarial audit evidence & verification scripts
│   ├── GO_NO_GO.md             # Production gate verdict (SHADOW-MODE ONLY)
│   ├── prompt_12_repro.md      # Clean-venv artifact rebuild & seed determinism
│   ├── reconcile.py            # Mathematical reconciliation of the 5 audit conflicts
│   ├── misspec.py              # Behavioral misspecification sensitivity analysis
│   ├── seed_sweep.py           # 5-seed end-to-end pipeline evaluation
│   └── run_mutations.py        # In-tree mutation testing harness
├── configs/
│   ├── data_config.yaml        # feature groups, paths (Stage 1)
│   └── cost_config.yaml        # interventions, costs, provenance (Stage 4)
├── docs/
│   ├── data_dictionary.md      # schema, ranges, evidence tiers, oracle warning
│   ├── JUDGE_QA.md             # 10 hard audit questions with evidence
│   └── pitch_script.md         # 5-minute competition pitch script
├── src/
│   ├── data/
│   │   ├── generator.py        # Stage 1 (frozen)
│   │   ├── split.py            # Stage 2: temporal split + val_cal/val_rep (Stage 3)
│   │   └── verify_splits.py    # boundary + leakage verification
│   ├── eda/
│   │   └── report.py           # Stage 2 EDA (4 diagnostic plots)
│   ├── models/
│   │   ├── rule_baseline.py    # Stage 2
│   │   ├── logistic_baseline.py# Stage 2
│   │   └── tree_model.py       # Stage 3 LightGBM (grid on val_cal only)
│   ├── policy/
│   │   └── cost_engine.py      # Stage 4 expected loss + router
│   ├── serve/
│   │   ├── audit.py            # PII allow-list sanitization & HMAC audit trail
│   │   ├── guardrails.py       # Production safety circuit breakers
│   │   ├── lookup.py           # pincode statistics rebuild
│   │   └── scorer.py           # serving entrypoint, SHA-256 validation, SHAP
│   └── eval/
│       ├── evaluate.py         # Stage 2 comparison (closed-form rule AP, matched-recall)
│       ├── calibration.py      # Stage 3 reliability + isotonic (val_cal)
│       ├── explainability.py   # Stage 3 SHAP summary + waterfalls
│       ├── bayes_ceiling.py    # Stage 3.5 theoretical maximum
│       ├── stage3_evaluate.py  # continuous results table
│       ├── stage4_evaluate.py  # 6-strategy portfolio + noise sensitivity
│       ├── stage5_test_reveal.py # one-shot held-out evaluation
│       ├── verify_calibration.py # bin-MAE artifact investigation
│       └── stress_test_noise.py# oracle-feature σ=0.04 stress test
├── tests/                      # 241 automated tests across 18 test files
├── scripts/
│   ├── benchmark_latency.py    # Latency benchmarking
│   ├── deposit_effectiveness_sensitivity.py # Stage 6 bounds
│   ├── freeze_artifact_hashes.py # SHA-256 provenance
│   ├── generator_v2.py         # Stage 6 uplift data
│   ├── stage6_uplift_ope.py    # Stage 6 Evaluation
│   └── verify_shap_sign.py     # SHAP class 1 verification
├── reports/
│   ├── stage2_baseline_results.md
│   ├── stage3_results.md       # incl. bin-MAE decomposition footnote
│   ├── stage4_financial_results.md
│   ├── stage5_test_results.md  # transfer table + pre-registered checks + Stage 5.2 Realized P&L
│   ├── stage4/threshold_vs_loss_curve.png
│   └── eda/ stage3/            # plots
└── models/                     # committed for one-command demo (with artifact_hashes.json)
```

Model artifacts and the synthetic dataset are committed (synthetic, no privacy concern — and committing them fixes the Stage 5 SHA-256 value for any clone, strengthening the hash assertion). Every number remains reproducible via the chain below.

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

# Stage 6 — Uplift OPE and financial bounds
python scripts/generator_v2.py --seed 42
python scripts/stage6_uplift_ope.py
python scripts/deposit_effectiveness_sensitivity.py
python scripts/freeze_artifact_hashes.py

# Adversarial audit verification
python audit/reconcile.py
python audit/misspec.py
python audit/seed_sweep.py

# Full test suite (241 tests)
python -m pytest tests/ -q

# Serving layer + demo (artifacts are committed — no generation needed)
python src/serve/lookup.py        # rebuild pincode lookup from TRAIN only
python -m pytest tests/ -q        # expect: 241 passed
streamlit run dashboard.py        # 5-view decision & governance console
```

## Honest Limitations
- **Synthetic data.** Metrics are upper bounds on a known generator; real-world drift, labeling noise, and adversarial adaptation are not simulated (and stated where it matters).
- **Oracle feature (disclosed above):** `hist_rate` is ground truth, not an estimate — optimism measured at Δ PR-AUC ≈ 0.003.
- **One-order cost model.** The expected-loss model prices a single order; customer lifetime value of dropped good customers is not modeled.
- **No production REST API.** The scoring module (`src/serve/`, ~15–30ms p99 per order depending on hardware) is production-ready but not exposed as a hosted REST endpoint — the analytics core is the deliverable. A Streamlit demo **is** live at [rto-shield-nlthpydtndpkupyfgfl3yy.streamlit.app](https://rto-shield-nlthpydtndpkupyfgfl3yy.streamlit.app) for interactive evaluation.
- **Isotonic tradeoff is real:** the calibrated primary gives up ₹10.8k of test savings vs the uncalibrated router (disclosed, pre-registered).
- **Stage-6 Causality Caveat:** The reported bounds [1.23×, 2.39×] rely on semi-synthetic uplift data; unobserved confounders in real-world friction effectiveness could widen these bounds.

## License
MIT
