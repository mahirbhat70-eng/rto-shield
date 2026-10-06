# Final Hardening Review Packet for Independent Reviewer

> **Auditor Note for Reviewer using an Independent LLM / Model Family:**  
> This packet provides an adversarial, unvarnished index of all modified files, exact reproduction commands, verified results, root cause resolutions for earlier conflicting claims, questions we are least sure about, and surprising discoveries from the Phase A–F hardening pass.

---

## 1. Inventory of Modified and Added Files

| File | Status | Nature of Change / Hardening Added |
|---|---|---|
| `scripts/generate_headline_numbers.py` | Added | Phase A1: Single source of truth generating `reports/headline_numbers.json` from frozen artifacts and test data. |
| `reports/headline_numbers.json` | Generated | Phase A1: Pinned metrics across all models, financial outcomes, action mix, latency, and test count. |
| `tests/test_headline_consistency.py` | Added | Phase A2: Programmatic assertion that numbers across docs match `headline_numbers.json` exactly. |
| `audit/run_b1_breakeven.py` | Added | Phase B1: Bisection crossover solver (COD 16.6%, Platform 11.0%) reconciling earlier 15% conflicts. |
| `src/serve/guardrails.py` | Modified | Phase B2/B4: Added `merchant_break_even` simulation gate; fixed default `min_base_rto_rate` bug. |
| `audit/seed_sweep.py` & `.csv` | Modified | Phase B3: Added `expected_savings`, `realized_savings`, and `gap_pct` across 5 seeds for all 4 models. |
| `audit/run_b5_misspecification.py` | Added | Phase B5: Joint 108-cell misspecification grid saved to `audit/misspecification_grid.csv`. |
| `audit/run_c1_realism.py` | Added | Phase C1/C2: Pipeline re-evaluation with noise sigma 0.10, 0.20, and realistic rolling 30-day prior rate. |
| `tests/test_data_realism.py` | Added | Phase C3/E2: Verifies rolling feature is strictly prior on 50 random rows and lookup is train-only. |
| `audit/run_d1_seeds.py` | Added | Phase D1: Evaluates relative floor ($\ge 90\%$ ceiling, $\ge +0.10$ shuffled) showing seeds 101/999 pass. |
| `audit/run_mutations.py` | Modified | Phase D2: Full 18-mutation sabotage suite (17/18 caught = 94.4% catch rate). |
| `audit/run_d3_model_choice.py` | Added | Phase D3: Statistical model selection rule implementation (Logistic Regression selected by 1-SE rule). |
| `src/serve/scorer.py` | Modified | Phase E1: Fail-closed production mode check (`RTO_SHIELD_PINNED_DIGESTS`), dev warning fallback. |
| `scripts/freeze_artifact_hashes.py` | Modified | Phase E2: Pinned all 13 artifacts across `models/` and `data/` into `models/artifact_hashes.json`. |
| `models/artifact_hashes.json` | Modified | Phase E2: Complete 13-artifact cryptographic manifest. |
| `tests/test_artifact_integrity.py` | Modified | Phase E1/E2: Added completeness check, production fail-closed refusal, and dev fallback tests. |
| `tests/test_audit.py` | Modified | Phase E4: Property-based testing via `hypothesis` ensuring zero PII leak outside allow-list. |
| `audit/run_e3_reproducibility.py` | Added | Phase E3: Deterministic bitwise reproducibility verification across isolated seeds in clean venv. |
| `audit/PILOT_DESIGN.md` | Added | Phase F1: Staged 3-phase pilot plan with statistical sample-size power calculation and stopping rules. |
| `README.md`, `claim-matrix.md` | Modified | Synchronized test suite count (258 passing tests) and headline figures. |

---

## 2. Core Claims & Exact Reproduction Commands

| Claim | Verified Value | Exact Reproduction Command |
|---|---|---|
| **Headline Realized Savings (Frozen)** | ₹69,786.08 (on 7,174 COD test orders) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['realized_savings_inr'])"` |
| **Headline Expected Savings (Frozen)** | ₹71,741.02 (+2.80% expected-realized gap) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['expected_savings_inr'])"` |
| **Test Set Statistical Metrics (N=14,980)** | LR: PR 0.3434, ROC 0.6854, Brier 0.1475<br>Uncal: PR 0.3433, ROC 0.6883, Brier 0.1473<br>Platt: PR 0.3433, ROC 0.6883, Brier 0.1477<br>Iso: PR 0.3313, ROC 0.6874, Brier 0.1476 | `python scripts/generate_headline_numbers.py` |
| **COD Break-Even Crossover** | Exact: **16.6%** COD RTO (Platform: **11.0%**) | `python audit/run_b1_breakeven.py` |
| **5-Seed Expected vs Realized Gaps** | Platt LGBM: **+0.41%** (Std 4.8%)<br>LR: **+5.67%** (Std 4.7%)<br>Isotonic LGBM: **+8.80%** (Std 4.5%) | `python audit/run_d3_model_choice.py` |
| **Served Policy vs Pure Savings** | Served: ₹55,065.83 vs Pure: ₹69,786.08 (-21.09% cap haircut) | `python -c "import pandas as pd; from audit.reconcile import *; tc=pd.read_csv('data/processed/test.csv'); tc=tc[tc['payment_method']=='COD']; m=joblib.load('models/tree_model_calibrated.pkl'); p=m.predict_proba(tc.drop(columns=['rto_label','timestamp','order_id']))[:,1]; g=ProductionGuardrails(intervention_rate_cap=0.60); acts=[g.guard_decision(a,p[i]) for i,a in enumerate(frozen_actions(tc,p))]; print(realized_pl(acts, tc['rto_label'].values, tc['order_value'].values, CFG))"` |
| **Data Realism (Rolling 30D Feature)** | ₹67,768 (97.1% of savings survive) | `python audit/run_c1_realism.py` |
| **Sabotage Mutation Catch Rate** | **17/18 caught (94.4%)** | `python audit/run_mutations.py` |
| **Clean Virtual Environment Tests** | **258 passed, 0 failed in 73.7s** | `audit\clean_env\Scripts\python.exe -m pytest tests/ -q` |
| **Bitwise Clean Reproducibility** | Run 1 == Run 2 (identical seeds) = True<br>Run 1 != Run 3 (diff seed) = True | `audit\clean_env\Scripts\python.exe audit/run_e3_reproducibility.py` |

---

## 3. The Questions We Are LEAST Sure About

1. **Real-World Verification Drop-Off Rate ($\delta_{\text{verify}}$):**
   - In our synthetic environment, good-customer drop-off is assumed to be $5\%$.
   - Sensitivity sweeps show that if real-world drop-off exceeds $10\%$, the net financial benefit of address verification degrades by $60\%$, and if it exceeds $15\%$, verification becomes net-negative for low-margin goods.
   - *Only real merchant Stage 2 A/B data can resolve this.*

2. **Zero Forward Delivery Cost Assumption:**
   - The cost model canonicalized in `configs/cost_config.yaml` prices forward delivery at ₹0 (attributing delivery logistics to baseline operations and only charging the reverse logistics fee of ₹150 + lost margin).
   - If a merchant operates on low margins ($<15\%$) and carrier forward freight is ₹60 per attempt, an RTO incurs ₹60 forward + ₹150 reverse = ₹210. Conversely, preventing an order avoids both. The net savings equation shifts significantly depending on the merchant's carrier contract structure.

3. **Small-Sample Empirical Shrinkage ($M=10$):**
   - In Phase C, empirical rolling rates for low-volume pincodes were shrunk toward the platform prior using $M=10$. While this preserved 97.1% of savings on synthetic data, extreme long-tail pincodes (Tier 4 / rural) with non-stationary courier coverage may experience out-of-distribution drift in production.

---

## 4. Results That Surprised Us

1. **Simpler Logistic Regression Beat LightGBM in 5-Seed Realized Net Savings:**
   - Logistic Regression achieved **₹66,768.25** mean realized savings across 5 seeds, beating Platt LightGBM (₹66,038) and Isotonic LightGBM (₹66,333).
   - Under the written model selection rule (simplest model within 1 SE of best), Logistic Regression qualifies as rank 1!
   - Cause: In noisy tabular domains with low signal-to-noise ratio, linear models with calibrated log-odds do not overfit boundary bins, avoiding spurious high-friction decisions.

2. **Isotonic Calibration Was Biased Overoptimistic (+8.80% Gap):**
   - Isotonic calibration produced an average expected savings of ₹72,338 vs realized ₹66,333 (an 8.80% overestimation gap).
   - By contrast, Platt scaling had a mean gap of only **+0.41%** (₹66,307 expected vs ₹66,038 realized).
   - Cause: Isotonic regression fits step-functions on calibration splits; when evaluated out-of-sample, boundary plateaus misstate the expected loss on orders close to the decision threshold.

3. **Production Rate Limiting (60% Cap) Imposes a -21.09% Haircut on Realized Savings:**
   - In the pure unconstrained policy, 81.63% of COD orders receive interventions (`VERIFY` or `DEPOSIT`), generating ₹69,786.08.
   - When the production 60% rate limiter is applied to safeguard merchant checkout volume, 1,560 orders are forced back to `ALLOW_COD`, reducing realized savings to ₹55,065.83.
   - This proves that operational guardrails come with a direct, quantifiable opportunity cost that merchants must balance against brand friction risk.
