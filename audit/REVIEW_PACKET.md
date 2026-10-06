# Final Hardening Review Packet for Independent Reviewer

> **Auditor Note for Reviewer using an Independent LLM / Model Family:**  
> This packet provides an adversarial, unvarnished index of all modified files, exact reproduction commands, verified results, root cause resolutions for earlier conflicting claims, questions we are least sure about, and surprising discoveries from the Phase A–F hardening pass.

---

## 1. Inventory of Modified and Added Files

| File | Status | Nature of Change / Hardening Added |
|---|---|---|
| `scripts/generate_headline_numbers.py` | Added / Hardened | Single source of truth generating `reports/headline_numbers.json` from frozen artifacts, test data, and 5-seed sweep stats. |
| `reports/headline_numbers.json` | Generated | Pinned metrics across all models, financial outcomes, action mix, latency, and test count (263). |
| `tests/test_headline_consistency.py` | Added | Programmatic assertion that numbers across docs match `headline_numbers.json` exactly. |
| `audit/run_b1_breakeven.py` | Added | Bisection crossover solver (COD 16.60%, Platform 14.19%) reconciling earlier 15% conflicts. |
| `src/serve/guardrails.py` | Modified | Added `merchant_break_even` simulation gate with 95% lower bound check; fixed `min_base_rto_rate` bug; implemented `cap_mode='ranked'`. |
| `audit/seed_sweep.py` & `.csv` | Modified | Computed `expected_savings`, `realized_savings`, and `gap_pct` across 5 seeds for all 4 models. |
| `audit/correction_econ.py` | Added | Joint 81-cell misspecification grid, bootstrap crossover CI, and ranked cap analysis. |
| `audit/run_c1_realism.py` | Modified | Pipeline re-evaluation under rolling 30D empirical rate regime (94.01% survival) and Zipf distribution. |
| `tests/test_data_realism.py` | Added | Verifies rolling feature is strictly prior on 50 random rows and lookup is train-only. |
| `audit/run_mutations.py` | Modified | Expanded 20-mutation sabotage suite (20/20 caught = 100.0% catch rate). |
| `audit/run_d3_model_choice.py` | Added | Statistical model selection rule evaluation (Logistic Regression selected by 1-SE rule). |
| `src/serve/scorer.py` | Modified | Fail-closed production mode check (`RTO_SHIELD_PINNED_DIGESTS`), fails closed when env is unset or unrecognised. |
| `scripts/freeze_artifact_hashes.py` | Modified | Pinned all 13 artifacts across `models/` and `data/` into `models/artifact_hashes.json`. |
| `models/artifact_hashes.json` | Modified | Complete 13-artifact cryptographic manifest. |
| `tests/test_artifact_integrity.py` | Modified | Added completeness check, production fail-closed refusal, and dev fallback tests. |
| `tests/test_audit.py` | Modified | Property-based testing via `hypothesis` ensuring zero PII leak outside allow-list. |
| `audit/run_e3_reproducibility.py` | Added | Deterministic bitwise reproducibility verification across isolated seeds in clean venv. |
| `audit/PILOT_DESIGN.md` | Added | Staged 3-phase pilot plan with statistical sample-size power calculation and stopping rules. |
| `configs/cost_config.yaml` | Modified | Added assumed operational friction costs (deposit ₹7.00, verify ₹2.00). |
| `tests/test_cost_engine.py` | Modified | Added assertions guarding non-zero deposit operational costs. |
| `tests/test_guardrails.py` | Modified | Added assertions guarding ranked rate-cap behavior. |
| `tests/test_stage2.py` & `stage3.py`| Modified | Guarded monotonic temporal splitting, tree max_depth >= 2, and calibrator fit leakage. |
| `README.md`, `claim-matrix.md` | Modified | Scrubbed all overclaims; synchronized test suite count (263 passing tests) and headline figures. |

---

## 2. Core Claims & Exact Reproduction Commands

| Claim | Verified Value | Exact Reproduction Command |
|---|---|---|
| **Headline Realized Savings (Frozen)** | ₹69,786.08 (on 7,174 COD test orders, ₹9.73/order) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['realized_savings_inr'])"` |
| **Headline Expected Savings (Frozen)** | ₹71,741.02 (+2.80% expected-realized gap) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['expected_savings_inr'])"` |
| **5-Seed Shipped Model Stats** | Realized: ₹66,333.14 ± 2,147.73; Expected: ₹72,135.02 ± 2,855.90; Gap: 8.80% ± 4.46% | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['five_seed_sweep_shipped_model'])"` |
| **Test Set Statistical Metrics (N=14,980)** | LR: PR 0.3434, ROC 0.6854, Brier 0.1475<br>Uncal: PR 0.3433, ROC 0.6883, Brier 0.1473<br>Platt: PR 0.3433, ROC 0.6883, Brier 0.1477<br>Iso: PR 0.3313, ROC 0.6874, Brier 0.1476 | `python scripts/generate_headline_numbers.py` |
| **COD Break-Even Crossover** | Mean: **16.60%** COD RTO (Bootstrap 95% CI: [16.47%, 16.73%]; Platform: **14.19%**) | `python audit/run_b1_breakeven.py` |
| **Served Policy vs Pure Savings** | Pure: ₹69,786.08<br>Naive 60% Cap: ₹55,065.83 (-21.09%)<br>Ranked 60% Cap: ₹66,681.50 (-4.45%) | `python audit/correction_econ.py` |
| **Data Realism (Rolling 30D Feature)** | ₹67,268.35 (94.01% of savings survive; loss of ₹597.03 per 1k COD) | `python audit/run_c1_realism.py` |
| **Sabotage Mutation Catch Rate** | **20/20 caught (100.0%)** | `python audit/run_mutations.py` |
| **Clean Virtual Environment Tests** | **263 passed, 0 failed in 79.85s** | `audit\clean_env\Scripts\python.exe -m pytest tests/ -q` |
| **Bitwise Clean Reproducibility** | Run 1 == Run 2 (identical seeds) = True<br>Run 1 != Run 3 (diff seed) = True | `audit\clean_env\Scripts\python.exe audit/run_e3_reproducibility.py` |

---

## 3. The Questions We Are LEAST Sure About

1. **Real-World Verification & Deposit Drop-Off Rates ($\delta_{\text{verify}}$, $\delta_{\text{deposit}}$):**
   - In our synthetic environment, good-customer drop-off is assumed to be $5\%$ for verification and $15\%$ for deposit.
   - Sensitivity sweeps show that if real-world deposit drop-off exceeds $50\%$, or verification drop-off exceeds $10\%$, a frozen uncalibrated policy becomes net-negative.
   - *Only real merchant Stage 2 A/B data can resolve this.*

2. **Zero Forward Delivery Cost Assumption:**
   - The baseline cost model canonicalized in `configs/cost_config.yaml` prices forward delivery at ₹0 (attributing delivery logistics to baseline operations and only charging the reverse logistics fee of ₹150 + lost margin).
   - If a merchant operates on low margins ($<15\%$) and carrier forward freight is ₹60 per attempt, an RTO incurs ₹60 forward + ₹150 reverse = ₹210. Conversely, preventing an order avoids both. The net savings equation shifts significantly depending on the merchant's carrier contract structure.

3. **Small-Sample Empirical Shrinkage ($M=10$) on Zipf Long-Tail:**
   - In Task 1 Zipf volume testing, 66.1% of pincodes have fewer than 5 orders per month.
   - While shrinkage toward global prior preserves model lift (+24.3% over a pincode rule), extreme long-tail pincodes (Tier 4 / rural) with non-stationary courier coverage may experience out-of-distribution drift in production.

---

## 4. Results That Surprised Us

1. **Simpler Logistic Regression Beat LightGBM in 5-Seed Realized Net Savings:**
   - Logistic Regression achieved **₹66,768.25** mean realized savings across 5 seeds, beating Platt LightGBM (₹66,038) and Isotonic LightGBM (₹66,333).
   - Under the written model selection rule (simplest model within 1 SE of best), Logistic Regression qualifies as rank 1!
   - Cause: In noisy tabular domains with low signal-to-noise ratio, linear models with calibrated log-odds do not overfit boundary bins, avoiding spurious high-friction decisions.

2. **Isotonic Calibration Was Biased Overoptimistic (+8.80% Gap):**
   - Isotonic calibration produced an average expected savings of ₹72,135 vs realized ₹66,333 (an 8.80% overestimation gap).
   - By contrast, Platt scaling had a mean gap of only **+0.41%** (₹66,307 expected vs ₹66,038 realized).
   - Cause: Isotonic regression fits step-functions on calibration splits; when evaluated out-of-sample, boundary plateaus misstate the expected loss on orders close to the decision threshold.

3. **Ranked Cap Recovers 79% of the Capacity Haircut:**
   - In the pure unconstrained policy, 81.63% of COD orders receive interventions (`VERIFY` or `DEPOSIT`), generating ₹69,786.08.
   - When a naive arrival-order 60% rate limiter is applied, realized savings drops by -21.09% (to ₹55,065.83).
   - However, when using a benefit-ranked cap (`cap_mode='ranked'`), realized savings is ₹66,681.50 (-4.45% haircut), recovering 79% of the lost capacity haircut.
