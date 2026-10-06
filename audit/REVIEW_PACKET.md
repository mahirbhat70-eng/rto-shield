# Final Hardening Review Packet for Independent Reviewer

> **Auditor Note for Reviewer using an Independent LLM / Model Family:**  
> This packet provides an adversarial, unvarnished index of all modified files, exact reproduction commands, verified results, root cause resolutions for earlier conflicting claims, questions we are least sure about, and surprising discoveries from the comprehensive consistency and correctness pass. Do not approve our work automatically; audit each claim against the reproduction commands below.

---

## 1. Inventory of Modified and Added Files

| File | Status | Nature of Change / Hardening Added |
|---|---|---|
| `scripts/regenerate_all.py` | Added / Hardened | Single master orchestrator regenerating ALL canonical JSON reports, CSV audits, and markdown tables from frozen config. |
| `scripts/generate_headline_numbers.py` | Hardened | Single source of truth generating `reports/headline_numbers.json` from frozen artifacts and test data. Adds per-1k COD metrics, oracle ceiling, and dynamic notes. |
| `reports/headline_numbers.json` | Generated | Pinned headline metrics: ₹54,936.40 realized savings (₹7,657.71/1k COD), ₹56,871.59 expected savings (₹7,927.46/1k COD), 271 passing tests. |
| `configs/cost_config.yaml` | Hardened | Aligned decomposed friction components to exactly ₹7.00 deposit (PG ₹3.80 + Refund ₹1.20 + Support ₹1.25 + WhatsApp ₹0.75) and ₹2.00 verify. |
| `src/policy/cost_engine.py` | Hardened | Added decomposed friction calculation methods and wired them into intervention pricing and expected loss. |
| `tests/test_cost_engine.py` | Hardened | Added behavioral mutation test (`test_deposit_component_zero_mutation_changes_engine_loss`) verifying zeroing any component changes engine loss. |
| `tests/test_no_stale_numbers.py` | Added | Programmatic assertion that every JSON report matches current config SHA-256 and that break-even sweep at natural rate reproduces headline realized savings. |
| `reports/seed_sweep.json` & `audit/seed_sweep.csv` | Rebuilt | Rebuilt 5-seed sweep across 4 models where EACH model routes its own decisions on full 100k generated splits with true Bayes PR-AUC ceiling and Oracle Realized Savings ceiling. |
| `reports/breakeven.json` | Rebuilt | Reweighting sweep (crossover 17.02%, CI [16.46%, 17.66%], slope ₹4,884.48/point) and odds shift comparison (crossover 17.40%, safety-gated 18.25%). |
| `reports/cap_table.json` | Generated | 40/60/80/100% cap table for Naive vs Ranked selection with served savings, customer loss, revenue lost, and RTOs prevented. |
| `reports/misspecification_grid.json` & `audit/misspecification_grid.csv` | Generated | 81 joint sensitivity cells evaluated on canonical config. Win rate vs Always-Allow: 76.54%. |
| `reports/realism_sweep.json` | Generated | Regimes (a) to (e) evaluated on canonical config with per-1k COD metrics, lift, and Zipf distribution statistics. |
| `reports/model_comparison.json` | Generated | Generator v2 (additive, H=0.003) vs v3 (interactions, H=0.184) model comparison and 1-SE D3 rule evaluation. |
| `reports/decomposed_costs.json` | Generated | Decomposed operational cost matrix at mean basket (₹826.89), ₹200, ₹3,000, plus LTV sensitivity. |
| `reports/latency_summary.json` | Generated | OS-level hardware descriptors (AMD Ryzen 5 5600H) and 5x1000 benchmark distribution ranges. |
| `audit/run_mutations.py` | Hardened | Expanded 26-mutation sabotage suite (26/26 caught = 100.0% catch rate). |
| `audit/prompt12/repro_check.py` | Hardened | Bitwise deterministic clean-rebuild verification across isolated seeds (13/13 artifacts match). |
| `README.md`, `claim-matrix.md`, `docs/JUDGE_QA.md` | Hardened | Programmatically synced test counts (271 passing tests) and headline figures via `scripts/sync_reports_to_docs.py`. |

---

## 2. Core Claims & Exact Reproduction Commands

| Claim | Verified Value | Exact Reproduction Command |
|---|---|---|
| **Headline Realized Savings (Canonical)** | ₹54,936.40 (on 7,174 COD test orders, ₹7,657.71 per 1,000 COD orders; old zero-friction was ₹69,786.08) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['realized_savings_inr'], d['savings_per_1k_cod_inr'])"` |
| **Headline Expected Savings (Canonical)** | ₹56,871.59 (₹7,927.46 per 1,000 COD orders; +3.52% expected-realized gap; old zero-friction was ₹71,741.02) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['expected_savings_inr'], d['expected_savings_per_1k_cod_inr'])"` |
| **Oracle Ceiling Realized Savings** | ₹229,164.00 (₹31,943.69 per 1,000 COD orders; model captures 23.97% of oracle ceiling) | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['oracle_ceiling_inr'], d['pct_of_oracle_ceiling'])"` |
| **5-Seed Shipped Model Stats (Canonical)** | Realized: ₹52,214.96 ± 1,738.03 (₹7,240.60 ± 294.41 / 1k COD); Expected: ₹57,230.01 ± 2,939.19; Gap: 9.65% ± 5.56% | `python -c "import json; d=json.load(open('reports/headline_numbers.json')); print(d['five_seed_sweep_shipped_model'])"` |
| **Test Set Statistical Metrics (N=14,980)** | LR: PR 0.3434, ROC 0.6854, Brier 0.1475<br>Uncal: PR 0.3433, ROC 0.6883, Brier 0.1473<br>Platt: PR 0.3433, ROC 0.6883, Brier 0.1477<br>Iso: PR 0.3313, ROC 0.6874, Brier 0.1476 | `python scripts/generate_headline_numbers.py` |
| **COD Break-Even Crossover (Reweighting)** | Mean: **17.02%** COD RTO (Bootstrap 95% CI: [16.46%, 17.66%]; Platform: **8.15%**; Slope: ₹4,884.48/point or ₹680.86/1k COD/point) | `python -c "import json; d=json.load(open('reports/breakeven.json')); print(d['reweighting_crossover_pct'], d['reweighting_crossover_bootstrap_95_ci'])"` |
| **COD Break-Even Crossover (Guardrail Odds Shift)**| Point estimate: **17.40%**; Safety-gated lower 95% CI bound (> ₹0.25 margin): **18.25%** | `python -c "import json; d=json.load(open('reports/breakeven.json')); print(d['odds_shift_crossover_point_pct'], d['guardrail_safety_gated_crossover_pct'])"` |
| **Controls Lift over Pincode Model** | Pincode model: ₹40,691.15 (₹5,672.03/1k COD)<br>Shipped model: ₹54,936.40 (₹7,657.71/1k COD)<br>Lift: **+35.01%** (+₹1,985.68 per 1k COD) | `python -c "import json; d=json.load(open('reports/controls.json')); print(d['lift_over_pincode_rule_pct'])"` |
| **60% Ranked Rate Cap Retention** | Pure: ₹54,936.40<br>Naive 60% Cap: ₹31,009.07 (-43.56%)<br>Ranked 60% Cap: ₹51,715.91 (-5.86%, captures 94.14% of pure savings) | `python -c "import json; d=json.load(open('reports/cap_table.json')); print(d['caps'][1])"` |
| **Sabotage Mutation Catch Rate** | **26/26 caught (100.0%)** | `python audit/run_mutations.py` |
| **Full Test Suite Status** | **271 passed, 0 failed** in 56.7s | `python -m pytest tests/ -q` |
| **Bitwise Clean Reproducibility** | Determinism: PASS (13/13 artifacts identical across runs)<br>Parity: PASS (13/13 artifacts match committed hashes) | `python audit/prompt12/repro_check.py --runs 2` |

---

## 3. Every Discrepancy Resolved and Where Earlier Reports Were Wrong

1. **Two Incompatible Savings Scales:**
   - *Earlier error:* Seed 42 realized was reported as ₹13,293.75 in the 5-seed sweep table, but ₹54,936.40 in the headline table.
   - *Root cause:* The earlier 5-seed sweep script generated only 20,000 orders total and evaluated on a 6,000-order slice with ~2,800 COD orders, whereas the headline evaluated on the canonical 14,980-order test split with 7,174 COD orders (2.5x more orders).
   - *Resolution:* All tables now report on the unified evaluation base: full 100k generated temporal splits ($N_{COD} \approx 7,174$ per seed), reporting both Total INR and Normalized INR per 1,000 COD orders. On this unified base, Seed 42 realized is ₹54,801.29 (₹7,638.87/1k COD), reproducing the headline within sampling tolerance!

2. **Broken 5-Seed Table & Identical Models:**
   - *Earlier error:* In the earlier 5-seed sweep, LGBM uncalibrated, Platt, and Isotonic showed identical realized savings in every seed, and LR showed 100.9% of ceiling.
   - *Root cause:* The script passed `p_tree_s` to all three LGBM model variations without fitting calibration, and defined "ceiling" as the uncalibrated tree's PR-AUC!
   - *Resolution:* Completely rebuilt `seed_sweep.py` / `regenerate_all.py`. Each model is properly fitted and calibrated on validation splits, computes its own predictions, and routes its OWN actions through the CostEngine. The ceilings are now mathematically defined: Bayes Ceiling PR-AUC = 0.3497, and Oracle Realized Savings Ceiling = ₹229,164.00 (₹31,943.69/1k COD).

3. **Break-Even Sweep Crossover Conflicts:**
   - *Earlier error:* One report stated crossover at 16.60%, another 17.02%, another 18.25%, with wildly different slopes (₹11.84k vs ₹6.2k vs ₹4.88k).
   - *Root cause:* There are TWO distinct ways to vary the COD RTO rate: (i) Importance Reweighting of test orders holding feature-label distributions fixed, and (ii) Odds/Intercept Shifting in the generative model. Furthermore, some scripts reported point estimate crossovers while `merchant_break_even` reported the safety-gated crossover requiring the lower 95% bound > ₹0.25 margin!
   - *Resolution:* Both methods are now explicitly distinguished side-by-side in `reports/breakeven.json`:
     - Reweighting point estimate crossover: **17.02%** (Slope: ₹4,884.48/point, ₹680.86/1k COD/point).
     - Odds-shifting point estimate crossover: **17.40%**.
     - Guardrail safety-gated crossover (lower 95% bound > ₹0.25 margin): **18.25%**.

4. **Deposit Cost Component Wiring:**
   - *Earlier error:* Components in `configs/cost_config.yaml` summed to ₹7.20 (4.00 + 1.20 + 1.25 + 0.75), but config stated ₹7.00. The engine also discarded components and only read `friction_cost: 7.00`.
   - *Root cause:* Payment gateway fixed fee was listed as ₹2.00 instead of ₹1.80 ($100 \times 0.02 + 1.80 = 3.80$; $3.80 + 1.20 + 1.25 + 0.75 = 7.00$ exactly). The engine also lacked dynamic wiring.
   - *Resolution:* Config updated to exact arithmetic (₹7.00 deposit, ₹2.00 verify). `CostEngine.compute_decomposed_friction()` dynamically computes component sums, and behavioral mutation tests verify that zeroing any component changes expected loss by that component's contribution.

5. **Controls Flipped Without Explanation:**
   - *Earlier error:* Random action mix flipped from positive to negative, and pincode rule went from ₹53.7k to ₹35.3k.
   - *Root cause:* Under ₹0 friction, random interventions appeared positive because preventing RTOs without fee penalty offset conversion drops. With ₹7 deposit and ₹2 verify friction, random interventions incur ~₹20,000 in operational waste, turning random action mix to -₹11,926.63 (-₹1,662.48/1k COD). Pincode rule differed between an ad-hoc if/else threshold rule vs routing the pincode rate through the CostEngine. Routing via CostEngine achieves ₹40,691.15, over which the shipped model delivers +35.01% lift (+₹1,985.68/1k COD).

6. **Hardware & Latency Claims:**
   - *Earlier error:* Intel CPU reported in early transcripts, and thermal throttling cited without evidence.
   - *Resolution:* Machine captured directly from OS: AMD Ryzen 5 5600H with Radeon Graphics (6 cores, 12 logical processors, 15.3 GB RAM, Windows 11 Build 26100). Latency ranges reported with full distribution: With SHAP p50 = 9.6–11.8 ms, p95 = 24–27 ms, p99 = 38–45 ms, cold start = 46 ms; Without SHAP p50 = 5.0–6.6 ms, p95 = 10–14 ms, p99 = 18–25 ms.

---

## 4. The Questions We Are LEAST Sure About

1. **Real-World Verification & Deposit Drop-Off Rates ($\delta_{\text{verify}}$, $\delta_{\text{deposit}}$):**
   - Good-customer drop-off is assumed to be 5% for verification and 40% for deposit.
   - Sensitivity sweeps show that if deposit drop-off exceeds 50% or verification drop-off exceeds 10%, a frozen policy becomes net-negative.
   - *Only real merchant A/B pilot data can resolve this.*

2. **Small-Sample Empirical Shrinkage ($M=10$) on Zipf Long-Tail:**
   - Under Zipf volume ($\alpha=1.1$), 66.1% of pincodes have fewer than 5 orders per month.
   - Out-of-distribution drift and extreme courier variance in rural pincodes require live merchant validation.

3. **Model Selection Circularity on Non-Linear Interactions:**
   - On additive synthetic data (generator v2), Logistic Regression achieves realized savings within 1 SE of LightGBM.
   - On generator v3, LightGBM wins because interaction terms were explicitly written into the generator. Real merchant data will determine the true non-linear structure.

---

## 5. Explicit List of What These Results Do NOT Prove

1. **Synthetic Data Only:** All evaluations are conducted on synthetic order distributions.
2. **Assumed Intervention Effects:** RTO reduction (30% verify, 80% deposit) and checkout drop-off (5% verify, 40% deposit) are assumed parameters, not measured in live A/B trials.
3. **No Real Merchant Validation:** RTO-Shield has not yet been deployed to live production traffic.
4. **Long-Term Customer Lifetime Value (LTV):** Brand equity erosion and customer attrition from friction are not captured in the baseline single-order financial model.
