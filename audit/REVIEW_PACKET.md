# Review Packet for Independent Reviewer (Task 15)

> **Auditor Note for Reviewer using a DIFFERENT LLM / Model Family:**  
> This packet provides an adversarial, unvarnished index of all modified files, exact reproduction commands, verified results, root cause resolutions for earlier conflicting claims, and the remaining questions you should rigorously probe.

---

## 1. Inventory of Modified and Added Files

| File | Status | Nature of Change |
|---|---|---|
| `audit/seed_sweep.py` | Added | 5-seed clean evaluation with uncalibrated, Platt, isotonic LightGBM and Logistic Regression |
| `audit/seed_sweep.csv` | Added | Raw tabular records across seeds 42, 101, 2024, 777, 999 |
| `audit/task3_model_metrics.py` | Added | Single-script test set evaluation (PR-AUC, ROC-AUC, Brier, Log Loss, ECE) for all 4 model variants |
| `audit/control_policies.py` | Added | Strict evaluation of random score routing, action distribution match, and pincode-only rules vs model |
| `src/serve/scorer.py` | Modified | Added `SecurityError`, pre-deserialization SHA-256 verification, and support for out-of-boundary pinned hashes via `RTO_SHIELD_PINNED_DIGESTS` |
| `src/serve/guardrails.py` | Added | Production safety guardrails (Kill Switch, Rate Cap, High Value Review, Dynamic Break-Even Gate) |
| `src/serve/audit.py` | Modified | Replaced ad-hoc regex PII masking with strict `AUDIT_ALLOWLIST_KEYS` and salt-hashing of IDs |
| `tests/test_artifact_integrity.py` | Modified | Added 1-byte file mutation load refusal test and out-of-boundary hash manifest override test |
| `tests/test_stage3.py` | Modified | Added model degradation assertions: test PR-AUC $\ge 0.32$ and test gap vs label-shuffled baseline $\ge 0.10$ |
| `tests/test_sensitivity.py` | Modified | Added frozen-policy single-parameter misspecification sweeps (deposit red 60/40/20%, verify red 15/0%, verify drop 10/20%) |
| `tests/test_guardrails.py` | Added | Unit tests for all guardrails and closed-form dynamic merchant break-even calculation formula |
| `tests/test_independent_thresholds.py` | Added | Closed-form threshold derivation test and automated assertion that README test count matches `pytest --collect-only` |
| `README.md`, `claim-matrix.md`, `docs/JUDGE_QA.md`, `docs/pitch_script.md` | Modified | Synchronized test suite count to 244 tests; clarified TreeSHAP (~15ms) vs raw scoring (~3ms) latency; reconciled design-point savings to Rs 69,786.08 |

---

## 2. Core Claims & Exact Reproduction Commands

| Claim | Verified Value | Exact Reproduction Command |
|---|---|---|
| **5-Seed Shipped Model Savings** | Mean: Rs 66,333.14 (Std: Rs 2,147.73) | `python -c "import pandas as pd; df=pd.read_csv('audit/seed_sweep.csv'); print(df['LGBM_iso_savings_Rs'].mean(), df['LGBM_iso_savings_Rs'].std())"` |
| **Test Set Statistical Metrics (N=14,980)** | LR: PR 0.3434, LogLoss 0.4627, Brier 0.1475<br>Uncal LGBM: PR 0.3433, LogLoss 0.4621, Brier 0.1473<br>Iso LGBM: PR 0.3313, LogLoss 0.4729, Brier 0.1476<br>Platt LGBM: PR 0.3433, LogLoss 0.4639, Brier 0.1477 | `python audit/task3_model_metrics.py` |
| **Control Policies vs Model** | Baseline: Rs 0.00<br>Random Uniform: -Rs 11,331.20<br>Random Action Mix: -Rs 19,612.62<br>Pincode-Only Rule: Rs 53,687.54<br>Primary Model: Rs 69,786.08 (+30.0% lift) | `python audit/control_policies.py` |
| **Artifact Byte Tamper Refusal** | Refuses with `SecurityError: SHA-256 digest mismatch` | `python -m pytest tests/test_artifact_integrity.py -v` |
| **Model Degradation & Shuffled Gap** | Production PR-AUC 0.3313 vs Shuffled 0.1941 (Gap = 0.1372 $\ge 0.10$) | `python -m pytest tests/test_stage3.py -k test_stage3_performance_floor -v` |
| **Misspecification Breakdown** | Deposit Red 20%: -Rs 6,533.92 (Loss)<br>Joint Pessimistic 1: -Rs 5,870.94 (Loss)<br>Deposit Red $\ge 40\%$ & Verify Red $\ge 0\%$: Profitable | `python -m pytest tests/test_sensitivity.py -k test_frozen_policy_single_parameter_misspecifications -v` |
| **Automated Test Suite Count** | Exactly 244 tests passing, 0 failures, synchronized with README | `python -m pytest tests/ -q` |

---

## 3. Discrepancy & Root Cause Explanations

1. **Reconciliation of Rs 68,480.08 vs Rs 69,786.08 (Task 2):**
   - *Cause:* `audit/run_step3_verifications.py` line 132 hardcoded a fictitious Rs 0.50 friction cost on `REQUIRE_DEPOSIT` ($2,612 \times 0.50 = \text{Rs } 1,306.00$). Subtracting Rs 1,306 from Rs 69,786.08 yielded Rs 68,480.08.
   - *Fix:* Reconciled to canonical `configs/cost_config.yaml` where deposit friction cost is 0 (UPI token collection has no gateway MDR). Verified headline realized savings is **Rs 69,786.08**.

2. **Ceiling Discrepancies (~0.46 vs ~0.35) (Task 1):**
   - *Cause:* Population definitions differed. The observable ceiling on ALL orders using feature conditional expectation $\mathbb{E}[y \mid x]$ is **0.3497** (Seed 42) / Mean **0.3452**. Prompt 6 used the latent generator prior $p_{latent} = \text{sigmoid}(z + \epsilon)$ which includes unobservable noise, yielding an apparent ceiling of ~0.46 to 0.48.

3. **Break-Even Discrepancies (-Rs 26,921 vs -Rs 10,902) (Task 7):**
   - *Cause:* The test set has an overall platform RTO rate of 19.78% ($N=14,980$) but a COD-subset RTO rate of 28.27% ($N=7,174$).
   - A 15% overall platform RTO rate resamples COD orders down to ~21.8% COD RTO, producing **-Rs 26,921.43** under the old cost weights. Direct COD resampling at 15% COD RTO produces **-Rs 21,848.68** under canonical costs. Both populations are now explicitly labeled in `audit/audit_prompt11.py`.

---

## 4. Questions We Are Least Sure About (For the Independent Reviewer)

1. **Synthetic Generator Linearity vs Real-World Nonlinearity:**
   - The generator in `src/data/generator.py` generates risk as an additive linear combination without feature interactions. This explains why Logistic Regression (PR-AUC 0.3434) beats or matches LightGBM (0.3433 uncal, 0.3313 iso). In real merchant traffic with fraud syndicates and courier non-linearities, will LightGBM outperform LR, or is an ensemble strictly superior?
2. **Behavioral Customer Elasticity Unobservability:**
   - Offline holdout evaluations assume fixed behavioral responses ($40\%$ drop on deposit, $80\%$ RTO reduction). As shown in Task 10, if deposit RTO reduction falls below $25.1\%$, or in joint misspecification, the policy turns net negative. Only live shadow mode or randomized pilot bandit testing can validate actual customer elasticity.
3. **Platt vs Isotonic Calibration in Serving:**
   - Isotonic calibration creates 97 discrete probability plateaus (79.8% flat curve share), degrading fine-grained ranking. While Platt scaling is smooth and preserves ranking, it assumes a sigmoidal distortion. Should the production system migrate entirely to Platt scaling or temperature scaling?
