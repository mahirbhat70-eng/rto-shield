# Stage 5 Test Reveal Results

## 1. Split Profile & Bayes Ceiling
| Metric | val_rep | test | Change | Note |
|--------|---------|------|--------|------|
| Split Size (Rows) | 7,578 | 14,980 | +97.7% | Expected (val split in half for cal/rep) |
| COD Subset Size | 3,645 | 7,174 | +96.8% | Expected |
| RTO Rate | 0.2010 | 0.1978 | -1.6% | PASS (Band: [0.18, 0.22]) |
| Bayes Ceiling PR-AUC | 0.3462 | 0.3497 | +1.0% | PASS (Band: [0.31, 0.39]) |
| Ceiling Max Decile Dev | 0.0252 | 0.0345 | +36.9% | PASS (≤ ~0.035) |

## 2. Model Metrics (Frozen Artifacts)
| Model | Metric | val_rep | test | Change | Note |
|-------|--------|---------|------|--------|------|
| **Rule Baseline** | PR-AUC | 0.2311 | 0.2339 | +1.2% | |
| **LR@0.5** | PR-AUC | 0.3406 | 0.3434 | +0.8% | |
| **LR@0.5** | Brier | 0.1497 | 0.1475 | -1.5% | |
| **LR@0.5** | W-Bin-MAE | 0.0048 | 0.0045 | -6.2% | |
| **Uncal Tree** | PR-AUC | 0.3409 | 0.3433 | +0.7% | |
| **Uncal Tree** | Brier | 0.1496 | 0.1473 | -1.5% | |
| **Uncal Tree** | W-Bin-MAE | 0.0032 | 0.0079 | **+146.9%** | **FLAG (>10%)** |
| **Cal Tree (PRIMARY)** | PR-AUC | 0.3276 | 0.3313 | +1.1% | PASS (Band: [0.29, 0.37]) |
| **Cal Tree (PRIMARY)** | Brier | 0.1498 | 0.1476 | -1.5% | |
| **Cal Tree (PRIMARY)** | W-Bin-MAE | 0.0074 | 0.0086 | **+16.2%** | **FLAG (>10%)** |

*Pre-Registered Checks:*
- Cal Tree PR-AUC ratio to Ceiling: **0.947** (PASS, ≥ 0.88)
- FLAG: Uncal tree W-Bin-MAE +146.9%. Absolute error remains <=0.86%; metric is high-variance across windows; no action; primary unchanged.
- FLAG: Cal tree (PRIMARY) W-Bin-MAE +16.2%. Absolute error remains <=0.86%; metric is high-variance across windows; no action; primary unchanged.

### Test Per-Bin Table (Calibrated Tree)
| p_bin | rows | mean_predicted | empirical_rate |
|-------|------|----------------|----------------|
| (-0.001, 0.1] | 2,375 | 0.0685 | 0.0707 |
| (0.1, 0.2] | 5,988 | 0.1444 | 0.1364 |
| (0.2, 0.3] | 3,974 | 0.2460 | 0.2519 |
| (0.3, 0.4] | 2,072 | 0.3598 | 0.3431 |
| (0.4, 0.5] | 526 | 0.4460 | 0.4658 |
| (0.5, 0.6] | 39 | 0.5984 | 0.4615 |
| (0.7, 0.8] | 1 | 0.7150 | 0.0000 |
| (0.9, 1.0] | 5 | 1.0000 | 0.6000 |

*Observation:* Deviations in the populated body alternate in sign (-0.002, +0.008, -0.006, +0.017, -0.020), which indicates sampling noise rather than systematic one-signed drift.

## 3. Policy Evaluation (COD Subset)
*(Note: Absolute numbers scale ~2x vs val_rep because test has ~2x the COD rows)*

- **COD-Subset RTO Rate:** 28.27%
- **Mean Calibrated P:** 0.2832
- **Order Value (COD):** Mean ₹826.89 / Median ₹607.57

| Strategy | Savings vs Baseline | Action Dist (ALLOW / VERIFY / DEPOSIT / PREPAID) | Orders Touched | Friction Spend | Expected RTOs Prevented | Expected Good-Customer Drops |
|----------|---------------------|--------------------------------------------------|----------------|----------------|-------------------------|------------------------------|
| 1. Baseline | 0.00 | 100.0% / 0.0% / 0.0% / 0.0% | 0 | 0 | - | - |
| 2. Binary PREPAID | 929.63 | 99.3% / 0.0% / 0.0% / 0.7% | 49 | 0 | 17.02 | 12.63 |
| 3. Binary VERIFY | 35,919.42 | 17.0% / 83.0% / 0.0% / 0.0% | 5,954 | 11,908 | 546.48 | 206.62 |
| 4. Primary (Cal) | 71,741.02 | 18.4% / 45.2% / 36.4% / 0.0% | 5,856 | 6,488 | 951.16 | 820.08 |
| 5. Sens (Uncal) | 82,579.63 | 18.7% / 46.5% / 34.8% / 0.0% | 5,832 | 6,672 | - | - |
| 6. Sens (Clipped) | 71,881.29 | 18.4% / 45.2% / 36.4% / 0.0% | 5,856 | 6,488 | - | - |

*Pre-Registered Checks:*
- Primary PREPAID_ONLY share == 0%: **PASS** (0.0%)
- Loss Ordering (Multi Primary <= Verify Block <= Prepaid Block): **PASS** (-619,507 <= -583,686 <= -548,696)
- Prepaid Block within 2% of Baseline Loss: **PASS** (1.001x)
- Primary Savings Uplift [8%, 18%] of Baseline Loss: **PASS** (13.1%)

## 4. Noise Sensitivity Test
- **Clean Primary Savings:** 71,741.02
- **Noisy Primary Savings:** 73,672.00
- **Delta:** +1,930.98 (+2.7%)
- *Pre-Registered Check:* Noisy savings within ±10% of clean: **PASS** (+2.7%)

## Stage 5.2: Operating-Point Metrics & Realized P&L

### 1. Policy Operating-Point P/R (Binary: VERIFY/DEPOSIT = Positive)
- **Precision:** 0.2963
- **Recall:** 0.8555
- **F1:** 0.4401
- **Confusion Matrix:**
  - True Negative (ALLOW, actual 0): 1025
  - False Positive (VERIFY/DEPOSIT, actual 0): 4121
  - False Negative (ALLOW, actual 1): 293
  - True Positive (VERIFY/DEPOSIT, actual 1): 1735

### Per-Action Calibration
| Action | n | Mean P | Empirical RTO | \|Diff\| |
|---|---|---|---|---|
| REQUIRE_DEPOSIT | 2612 | 0.3279 | 0.3247 | 0.0032 |
| VERIFY_ADDRESS | 3244 | 0.2733 | 0.2734 | 0.0001 |
| ALLOW_COD | 1318 | 0.2191 | 0.2223 | 0.0032 |

### 2. Realized-Label Portfolio P&L
*(Evaluated using actual `rto_label` in the cost engine, with expected intervention effects)*

| Strategy | Realized Savings (INR) | EL(P) Savings (INR) | Delta (Realized - EL) |
|---|---|---|---|
| 1. Baseline | 0.00 | 0.00 | 0.00 |
| 2. Binary PREPAID | -954.67 | 929.63 | -1,884.29 |
| 3. Binary VERIFY | 36,018.74 | 35,919.42 | 99.32 |
| 4. Primary Multi-Action | **69,786.08** | **71,741.02** | **-1,954.95** |
| 5. Sensitivity: Uncal | 69,901.99 | 82,579.63 | -12,677.64 |
| 6. Sensitivity: Clipped | 69,786.08 | 71,881.29 | -2,095.21 |

### 3. Monte Carlo Simulation (5,000 draws)
*(Bernoulli-sampling individual intervention drops & reductions on Primary Multi-Action)*
- **Mean Savings:** ₹69,942.31
- **P5:** ₹63,935.40
- **P95:** ₹75,901.15
- **P(savings > 0):** 100.0%
