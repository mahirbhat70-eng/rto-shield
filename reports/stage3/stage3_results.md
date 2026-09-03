# Stage 3 Results on `val_rep`

| Model                    |   Precision |   Recall |     F1 |   PR-AUC |   ROC-AUC |   Brier |   Flag Rate |   Prec@Rule Recall |
|:-------------------------|------------:|---------:|-------:|---------:|----------:|--------:|------------:|-------------------:|
| Rule Baseline            |      0.2386 |   0.3120 | 0.2704 |   0.2118 |  nan      |  0.3361 |      0.2610 |             0.2386 |
| Logistic Regression      |      0.5333 |   0.0264 | 0.0504 |   0.3395 |    0.6728 |  0.1497 |      0.0099 |             0.3764 |
| Uncalibrated Tree (LGBM) |      0.5614 |   0.0212 | 0.0408 |   0.3409 |    0.6741 |  0.1496 |      0.0075 |             0.3645 |
| Calibrated Tree (LGBM)   |      0.6111 |   0.0145 | 0.0284 |   0.3276 |    0.6729 |  0.1498 |      0.0048 |             0.3643 |

Uniform bin-MAE (n_bins=10, uniform): LR 0.0410 (7/10 bins), uncalibrated tree 0.0327 (7/10), calibrated tree 0.1685 (9/10). Decomposition: 95.5% of the calibrated figure comes from three single-row bins above P=0.7 — one genuine isotonic tail overfit (a label-0 row predicted at P=1.0000, above the generator's 0.85 clip bound) and two rows predicted 0.75/0.81 with label=1 (directionally correct; single-row empirical rates are mechanically 0/1). The populated body (P<=0.6, 7,575/7,578 rows) tracks within 0.027 per bin. Row-weighted bin-MAE: calibrated 0.0074 (best of three). Brier (0.1498 vs 0.1497) and ROC-AUC (0.6729) unaffected. Financial impact (Stage 4): clipping the tail shifts portfolio loss by ₹21; isotonic ranking compression costs ₹4,980 in policy savings vs the uncalibrated router (disclosed sensitivity row).
