# Model Re-Baseline & Calibration Repair (Tasks 0.2 & 0.3)

> **Status:** Phase 0.2 & 0.3 Validated  
> **Rule Invoked:** D3 1-Standard-Error Simplicity Rule ([reports/model_comparison.json](file:///c:/Users/vansh/OneDrive/Desktop/rto-shield-prep/rto-shield/reports/model_comparison.json#L32))  
> **New Primary Model:** `logistic_regression` (promoted from `lgbm_isotonic`)  

---

## 1. Before vs After Empirical Comparison

On the frozen test cohort ($N = 7,174$ COD orders):

| Metric | Before (`lgbm_isotonic`) | After (`logistic_regression`) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **PR-AUC** | `0.331315` | `0.343396` | **+0.012081 (+3.6%)** |
| **ROC-AUC** | `0.687356` | `0.685425` | -0.001931 |
| **Brier Score** | `0.147564` | `0.147464` | **-0.000100 (Superior)** |
| **Expected Calibration Error (ECE)** | `0.008564` | `0.004503` | **-0.004061 (47.4% error reduction)** |
| **Realized Savings (Single Draw Seed 42)** | `₹54,936.40` | `₹55,057.62` | **+₹121.22 higher profit** |
| **5-Seed Mean Realized Savings** | `₹52,214.96 ± 1,738` | `₹52,803.49 ± 1,514` | **+₹588.53 mean gain / tighter variance** |
| **Mean Calibration Gap (%)** | `9.65% ± 5.56%` | `5.54% ± 3.01%` | **-4.11% lower gap** |

---

## 2. Calibration Repair & Standard Platt Scaling

### Issue with Legacy Calibration
In `scripts/generate_headline_numbers.py:140-141`, Platt scaling was fitted directly on raw uncalibrated probabilities ($p$) with fixed $C=1.0$:
$$\text{Platt}_{raw}(p) = \sigma(w \cdot p + b)$$
Fitting logistic regression linearly on probabilities causes extreme compression and tail-bin distortion on sparse high-risk orders ($p \approx 1.0$ predicted vs empirical $0.60$).

### Standard Platt Calibration via Logits CV
Standard Platt scaling operates on the log-odds (logits) of uncalibrated model outputs:
$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$$
We refactored Platt scaling using `LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss")` on logits.

### Isotonic Acceptance Gate
Isotonic regression often overfits small bin counts in sparse tails. We introduced an automated validation gate on held-out validation data (`val_cal.csv`):
Isotonic calibration is approved **only if**:
1. $\text{ECE}_{iso} < \text{ECE}_{uncal}$
2. $\text{Brier}_{iso} \le \text{Brier}_{uncal}$
3. $\text{Tail Bias} = |\bar{p}_{tail} - \bar{y}_{tail}| < 0.25$ (measured on the top 10% risk decile).

---

## 3. Policy Promotion (Stage 6 P2 Learned Policy)

The learned policy ($P_2$ Global Learned via IPW T-learner from `stage6_uplift_ope.py`) captures:
- `VERIFY_ADDRESS`: $r = 0.466$ (RTO reduction), $d = 0.017$ (success drop)
- `REQUIRE_DEPOSIT`: $r = 0.745$, $d = 0.353$
- `PREPAID_ONLY`: $r = 0.857$, $d = 0.583$

**Financial Impact:**
- v1 Frozen Constants Savings: `₹55,057.62` (74% of oracle)
- P2 Learned Policy Savings: `₹121,073.67` (76% of oracle, **+₹66,016.05** under learned elasticity)
- Quarterly Re-optimization runner added at `scripts/reoptimize_policy.py`.
