# Judge Q&A — RTO Shield

> Every answer below traces to a frozen artifact or a reproducible command.  
> No number in this file was written from memory.

---

## Q1: "What is your headline result?"

**A:** The multi-action policy earns **₹71,741 in expected-loss savings** on a fully held-out test window
(7,174 COD orders, never touched during training or threshold tuning) — **2.0× the savings of the
best single-threshold strategy**. Realized-label P&L cross-check: **₹69,786** (−2.7% of EL estimate,
within the Monte Carlo P5–P95 band of ₹63,935–₹75,901).

**Artifact:** `reports/stage5_test_results.md` § 3 & § 2

---

## Q2: "How do I know the test set was never contaminated?"

**A:** Three mechanical guarantees:
1. Temporal split — all splits are ordered by `order_date`; test = the chronologically final window.
2. Threshold tuning used `val_cal` only (`val_rep` and `test` were sealed); see `src/policy/` for the argmin loop.
3. Cold-clone verification: after force-pushing the frozen artifacts, CI runs `pytest` from a fresh checkout on every push. All 57 tests pass — including `test_split_integrity` which explicitly checks date monotonicity.

**Artifact:** `.github/workflows/ci.yml`, `tests/test_stage3.py::test_split_date_monotonicity`

---

## Q3: "94.7% of Bayes ceiling — what does that actually mean?"

**A:** We computed the PR-AUC achievable by a hypothetical oracle that knows the exact empirical RTO
rate per calibration bin. That ceiling is **0.3497** on test. Our calibrated LightGBM scores
**0.3313** — i.e., 94.7% of the maximum any model family could achieve given this feature set and
label noise. The ceiling was pre-registered before test reveal.

**Implication:** More complex models (XGBoost, neural nets) cannot close the remaining 5.3% gap in
any financially meaningful way. **The value is in the decision layer, not the model.**

**Artifact:** `reports/stage5_test_results.md` § 1 (Bayes Ceiling PR-AUC 0.3497)

---

## Q4: "Why does uncalibrated router show higher EL savings (₹82,580) than calibrated (₹71,741)?"

**A:** Calibration bought honest probabilities — which forced the router to correctly classify borderline
orders into the cheaper VERIFY bucket instead of confidently pushing into DEPOSIT. The uncalibrated
model is overconfident on the high-risk tail. In realized-label P&L, the gap collapses to ₹116
(immaterial). We accepted the calibration tax for honest confidence intervals and per-action
calibration error ≤ 0.003 across all three actions.

**Artifact:** `reports/stage5_test_results.md` § 5 (Sensitivity: Uncal) and § 2 (Realized P&L)

---

## Q5: "Your policy drops 820 good customers. Isn't that too high?"

**A:** 820 expected good-customer drops on 7,174 COD orders (~11.4%) is the cost of the high-recall
DEPOSIT band. The policy trades them for **951 expected RTOs prevented** at ₹826.89 mean order
value. Net EV is positive by construction (argmin cost engine, tuned on val_cal). Friction spend:
₹6,488 total = ₹6.82 per order touched. Binary VERIFY drops only 207 good customers but prevents
only 546 RTOs and costs 2× in friction per RTO prevented.

**Artifact:** `reports/stage5_test_results.md` § 3

---

## Q6: "How long does a single inference take?"

**A:** End-to-end latency (feature construction → model score → argmin decision → response):

| Percentile | Latency |
|-----------|---------|
| p50 | ~3ms |
| p95 | ~8ms |
| p99 | ~15ms |

Measured via `scripts/benchmark_latency.py` (1,000 warmup + 10,000 timed iterations, single core).
The decision engine alone (argmin cost loop over 4 actions) is <0.1ms.

**Command:** `python scripts/benchmark_latency.py`

---

## Q7: "What would break this system in production?"

**A:** See `WHAT_BROKE.md` for the full engineering failure log. Three production risks identified:

1. **Pincode cold-start**: new pincodes have no historical RTO rate; model falls back to global mean (0.28), biasing toward VERIFY.
2. **COD charge drift**: `num_cod_charge` is the top SHAP feature. Retrain trigger: p75 COD charge drifts >20% over 30 days.
3. **Label delay**: RTO labels arrive 7–14 days post-delivery. Any retraining loop must account for the label lag window.

---

## Q8: "Is the 2.0× claim cherry-picked?"

**A:** No. The baseline is the **best** single-threshold strategy on the same test window: Binary VERIFY
at threshold 0.20, saving ₹35,919. Multi-action saves ₹71,741. Ratio: 2.0× (71,741 / 35,919 = 1.998).
Binary PREPAID (₹930) is included as a dominance check, not to flatter the ratio.

**Artifact:** `reports/stage5_test_results.md` § 3, rows 3 and 4

---

## Q9: "Can I reproduce the numbers right now?"

**A:** Yes. See **Judge Quickstart** in `README.md`:
```bash
git clone https://github.com/mahirbhat70-eng/rto-shield
cd rto-shield
pip install -r requirements.txt
pytest tests/ -v   # 57/57 pass
python scripts/benchmark_latency.py
```
Frozen model artifacts in `src/models/` and frozen reports in `reports/` are committed to the repo.
No external data download required.

---

## Q10: "Why Streamlit for the demo?"

**A:** The Streamlit demo is a judge-facing evaluation interface, not a production deployment pattern.
Production would expose `src/serve/` as a REST endpoint (FastAPI, ~20 lines). Streamlit was chosen
for interactive transparency: judges see the full decision surface (expected-loss table, SHAP features,
calibrated P) in real time without reading code.

**Live demo:** https://rto-shield-nlthpydtndpkupyfgfl3yy.streamlit.app

---

## Q11: Your "realized savings of ₹69,786 (−2.7% vs forecast)" — since no order was actually intervened upon, what exactly did you validate?

**A:** Forecast consistency, not intervention efficacy. "Realized" = each held-out order's actual, un-intervened outcome label scored through the same cost model and the routed action's assumed effects (`src/eval/stage5_2_realized_pl.py`). No counterfactual outcome exists for intervened orders, so no offline evaluation can validate efficacy — that requires the randomized pilot in our roadmap. What the −2.7% gap does prove: the calibrated forecast held on real outcomes under the model's own assumptions, while the uncalibrated router's forecast failed the identical test at −15.3% — which is why calibration was non-negotiable even at a ₹10.8k headline cost.

**Artifact:** `src/eval/stage5_2_realized_pl.py`, `reports/stage5_test_results.md` § Stage 5.2, per-action calibration |Δ| ≤ 0.0032
