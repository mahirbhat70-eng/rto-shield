# Claim Matrix — Every Number, Its Source, Its Command

> If a number appears in the README or submission form, it appears here with the exact artifact and command that produced it.

| Claim | Value | Source Artifact | Reproducing Command |
|-------|-------|-----------------|---------------------|
| Portfolio profit uplift | 13.1% | `reports/stage5_test_results.md` § 3 | `pytest tests/test_stage5.py::test_primary_savings_uplift -v` |
| Multi-action savings (EL) | ₹71,741 | `reports/stage5_test_results.md` § 3, row 4 | `pytest tests/test_stage5.py -v` |
| Multi-action savings (realized) | ₹69,786 | `reports/stage5_test_results.md` § 2, row 4 | `pytest tests/test_stage5.py -v` |
| 2.0× vs best single-threshold | 71741 / 35919 = 1.998 | `reports/stage5_test_results.md` § 3, rows 3–4 | Manual: 71741 / 35919 |
| Bayes ceiling PR-AUC | 0.3497 | `reports/stage5_test_results.md` § 1 | `pytest tests/test_stage5.py::test_bayes_ceiling -v` |
| Primary model PR-AUC (test) | 0.3313 | `reports/stage5_test_results.md` § 2 | `pytest tests/test_stage5.py -v` |
| % of Bayes ceiling | 94.7% | 0.3313 / 0.3497 | Manual: 0.3313/0.3497 |
| COD-subset RTO rate | 28.27% | `reports/stage5_test_results.md` § 3 | `pytest tests/test_stage5.py -v` |
| Mean calibrated P | 0.2832 | `reports/stage5_test_results.md` § 3 | `pytest tests/test_stage5.py -v` |
| Mean order value (COD) | ₹826.89 | `reports/stage5_test_results.md` § 3 | `pytest tests/test_stage5.py -v` |
| Test COD subset size | 7,174 orders | `reports/stage5_test_results.md` § 1 | `pytest tests/test_stage3.py::test_split_integrity -v` |
| RTOs prevented (primary) | 951 | `reports/stage5_test_results.md` § 3, row 4 | `pytest tests/test_stage5.py -v` |
| Good-customer drops (primary) | 820 | `reports/stage5_test_results.md` § 3, row 4 | `pytest tests/test_stage5.py -v` |
| Friction spend (primary) | ₹6,488 | `reports/stage5_test_results.md` § 3, row 4 | `pytest tests/test_stage5.py -v` |
| Action dist (primary) | 18.4/45.2/36.4/0.0 | `reports/stage5_test_results.md` § 3, row 4 | `pytest tests/test_stage5.py -v` |
| Noise sensitivity delta | +2.7% | `reports/stage5_test_results.md` § 4 | `pytest tests/test_stage5.py::test_noise_sensitivity -v` |
| Monte Carlo mean savings | ₹69,942 | `reports/stage5_test_results.md` § 2 MC | `pytest tests/test_stage5.py -v` |
| Monte Carlo P5 | ₹63,935 | `reports/stage5_test_results.md` § 2 MC | `pytest tests/test_stage5.py -v` |
| Monte Carlo P95 | ₹75,901 | `reports/stage5_test_results.md` § 2 MC | `pytest tests/test_stage5.py -v` |
| P(savings > 0) | 100% | `reports/stage5_test_results.md` § 2 MC | `pytest tests/test_stage5.py -v` |
| VERIFY action calibration error | 0.0001 | `reports/stage5_test_results.md` § 2 per-action | `pytest tests/test_stage5.py -v` |
| DEPOSIT action calibration error | 0.0032 | `reports/stage5_test_results.md` § 2 per-action | `pytest tests/test_stage5.py -v` |
| ALLOW action calibration error | 0.0032 | `reports/stage5_test_results.md` § 2 per-action | `pytest tests/test_stage5.py -v` |
| Calibration cost vs uncal | ₹10,839 EL | 82580 − 71741 | `reports/stage5_test_results.md` rows 4–5 |
| Tail-clip sensitivity | Δ₹140 EL | 71881 − 71741 | `reports/stage5_test_results.md` rows 4–6 |
| Total automated tests | 57 | CI badge | `pytest tests/ -v --tb=short` |
| Pre-registered checks passed | 11/11 | `reports/stage5_test_results.md` throughout | `pytest tests/test_stage5.py -v` |
| Inference latency p50 | ~3ms | `scripts/benchmark_latency.py` | `python scripts/benchmark_latency.py` |
| Inference latency p95 | ~8ms | `scripts/benchmark_latency.py` | `python scripts/benchmark_latency.py` |
| Inference latency p99 | ~15ms | `scripts/benchmark_latency.py` | `python scripts/benchmark_latency.py` |
| PREPAID threshold | 0.48 | `reports/stage4_financial_results.md` | `pytest tests/test_stage4.py -v` |
| VERIFY threshold | 0.20 | `reports/stage4_financial_results.md` | `pytest tests/test_stage4.py -v` |
| Decision audit trail export | JSONL format | `app.py` session state | Streamlit UI export (`rto_audit_trail.jsonl`) |

---

## How to verify any number in 60 seconds

```bash
git clone https://github.com/mahirbhat70-eng/rto-shield
cd rto-shield
pip install -r requirements.txt
pytest tests/ -v          # all 57 tests, all frozen artifacts
```

The test suite is the claim matrix made executable. Every row in the table above corresponds to an assertion in `tests/`.
