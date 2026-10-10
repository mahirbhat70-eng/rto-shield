# Data Pipeline Engineering Notes (Task 0.5)

> **File:** `docs/commercial/data_pipeline_notes.md`  
> **Status:** Phase 0.5 Complete  
> **Scope:** Audit and documentation of historical feature generation and non-linear interactions  

---

## 1. History-Append Analysis (`generator_v2.py:167`)

### Issue
In `src/data/generator_v2.py:167`, the sequential order generation loop precomputes customer history:
```python
for i in order_rows:
    c = cust_idx[i]
    t = ts_vals[i]
    hist = history.setdefault(c, [])
    # ponytail: sequential multi-order history append deferred; prior_orders and prior_rto are 0 cold-start baseline
    n_res = 0
    n_rto = 0
    for arr, was_rto in hist:
        if arr <= t:
            n_res += 1
            n_rto += int(was_rto)
    prior_orders[i] = n_res
    prior_rto[i] = n_rto
```
Because `hist.append((arr, was_rto))` was intentionally deferred (to avoid lookahead bias on unresolved future package transit times in the synthetic generator), `prior_orders` and `prior_rto` evaluate to zero across all rows.

### Decision & Commercial Architecture
1. **Synthetic Cohort (Frozen Reproducibility):** In the synthetic test benchmark, all customers represent a **cold-start baseline** ($n=0$ prior orders). This ensures zero lookahead leakage and guarantees reproducibility across all 271 existing tests.
2. **Real-Data Production Ingestion (`src/ingest/shopify_csv.py`):** When ingesting real Shopify and Shiprocket CSVs, `prior_orders` and `prior_rto` are computed deterministically by querying strictly settled historical orders (`timestamp < current_order_timestamp`). Cold-start customers default to 0 with explicit model card disclosure.

---

## 2. Interaction Features Analysis (`generator.py:313-321`)

### Context
In `src/data/generator.py:308-326`, non-linear feature interaction terms are defined:
- `inter_ov_tier`: High order value $\times$ Tier-3 pincode ($0.75$)
- `inter_new_disc_cod`: New customer $\times$ Discount $\times$ COD ($1.10$)
- `inter_courier_tier`: Courier mismatch $\times$ Pincode Tier 3 ($0.60$)
- `inter_bot_cluster`: Velocity $\times$ Device cluster syndicate ($0.85$)

### Decision
- In standard generation (`generator_v2` / baseline additive mode), interactions are switched off (`interactions_term = 0.0`), reflecting the linear additive risk hypothesis where Logistic Regression wins under the D3 rule ($H = 0.003$).
- These non-linear terms are explicitly activated when running `generator_version == "v3"` (or `--interactions`), which drives the stress-testing matrix in `reports/model_comparison.json:35-65` where LightGBM with tree splits outperforms linear baselines ($H = 0.184$).
- The code is retained with clear branch gating so both additive (D3 LR) and non-linear (tree robustness) experiments remain 100% reproducible.
