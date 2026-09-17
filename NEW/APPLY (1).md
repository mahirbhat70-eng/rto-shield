# JSONL Audit Trail — Apply Instructions

Feature: one JSONL record per scoring decision in the Streamlit demo,
downloadable from the UI. Frozen pipeline untouched (serving/UI layer only).

## What's in this change

| File | Status | Purpose |
|------|--------|---------|
| `app.py` | modified | Captures elapsed time, appends one audit record per scored order, "Audit Trail" expander with Download (JSONL) + Clear buttons |
| `src/serve/audit.py` | new | Pure stdlib record builder (`build_audit_record`, `records_to_jsonl`) — no streamlit imports, CI-testable |
| `tests/test_audit.py` | new | 5 tests: record shape/schema, JSONL round-trip, decision-id determinism, JSON serializability, PREPAID passthrough logging |

Test suite: **62 passed (57 existing + 5 new)** on sklearn 1.9.0 / lightgbm 4.7.0 / shap 0.52.0 (your pinned versions).
Live smoke-tested headless: score 2 orders → "Audit Trail (2 decisions logged this session)" → download button renders.

## Record schema (`rto-shield-audit-v1`)

`schema_version`, `decision_id` (sha256 of canonical inputs + probability, 16 hex),
`timestamp_utc` (ISO-8601), `inputs` (13 raw fields), `p_rto`, `recommended_action`,
`expected_loss` (4-action EL table), `shap_top_factors` (top-3), `latency_ms`,
`artifacts` (model / cost_engine / release paths).

Same inputs → same `decision_id` → judges can replay any line through `score_order()`
and reproduce the decision bit-for-bit. Session-scoped only: nothing is written to
the filesystem, so it is cloud-safe and restarts clean.

## How to apply

```bash
git apply 0001-jsonl-audit-trail.patch
pytest tests/ -q            # expect: 62 passed
streamlit run app.py        # optional: score an order, expand Audit Trail, download JSONL
```

Suggested commit message:

```
feat(demo): JSONL audit trail — one artifact-traced record per scoring decision

- src/serve/audit.py: pure record builder (schema rto-shield-audit-v1),
  deterministic decision_id = sha256(canonical inputs + p_rto)
- app.py: session-scoped accumulation + Audit Trail expander with
  Download (JSONL) / Clear; zero filesystem writes (cloud-safe)
- tests/test_audit.py: 5 tests incl. JSONL round-trip and PREPAID logging
- 57 existing tests untouched and passing (62 total)
```

## Suggested claim-matrix row

| Claim | Artifact | Command |
|-------|----------|---------|
| Every demo decision is exportable as a reproducible JSONL audit record | `src/serve/audit.py`, `tests/test_audit.py` | `pytest tests/test_audit.py -q` |

## Suggested JUDGE_QA.md entry

Q: "How do I verify a decision the demo shows me?"
A: Expand "Audit Trail" at the bottom, download the JSONL, take any line, and replay
its `inputs` through `score_order()` — the returned probability, action, expected-loss
table, and SHAP factors match the record exactly (enforced by `tests/test_audit.py`).
