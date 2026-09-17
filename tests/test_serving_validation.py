"""
test_serving_validation.py — Edge-case coverage for the hardened serving path.

Covers the bugs fixed in the improvements branch (see IMPROVEMENTS.md):
  * non-COD payments (UPI/CC/DC/NB) must be passthroughs — previously they
    fell into the COD expected-loss table (Critical bug)
  * payment/category/courier canonicalization (case-insensitive)
  * coerce-then-check numeric validation (strings, bools, inf, nan, negatives)
  * pincode normalization + cold-start fallback
  * cod_charge cross-field contract
  * vectorized batch API: alignment guard, filtered-df safety, single==batch
    equivalence, vectorized EL == scalar EL
  * audit: model_version stamping, replay-stable decision_id
"""

import datetime
import numpy as np
import pandas as pd
import pytest
import joblib

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.serve.scorer import (
    score_order, route_order, validate_features, validate_and_clean,
    resolve_pincode_info, MODEL_VERSION, REQUIRED_INPUTS,
)
from src.serve.audit import build_audit_record, verify_audit_record
from src.policy.cost_engine import CostEngine, is_cod

VALID_PAYLOAD = {
    'order_value': 1000.0,
    'quantity': 1,
    'category': 'Electronics',
    'discount_pct': 0,
    'payment_method': 'COD',
    'cod_charge': 50,
    'account_age_days': 100,
    'prior_orders': 2,
    'prior_rto_count': 0,
    'pincode': '597542',
    'courier_id': 'Courier_A',
    'orders_last_24h': 0,
    'device_cluster_size': 1,
}


# ─── A. Payment semantics: one predicate, everywhere ─────────────────────

@pytest.mark.parametrize("method", ["UPI", "Credit Card", "Debit Card", "Net Banking", "PREPAID",
                                    "prepaid", "Prepaid", "PREPAID "])
def test_non_cod_payments_are_passthrough(method):
    payload = dict(VALID_PAYLOAD, payment_method=method, cod_charge=0)
    res = score_order(payload)
    assert res['recommended_action'] == 'PREPAID_PASSTHROUGH'
    assert all(v == 0.0 for v in res['el_table'].values())
    assert res['shap_top_factors'] == []  # SHAP skipped for passthrough
    assert 0.0 <= res['probability'] <= 1.0


@pytest.mark.parametrize("variant", ["COD", "cod", "Cod", "COD ", " cod"])
def test_cod_variants_are_cod(variant):
    payload = dict(VALID_PAYLOAD, payment_method=variant)
    res = score_order(payload)
    assert res['recommended_action'] != 'PREPAID_PASSTHROUGH'


def test_is_cod_matches_batch_and_single_paths():
    # The batch router and single scorer must agree on every payment value.
    df = pd.DataFrame({'payment_method': ['COD', 'UPI', 'cod', None, 'PREPAID', 'Net Banking']})
    batch = df['payment_method'].map(is_cod).tolist()
    assert batch == [True, False, True, False, False, False]


def test_caseload_canonicalization():
    payload = dict(VALID_PAYLOAD, category='electronics', courier_id='courier_a')
    res = score_order(payload)
    clean, _ = validate_and_clean(payload)
    assert clean['category'] == 'Electronics'
    assert clean['courier_id'] == 'Courier_A'
    assert any('Unknown' in w for w in res['warnings']) is False


def test_unknown_categorical_warns_but_scores():
    payload = dict(VALID_PAYLOAD, category='UNKNOWN_CAT', courier_id='UNKNOWN_COURIER')
    res = score_order(payload)
    assert 0 <= res['probability'] <= 1
    assert len(res['warnings']) == 2  # OOD warning for both fields


def test_none_categorical_rejected():
    payload = dict(VALID_PAYLOAD, category=None)
    with pytest.raises(ValueError, match="category"):
        score_order(payload)


# ─── B. Numeric hardening: coerce-then-check ─────────────────────────────

@pytest.mark.parametrize("field,value", [
    ('order_value', -500.0), ('quantity', -3), ('discount_pct', -20),
    ('cod_charge', -100), ('prior_orders', -5), ('prior_rto_count', -2),
    ('orders_last_24h', -10), ('device_cluster_size', -7),
])
def test_negative_values_rejected(field, value):
    payload = dict(VALID_PAYLOAD)
    payload[field] = value
    with pytest.raises(ValueError, match="below the minimum allowed value"):
        score_order(payload)


def test_string_numerics_coerced_not_crashed():
    payload = dict(VALID_PAYLOAD, order_value='826.89', quantity='2')
    res = score_order(payload)
    assert 0 <= res['probability'] <= 1  # no TypeError crash (old bug)


@pytest.mark.parametrize("bad", [float('inf'), float('-inf'), float('nan'), '1e400', 'abc', [1000], {'a': 1}])
def test_nonfinite_and_nonnumeric_rejected(bad):
    payload = dict(VALID_PAYLOAD, order_value=bad)
    with pytest.raises(ValueError):
        score_order(payload)


def test_bool_rejected():
    payload = dict(VALID_PAYLOAD, order_value=True)
    with pytest.raises(ValueError, match="got bool"):
        score_order(payload)


def test_negative_order_value_cannot_make_prepaid_win():
    # Old bug: negative V flipped the dead PREPAID_ONLY action to argmin.
    payload = dict(VALID_PAYLOAD, order_value=-500.0)
    with pytest.raises(ValueError):
        score_order(payload)


# ─── C. Pincode handling ─────────────────────────────────────────────────

@pytest.mark.parametrize("pin", ['597542 ', ' 597542', 597542, 597542.0])
def test_pincode_normalized(pin):
    payload = dict(VALID_PAYLOAD, pincode=pin)
    res = score_order(payload)
    assert res['probability'] > 0
    assert res['recommended_action'] != 'PREPAID_PASSTHROUGH'


@pytest.mark.parametrize("pin", ['59754', '5975423', '59-7542', 'ABCDE6'])
def test_malformed_pincode_rejected(pin):
    payload = dict(VALID_PAYLOAD, pincode=pin)
    with pytest.raises(ValueError, match="not a valid 6-digit code"):
        score_order(payload)


def test_cold_start_pincode_falls_back_to_global_prior():
    payload = dict(VALID_PAYLOAD, pincode='999999')
    res = score_order(payload)
    assert 0 <= res['probability'] <= 1
    assert any('cold start' in w.lower() for w in res['warnings'])
    info, is_cold = resolve_pincode_info('999999')
    assert is_cold
    assert 0 <= info['historical_pincode_rto_rate'] <= 1


def test_resolve_pincode_info_warm_path():
    info, is_cold = resolve_pincode_info('597542')
    assert not is_cold
    assert 'historical_pincode_rto_rate' in info


# ─── D. Cross-field contracts ────────────────────────────────────────────

def test_cod_charge_on_non_cod_rejected():
    payload = dict(VALID_PAYLOAD, payment_method='PREPAID', cod_charge=50)
    with pytest.raises(ValueError, match="cod_charge must be 0 for non-COD"):
        score_order(payload)


def test_new_customer_contract_warns():
    payload = dict(VALID_PAYLOAD, account_age_days=0, prior_orders=3)
    res = score_order(payload)
    assert any('new customers' in w for w in res['warnings'])


# ─── E. Result contract ──────────────────────────────────────────────────

def test_result_carries_model_version_and_warnings():
    res = score_order(dict(VALID_PAYLOAD))
    assert res['model_version'] == MODEL_VERSION
    assert len(MODEL_VERSION) == 12
    assert isinstance(res['warnings'], list)


def test_duplicate_scoring_is_deterministic():
    r1 = score_order(dict(VALID_PAYLOAD))
    r2 = score_order(dict(VALID_PAYLOAD))
    assert r1['probability'] == r2['probability']
    assert r1['recommended_action'] == r2['recommended_action']


# ─── F. Batch API: vectorized, aligned, equivalent ───────────────────────

@pytest.fixture(scope="module")
def batch_fixture():
    df = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    sub = df[df['payment_method'] == 'COD'].iloc[5:45]  # non-zero, non-contiguous labels
    model = joblib.load('models/tree_model_calibrated.pkl')
    proba = pd.Series(model.predict_proba(sub.drop(columns=['rto_label']))[:, 1], index=sub.index)
    return sub, proba


def test_batch_policy_on_filtered_df(batch_fixture):
    # Old bug: positional iloc with label-indexed series mispaired/crashed.
    sub, proba = batch_fixture
    engine = CostEngine()
    actions, losses = engine.get_optimal_policy(sub, proba)
    assert len(actions) == len(sub)
    assert set(actions) <= {'ALLOW_COD', 'VERIFY_ADDRESS', 'REQUIRE_DEPOSIT', 'PREPAID_ONLY'}


def test_batch_policy_length_guard(batch_fixture):
    sub, proba = batch_fixture
    engine = CostEngine()
    with pytest.raises(ValueError, match="row-aligned"):
        engine.get_optimal_policy(sub, proba.iloc[:-3])


def test_batch_and_single_agree(batch_fixture):
    sub, proba = batch_fixture
    engine = CostEngine()
    actions, losses = engine.get_optimal_policy(sub, proba)
    row = sub.iloc[0]
    feats = row.to_dict()
    feats.pop('rto_label'); feats.pop('historical_pincode_rto_rate'); feats.pop('pincode_tier')
    res = score_order(feats)
    assert res['recommended_action'] == actions[0]


def test_batch_non_cod_passthrough():
    engine = CostEngine()
    df = pd.DataFrame({
        'payment_method': ['COD', 'UPI', 'Credit Card'],
        'order_value': [500.0, 800.0, 1200.0],
    })
    actions, losses = engine.get_optimal_policy(df, [0.5, 0.9, 0.9])
    assert actions[1] == 'PREPAID_PASSTHROUGH' and losses[1] == 0.0
    assert actions[2] == 'PREPAID_PASSTHROUGH' and losses[2] == 0.0
    assert actions[0] in ('VERIFY_ADDRESS', 'REQUIRE_DEPOSIT', 'ALLOW_COD')


def test_vectorized_el_equals_scalar_el():
    engine = CostEngine()
    V = [186.0, 826.89, 2500.0, 607.57]
    P = [0.31, 0.12, 0.85, 0.50]
    names, mat = engine.evaluate_interventions_vectorized(V, P)
    for j, (v, p) in enumerate(zip(V, P)):
        scalar = engine.evaluate_interventions(v, p)
        for i, name in enumerate(names):
            assert np.isclose(mat[i, j], scalar[name], rtol=1e-12)


def test_batch_policy_is_faster_than_iterrows_equivalent(batch_fixture):
    # Guard against future regressions to iterrows: 40 rows must route in < 50 ms.
    import time
    sub, proba = batch_fixture
    engine = CostEngine()
    t0 = time.perf_counter()
    engine.get_optimal_policy(sub, proba)
    assert (time.perf_counter() - t0) < 0.05


# ─── G. Audit: model-pinned, replay-stable ───────────────────────────────

def test_audit_decision_id_replay_stable():
    feats = dict(VALID_PAYLOAD)
    res = score_order(feats)
    r1 = build_audit_record(feats, res, 10.0, timestamp=datetime.datetime(2026, 1, 1))
    r2 = build_audit_record(feats, res, 12.0, timestamp=datetime.datetime(2026, 9, 6))
    assert r1['decision_id'] == r2['decision_id']  # same payload + same model
    assert r1['timestamp'] != r2['timestamp']
    assert verify_audit_record(r1) and verify_audit_record(r2)


def test_audit_model_version_swap_detected():
    feats = dict(VALID_PAYLOAD)
    res = score_order(feats)
    record = build_audit_record(feats, res, 10.0)
    record = dict(record)
    record['model_version'] = 'deadbeef1234'  # forged provenance
    assert not verify_audit_record(record)


def test_route_order_still_works():
    assert route_order(dict(VALID_PAYLOAD)) in (
        'ALLOW_COD', 'VERIFY_ADDRESS', 'REQUIRE_DEPOSIT', 'PREPAID_ONLY')
