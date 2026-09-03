import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.serve.scorer import score_order, route_order, validate_features, REQUIRED_INPUTS

VALID_PAYLOAD = {
    'order_value': 1000,
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
    'device_cluster_size': 1
}

def assert_valid_result(res):
    assert 0 <= res['probability'] <= 1
    assert res['recommended_action'] in ["ALLOW_COD", "VERIFY_ADDRESS", "REQUIRE_DEPOSIT", "PREPAID_ONLY", "PREPAID_PASSTHROUGH"]
    for k, v in res['el_table'].items():
        assert np.isfinite(v)

import yaml
cfg = yaml.safe_load(open("configs/cost_config.yaml"))
INTS = cfg['interventions']
LOG_C = cfg['rto_logistics_cost']
M_PCT = cfg['average_margin_pct']

def el_from_config(P, V, name):
    i = INTS[name]
    return (i['friction_cost']
            + P * (1 - i['rto_reduction_pct']) * LOG_C
            - (1 - P) * (1 - i['success_drop_pct']) * V * M_PCT)

def test_unknown_category_courier():
    payload = VALID_PAYLOAD.copy()
    payload['category'] = 'UNKNOWN_CAT'
    payload['courier_id'] = 'UNKNOWN_COURIER'
    res = score_order(payload)
    assert_valid_result(res)

def test_missing_pincode_in_lookup():
    val = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    unseen = str(int(val['pincode'].max()) + 1)
    
    payload = VALID_PAYLOAD.copy()
    payload['pincode'] = unseen
    with pytest.raises(ValueError, match="not found in lookup table"):
        score_order(payload)

def test_missing_required_fields():
    for field in REQUIRED_INPUTS:
        payload = VALID_PAYLOAD.copy()
        del payload[field]
        with pytest.raises(ValueError, match=field):
            score_order(payload)

def test_nan_numeric_features():
    numeric_fields = ['order_value', 'quantity', 'discount_pct', 'cod_charge', 
                      'account_age_days', 'prior_orders', 'prior_rto_count', 
                      'orders_last_24h', 'device_cluster_size']
    for field in numeric_fields:
        payload = VALID_PAYLOAD.copy()
        payload[field] = np.nan
        with pytest.raises(ValueError, match=field):
            score_order(payload)

def test_extreme_values_rejected():
    for field, val in [('order_value', 1e8), ('device_cluster_size', 1e6),
                       ('order_value', 25001)]:
        payload = VALID_PAYLOAD.copy()
        payload[field] = val
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            score_order(payload)

def test_boundary_values_accepted():
    for field, val in [('order_value', 0), ('order_value', 25000),
                       ('device_cluster_size', 50), ('discount_pct', 100)]:
        payload = VALID_PAYLOAD.copy()
        payload[field] = val
        assert_valid_result(score_order(payload))

def test_logical_constraints():
    payload = VALID_PAYLOAD.copy()
    payload['prior_rto_count'] = 5
    payload['prior_orders'] = 3
    with pytest.raises(ValueError, match="prior_rto_count cannot be greater than prior_orders"):
        score_order(payload)

    payload = VALID_PAYLOAD.copy()
    payload['account_age_days'] = -5
    with pytest.raises(ValueError, match="account_age_days cannot be negative"):
        score_order(payload)

def test_duplicate_order_id():
    payload = VALID_PAYLOAD.copy()
    res1 = score_order(payload)
    res2 = score_order(payload)
    assert res1['probability'] == res2['probability']
    assert_valid_result(res2)

def test_prepaid_payment_method():
    payload = VALID_PAYLOAD.copy()
    payload['payment_method'] = 'PREPAID'
    res = score_order(payload)
    assert res['recommended_action'] == 'PREPAID_PASSTHROUGH'
    assert 0 <= res['probability'] <= 1
    assert all(v == 0.0 for v in res['el_table'].values())

def test_consistency():
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    sample = val_rep[val_rep['payment_method'] == 'COD'].sample(100, random_state=42)
    
    for _, row in sample.iterrows():
        features = row.to_dict()
        features.pop('historical_pincode_rto_rate', None)
        features.pop('pincode_tier', None)
        res = score_order(features)
        
        P = res['probability']
        order_value = features['order_value']
        margin = order_value * 0.20
        
        expected_el = {name: el_from_config(P, order_value, name) for name in INTS}
        
        argmin_action = min(expected_el, key=expected_el.get)
        assert res['recommended_action'] == argmin_action
        assert np.isclose(res['el_table']['ALLOW_COD'], expected_el['ALLOW_COD'], rtol=1e-9)

def test_explanation_integrity():
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    # Directions from data dictionary
    # COD +, historical_pincode_rto_rate +, tier 3 +, prior_rto_count +, category Apparel +
    # account_age_days -, prior_orders -, tier 1 -
    
    # Let's check some high risk rows
    # Take a large sample so we ensure we hit enough high P rows
    sample = val_rep.copy()
    high_risk_tested = 0
    
    for _, row in sample.iterrows():
        features = row.to_dict()
        features.pop('historical_pincode_rto_rate', None)
        features.pop('pincode_tier', None)
        if features['payment_method'] != 'COD':
            continue
        try:
            res = score_order(features)
        except ValueError:
            continue
            
        if res['probability'] > 0.5:
            top_feature, top_shap = res['shap_top_factors'][0]
            # Verify direction
            if 'num__prior_rto_count' in top_feature:
                assert top_shap > 0, f"prior_rto_count should drive risk UP, got {top_shap}"
            if 'num__account_age_days' in top_feature:
                if features['account_age_days'] > 100: # old accounts reduce risk
                    assert top_shap < 0, f"High account age should drive risk DOWN"
            
            high_risk_tested += 1
            if high_risk_tested >= 20:
                break
                
    assert high_risk_tested >= 10, f"Only {high_risk_tested} high-risk rows found — explanation coverage insufficient"
                
def test_shap_uses_class1_contributions():
    # For P > 0.5, total SHAP on the log-odds scale must be positive.
    # Catches the class-0 vs class-1 index inversion.
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    cod = val_rep[val_rep['payment_method'] == 'COD']
    
    res = None
    for _, row in cod.iterrows():
        features = row.to_dict()
        features.pop('historical_pincode_rto_rate', None)
        features.pop('pincode_tier', None)
        try:
            curr_res = score_order(features)
        except ValueError:
            continue
            
        if curr_res['probability'] > 0.5:
            res = curr_res
            break
            
    assert res is not None, "no high-risk row found in dataset"
    total = sum(v for _, v in res['shap_top_factors'][:3])
    print(f"\nSHAP Class 1 check: P={res['probability']:.4f}, top-3 sum={total:.4f}")
    assert total > 0, "SHAP values appear inverted (using class 0)"

def test_feature_completeness_regression():
    payload = {
        'order_value': 2400, 
        'quantity': 2, 
        'category': 'electronics', 
        'payment_method': 'COD', 
        'prior_orders': 5, 
        'prior_rto_count': 2, 
        'pincode_tier': 3
    }
    with pytest.raises(ValueError) as exc:
        validate_features(payload)
    
    msg = str(exc.value)
    assert 'Missing required fields' in msg
    assert 'pincode' in msg
    assert 'account_age_days' in msg
    assert 'device_cluster_size' in msg
    assert 'orders_last_24h' in msg
    assert 'discount_pct' in msg
    assert 'cod_charge' in msg
    assert 'Do not provide historical_pincode_rto_rate or pincode_tier' in msg
