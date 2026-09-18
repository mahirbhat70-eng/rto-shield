import pytest
import pandas as pd
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.serve.scorer import score_order, tree_cal, engine, PINCODE_LOOKUP

def test_score_order_matches_pipeline():
    # 1. score_order on 5 known val_rep COD rows == pipeline predict_proba (1e-9).
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    sample = val_rep[val_rep['payment_method'] == 'COD'].sample(5, random_state=42)
    
    for i, row in sample.iterrows():
        # Overwrite with lookup values for exact comparison
        pincode = str(row['pincode'])
        lookup_info = PINCODE_LOOKUP[pincode]
        row['historical_pincode_rto_rate'] = float(lookup_info['historical_pincode_rto_rate'])
        row['pincode_tier'] = lookup_info['pincode_tier']
        
        # Get raw pipeline probability
        df_row = pd.DataFrame([row])
        expected_p = tree_cal.predict_proba(df_row)[0, 1]
        
        # Get score_order probability
        features = row.to_dict()
        features.pop('historical_pincode_rto_rate', None)
        features.pop('pincode_tier', None)
        res = score_order(features)
        
        # Assert probabilities match to 1e-9
        assert np.isclose(res['probability'], expected_p, atol=1e-9)

def test_recommended_action_matches_cost_engine():
    # 2. recommended_action == cost_engine argmin on same inputs
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    sample = val_rep[val_rep['payment_method'] == 'COD'].sample(5, random_state=42)
    
    for i, row in sample.iterrows():
        features = row.to_dict()
        features.pop('historical_pincode_rto_rate', None)
        features.pop('pincode_tier', None)
        res = score_order(features)
        
        # Re-run cost engine directly
        order_value = features['order_value']
        p = res['probability']
        el_table = engine.evaluate_interventions(order_value, p)
        expected_action = min(el_table, key=el_table.get)
        
        assert res['recommended_action'] == expected_action
        
def test_unknown_pincode_cold_start():
    # 3. Unknown pincodes fall back to the global prior (matching docs/JUDGE_QA.md Q7).
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    unseen = str(int(val_rep['pincode'].max()) + 1)
    
    features = val_rep.iloc[0].to_dict()
    features.pop('historical_pincode_rto_rate', None)
    features.pop('pincode_tier', None)
    features['pincode'] = unseen
    
    res = score_order(features)
    assert res['recommended_action'] in ('ALLOW_COD', 'VERIFY_ADDRESS', 'REQUIRE_DEPOSIT')
    assert any('cold start' in w.lower() for w in res['warnings'])

def test_full_shap_cold_start():
    # 3b. Regression test: dashboard's full_shap must not crash on unseen pincodes
    import dashboard
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    unseen = str(int(val_rep['pincode'].max()) + 1)
    
    payload = val_rep.iloc[0].to_dict()
    payload['pincode'] = unseen
    payload['payment_method'] = 'COD'
    
    # This should not raise a KeyError
    pairs = dashboard.full_shap(payload)
    assert isinstance(pairs, list)
    assert len(pairs) <= 8

import os

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Latency tests can be flaky on CI runners")
def test_latency_under_100ms():
    # 4. Single score_order call < 100ms
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    features = val_rep[val_rep['payment_method'] == 'COD'].iloc[0].to_dict()
    features.pop('historical_pincode_rto_rate', None)
    features.pop('pincode_tier', None)
    
    # Warmup
    _ = score_order(features)
    
    start = time.time()
    res = score_order(features)
    elapsed = time.time() - start
    
    # Assert elapsed < 100ms
    assert elapsed < 0.100, f"score_order took {elapsed*1000:.1f}ms, expected < 100ms"

def test_payload_from_state_prepaid_zero_cod_charge():
    import streamlit as st
    import dashboard
    st.session_state[dashboard.WIDGET_KEYS["payment_method"]] = "PREPAID"
    st.session_state.pop(dashboard.WIDGET_KEYS["cod_charge"], None)
    payload = dashboard.payload_from_state()
    assert payload["payment_method"] == "PREPAID"
    assert payload["cod_charge"] == 0.0

def test_payload_from_state_cod_charge():
    import streamlit as st
    import dashboard
    st.session_state[dashboard.WIDGET_KEYS["payment_method"]] = "COD"
    st.session_state[dashboard.WIDGET_KEYS["cod_charge"]] = 42.0
    payload = dashboard.payload_from_state()
    assert payload["payment_method"] == "COD"
    assert payload["cod_charge"] == 42.0

def test_shap_html_empty_pairs():
    import dashboard
    out = dashboard.shap_html([])
    assert isinstance(out, str)
    assert "passthrough" in out.lower()

def test_prepaid_scoring_and_shap_render():
    import dashboard
    from src.serve.scorer import score_order
    payload = {
        "payment_method": "PREPAID",
        "order_value": 1200.0,
        "category": "Apparel",
        "quantity": 2,
        "discount_pct": 10.0,
        "cod_charge": 0.0,
        "account_age_days": 120,
        "prior_orders": 3,
        "prior_rto_count": 0,
        "orders_last_24h": 1,
        "device_cluster_size": 1,
        "pincode": "253407",
        "courier_id": "Courier_E"
    }
    res = score_order(payload)
    assert res["recommended_action"] == "PREPAID_PASSTHROUGH"
    assert all(v == 0.0 for v in res["el_table"].values())
    assert res["shap_top_factors"] == []
    assert any("prepaid" in w.lower() for w in res["warnings"])
    
    # Render shap_html with empty pairs (as done for PREPAID orders in dashboard.py)
    pairs = [] if payload["payment_method"] != "COD" else dashboard.full_shap(payload)
    html_out = dashboard.shap_html(pairs)
    assert "passthrough" in html_out.lower()

def test_clamp_prior_rto():
    import streamlit as st
    import dashboard
    st.session_state["ni_prior_orders"] = 1
    st.session_state["prior_rto_count"] = 3
    dashboard._clamp_prior_rto()
    assert st.session_state["prior_rto_count"] == 1

