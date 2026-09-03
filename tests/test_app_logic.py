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
        
def test_unknown_pincode_error():
    # 3. Unknown pincode error renders (catch in test).
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    unseen = str(int(val_rep['pincode'].max()) + 1)
    
    features = val_rep.iloc[0].to_dict()
    features.pop('historical_pincode_rto_rate', None)
    features.pop('pincode_tier', None)
    features['pincode'] = unseen
    
    with pytest.raises(ValueError) as exc:
        score_order(features)
        
    assert "not found in lookup table" in str(exc.value)

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
