import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

def test_stage3_boundary_check():
    val_cal = pd.read_csv("data/processed/val_cal.csv")
    val_rep = pd.read_csv("data/processed/val_rep.csv")
    
    cal_max = pd.to_datetime(val_cal['timestamp']).max()
    rep_min = pd.to_datetime(val_rep['timestamp']).min()
    assert cal_max < rep_min, "val_cal max timestamp is not strictly before val_rep min timestamp"

def test_stage3_inference_check():
    pipeline = joblib.load('models/tree_model.pkl')
    raw_val = pd.read_csv('data/raw/synthetic_orders.csv', dtype={'pincode': str})
    # Filter to get first 100 of val set equivalent
    sample = pd.read_csv('data/processed/val.csv', dtype={'pincode': str}).head(100)
    
    X = sample.drop(columns=['rto_label'])
    proba = pipeline.predict_proba(X)
    
    assert proba.shape == (100, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    
def test_stage3_performance_floor():
    pipeline = joblib.load('models/tree_model.pkl')
    val_rep = pd.read_csv('data/processed/val_rep.csv', dtype={'pincode': str})
    
    X = val_rep.drop(columns=['rto_label'])
    y = val_rep['rto_label'].values
    
    proba = pipeline.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, proba)
    
    assert pr_auc >= 0.23, f"Tree PR-AUC on val_rep fell below floor: {pr_auc}"

def test_stage3_noise_check():
    # From our printed SHAP values:
    # 1. cod_charge
    # 2. historical_pincode_rto_rate
    # 3. pincode_tier
    # 4. prior_rto_count
    # 5. category
    # Ensure quantity or device_cluster_size is not in top 5.
    # In explainability.py we output the top features, but we can't easily assert from the console.
    # We will compute a quick SHAP on a small sample to assert.
    
    booster = joblib.load('models/tree_model_booster.pkl')
    pipeline = joblib.load('models/tree_model.pkl')
    preprocessor = pipeline.named_steps['preprocessor']
    
    val_rep = pd.read_csv('data/processed/val_rep.csv', dtype={'pincode': str})
    X_raw = val_rep.drop(columns=['rto_label']).head(50)
    X_transformed = preprocessor.transform(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    
    import shap
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_transformed)
    
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values
        
    parent_features = {}
    for i, col in enumerate(feature_names):
        if col.startswith('num__'):
            parent = col[5:]
        elif col.startswith('cat__'):
            parent = col[5:].split('_')[0]
            if col[5:].startswith('payment_method'):
                parent = 'payment_method'
            elif col[5:].startswith('courier_id'):
                parent = 'courier_id'
            elif col[5:].startswith('pincode_tier'):
                parent = 'pincode_tier'
        else:
            parent = col
            
        mean_abs_shap = np.mean(np.abs(shap_values_pos[:, i]))
        if parent in parent_features:
            parent_features[parent] += mean_abs_shap
        else:
            parent_features[parent] = mean_abs_shap
            
    sorted_parents = [k for k, v in sorted(parent_features.items(), key=lambda x: x[1], reverse=True)]
    top_5 = sorted_parents[:5]
    
    assert 'quantity' not in top_5, "quantity unexpectedly in top 5 SHAP features"
    assert 'device_cluster_size' not in top_5, "device_cluster_size unexpectedly in top 5 SHAP features"

def test_stage3_artifact_check():
    artifacts = [
        'reports/stage3/shap_summary.png',
        'reports/stage3/shap_waterfall_high.png',
        'reports/stage3/shap_waterfall_low.png',
        'reports/stage3/calibration_curve.png',
        'reports/stage3/stage3_results.md',
        'models/tree_model.pkl',
        'models/tree_model_booster.pkl',
        'models/tree_model_calibrated.pkl'
    ]
    for artifact in artifacts:
        assert os.path.exists(artifact), f"Artifact missing: {artifact}"
