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
    # Relative floor: model PR-AUC must be >= 90% of observable Bayes ceiling
    from src.eval.bayes_ceiling import get_true_p
    p_ceil_val = get_true_p(val_rep)
    ceil_pr_val = average_precision_score(y, p_ceil_val)
    ratio_val = pr_auc / ceil_pr_val
    assert ratio_val >= 0.90, f"Tree PR-AUC on val_rep fell below 90% of ceiling: {ratio_val:.3f}"

    cal_pipeline = joblib.load('models/tree_model_calibrated.pkl')
    proba_cal = cal_pipeline.predict_proba(X)[:, 1]
    pr_auc_cal = average_precision_score(y, proba_cal)
    ratio_val_cal = pr_auc_cal / ceil_pr_val
    assert ratio_val_cal >= 0.90, f"Calibrated Tree PR-AUC on val_rep fell below 90% of ceiling: {ratio_val_cal:.3f}"

    # Model degradation tests on held-out test set:
    # 1. Assert production model test PR-AUC >= 90% of observable Bayes ceiling
    # 2. Assert clear gap to a label-shuffled random baseline (>= 0.10)
    from src.eval.bayes_ceiling import get_true_p
    test_df = pd.read_csv('data/processed/test.csv', dtype={'pincode': str})
    X_test = test_df.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
    y_test = test_df['rto_label'].values
    p_test = cal_pipeline.predict_proba(X_test)[:, 1]
    pr_test = average_precision_score(y_test, p_test)

    p_ceiling = get_true_p(test_df)
    ceiling_pr = average_precision_score(y_test, p_ceiling)
    ratio_to_ceiling = pr_test / ceiling_pr
    assert ratio_to_ceiling >= 0.90, (
        f"Production model PR-AUC ratio to observable ceiling fell below 90%: "
        f"{pr_test:.4f} / {ceiling_pr:.4f} = {ratio_to_ceiling:.3f}"
    )

    rng = np.random.default_rng(42)
    y_shuffled = rng.permutation(y_test)
    pr_shuffled = average_precision_score(y_shuffled, p_test)
    gap = pr_test - pr_shuffled
    assert gap >= 0.10, (
        f"Production model gap to shuffled-label baseline is too small: {gap:.4f} (must be >= 0.10)"
    )

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

    # Byte-level integrity is enforced separately in
    # tests/test_artifact_integrity.py (SHA-256 pinned via
    # models/artifact_hashes.json). Previously this check asserted existence
    # only, while WHAT_BROKE.md Bug 4 claimed byte-level verification.
    manifest = os.path.join(os.path.dirname(__file__), '..', 'models', 'artifact_hashes.json')
    assert os.path.exists(manifest), (
        "models/artifact_hashes.json missing — run scripts/freeze_artifact_hashes.py"
    )

def test_tree_model_min_depth_guard():
    """Enforce minimum tree depth to prevent degenerate decision stumps (max_depth=1)."""
    import inspect
    from src.models import tree_model
    src = inspect.getsource(tree_model.main)
    assert "'max_depth': [1" not in src and "'max_depth': 1" not in src, (
        "Tree model grid illegally configured with degenerate max_depth=1"
    )
    booster = joblib.load('models/tree_model_booster.pkl')
    assert booster.max_depth >= 3, f"Shipped booster depth too shallow: {booster.max_depth}"

def test_calibrator_source_leakage_guard():
    """Ensure calibration never fits on test data (data leakage)."""
    cal_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'eval', 'calibration.py')
    with open(cal_path, 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'data/processed/test.csv' not in src, (
        "Calibration script illegally references test data (data leakage)"
    )

