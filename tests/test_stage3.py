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
    """Behavioural guard: on data with non-linear feature interactions (Generator v3),
    a depth-1 decision stump model fails a relative PR-AUC floor (< 95% of depth-4 ceiling),
    proving that non-degenerate depth >= 2 is strictly required to capture feature interactions."""
    from src.data.generator import generate
    from lightgbm import LGBMClassifier
    df_v3 = generate(n_rows=20000, seed=42, generator_version="v3")
    tr = df_v3.iloc[:14000].reset_index(drop=True)
    te = df_v3.iloc[14000:].reset_index(drop=True)
    feats = ['order_value', 'discount_pct', 'pincode_tier', 'historical_pincode_rto_rate', 'orders_last_24h', 'device_cluster_size', 'prior_rto_count']
    X_tr, y_tr = tr[feats], tr['rto_label'].values
    X_te, y_te = te[feats], te['rto_label'].values

    # Train model using current PARAM_GRID configuration min depth
    from src.models import tree_model
    min_depth = min(tree_model.PARAM_GRID.get('max_depth', [4]))
    model = LGBMClassifier(max_depth=min_depth, n_estimators=50, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)
    pr_actual = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])

    # Reference ceiling (depth 4)
    ref = LGBMClassifier(max_depth=4, n_estimators=50, random_state=42, verbose=-1).fit(X_tr, y_tr)
    pr_ceil = average_precision_score(y_te, ref.predict_proba(X_te)[:, 1])

    ratio = pr_actual / pr_ceil
    assert ratio >= 0.95, (
        f"Model with min_depth={min_depth} achieves only {ratio:.1%} of interaction ceiling ({pr_actual:.4f} vs {pr_ceil:.4f}); "
        "fails required >= 95% interaction floor. Depth-1 stumps cannot capture non-linear interactions."
    )


def test_calibrator_source_leakage_guard():
    """Behavioural provenance guard: fitting the calibrator on held-out test data
    (data leakage) produces predictions that diverge detectably from authentic
    validation calibration and trips out-of-sample probability checks."""
    from src.eval import calibration
    uncal = joblib.load('models/tree_model.pkl')
    shipped = joblib.load('models/tree_model_calibrated.pkl')
    test_df = pd.read_csv('data/processed/test.csv', dtype={'pincode': str})
    X = test_df.drop(columns=['rto_label'])
    p_shipped = shipped.predict_proba(X)[:, 1]

    # Re-running the pipeline's calibrator fit must reproduce authentic validation calibration
    p_refit = calibration.fit_calibrator(uncal).predict_proba(X)[:, 1]
    assert np.allclose(p_refit, p_shipped, atol=1e-9), (
        f"Calibrator data provenance mismatch: max diff {np.abs(p_refit - p_shipped).max():.4g}. "
        "Calibrator was not fitted on authentic validation split (data leakage detected)."
    )


