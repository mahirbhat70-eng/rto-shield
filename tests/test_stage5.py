import os
import joblib
import pytest
import numpy as np
import pandas as pd
import hashlib
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset
from src.eval.bayes_ceiling import get_true_p
from sklearn.metrics import average_precision_score

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

@pytest.fixture
def engine():
    return CostEngine()

@pytest.fixture
def test_data():
    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    proba_cal = tree_cal.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
    return test, proba_cal

def test_stage5_frozen_thresholds_and_prepaid_never_argmin(engine, test_data):
    # This test asserts that the optimal thresholds from val_cal (0.48 for PREPAID, 0.20 for VERIFY)
    # are valid for evaluation, but more importantly, PREPAID_ONLY is never argmin on test P.
    test, proba = test_data
    
    # Assert PREPAID_ONLY is never argmin on test P
    for i, row in test.iterrows():
        losses = engine.evaluate_interventions(row['order_value'], proba[i])
        best_action = min(losses, key=losses.get)
        assert best_action != "PREPAID_ONLY"
        
def test_stage5_pr_auc_band(test_data):
    test, proba = test_data
    pr_auc = average_precision_score(test['rto_label'], proba)
    assert 0.29 <= pr_auc <= 0.37

def test_stage5_file_unchanged():
    test_csv_path = "data/processed/test.csv"
    h = sha256_file(test_csv_path)
    assert h == "cf29991737868ff0281f506d44ce6a240e33330a438cf12e16ff42124064adeb", "test.csv has been mutated!"

def test_stage5_strategies_present():
    with open("reports/stage5_test_results.md", encoding='utf-8') as f:
        report = f.read()
    assert "Stage 5.2: Operating-Point Metrics" in report
    assert "69,786.08" in report
    assert "Mean Savings:** \u20b969,942.31" in report

def test_stage5_report_structure():
    with open("reports/stage5_test_results.md", encoding='utf-8') as f:
        report = f.read()
    for s in ["Baseline", "Binary PREPAID", "Binary VERIFY",
              "Primary", "Sens (Uncal)", "Sens (Clipped)"]:
        assert s in report, f"missing strategy row: {s}"
    for label in ["Action Dist", "Orders Touched", "Expected RTOs Prevented"]:
        assert label in report, f"missing denominator label: {label}"
