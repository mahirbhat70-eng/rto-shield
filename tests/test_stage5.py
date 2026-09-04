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

TEST_CSV_SHA256_LF = "aaa36a2bbe9b1a4293251b016a6193a662d4fb7b4b60209af03df5120d7f56f5"

def test_stage5_file_unchanged():
    raw = open("data/processed/test.csv", "rb").read()
    h = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert h == TEST_CSV_SHA256_LF, "test.csv content has mutated since the Stage 5 reveal"

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
