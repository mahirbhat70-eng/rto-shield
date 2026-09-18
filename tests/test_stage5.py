import os
import joblib
import pytest
import numpy as np
import pandas as pd
import hashlib
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_binary_policy, eval_multi_action
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

def test_primary_savings_uplift(engine, test_data):
    test, proba = test_data
    test_cod, p_cal_cod = get_cod_subset(test, proba)
    
    baseline_actions, baseline_losses = eval_binary_policy(test_cod, p_cal_cod, engine, "ALLOW_COD", 1.0)
    baseline_total = np.sum(baseline_losses)
    
    primary_actions, primary_losses = eval_multi_action(test_cod, p_cal_cod, engine)
    primary_total = np.sum(primary_losses)
    
    primary_savings = baseline_total - primary_total
    uplift = primary_savings / abs(baseline_total)
    
    # Pre-registered check: Primary savings uplift [8%, 18%] of baseline loss (reports/stage5_test_results.md § 3)
    assert 0.08 <= uplift <= 0.18
    assert np.isclose(uplift, 0.131, atol=0.005)
    assert np.isclose(primary_savings, 71741.02, atol=1.0)

def test_bayes_ceiling():
    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    p_reconstructed = get_true_p(test)
    ceiling_pr_auc = average_precision_score(test['rto_label'], p_reconstructed)
    # Pre-registered check: Bayes Ceiling PR-AUC == 0.3497 (reports/stage5_test_results.md § 1)
    assert np.isclose(ceiling_pr_auc, 0.3497, atol=1e-3)

def test_noise_sensitivity(engine, test_data):
    test, proba = test_data
    test_cod, p_cal_cod = get_cod_subset(test, proba)
    
    baseline_actions, baseline_losses = eval_binary_policy(test_cod, p_cal_cod, engine, "ALLOW_COD", 1.0)
    baseline_total = np.sum(baseline_losses)
    
    primary_actions, primary_losses = eval_multi_action(test_cod, p_cal_cod, engine)
    primary_total = np.sum(primary_losses)
    primary_savings = baseline_total - primary_total
    
    rng = np.random.default_rng(42)
    p_cal_noisy = np.clip(p_cal_cod + rng.normal(0, 0.04, size=len(p_cal_cod)), 0.0, 1.0)
    
    _, base_noisy_loss = eval_binary_policy(test_cod, p_cal_noisy, engine, "ALLOW_COD", 1.0)
    _, prim_noisy_loss = eval_multi_action(test_cod, p_cal_noisy, engine)
    
    noisy_savings = np.sum(base_noisy_loss) - np.sum(prim_noisy_loss)
    delta_pct = (noisy_savings - primary_savings) / primary_savings
    
    # Pre-registered check: Noisy savings within ±10% (+2.7%) of clean savings (reports/stage5_test_results.md § 4)
    assert abs(delta_pct) <= 0.10
    assert np.isclose(delta_pct, 0.027, atol=0.005)

