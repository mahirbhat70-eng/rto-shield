import os
import joblib
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import eval_multi_action, get_cod_subset, eval_binary_policy

@pytest.fixture
def engine():
    return CostEngine()

def test_stage4_expected_loss_allow_cod(engine):
    order_value = 1000.0
    p_rto = 1.0
    
    losses = engine.evaluate_interventions(order_value, p_rto)
    assert np.allclose(losses['ALLOW_COD'], engine.rto_logistics_cost, rtol=1e-9)
    
def test_stage4_expected_loss_prepaid_only(engine):
    order_value = 1000.0
    p_rto = 0.0
    
    losses = engine.evaluate_interventions(order_value, p_rto)
    expected = -(order_value * engine.average_margin_pct * (1.0 - 0.70))
    assert np.allclose(losses['PREPAID_ONLY'], expected, rtol=1e-9)

def test_stage4_multi_action_vs_verify_block(engine):
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    proba = tree_cal.predict_proba(val_rep.drop(columns=['rto_label']))[:, 1]
    
    val_rep_cod, proba_cod = get_cod_subset(val_rep, proba)
    
    # We use t=0.0 to 1.0, the test doesn't specify a specific t, just "Binary VERIFY block total loss".
    # Since multi-action evaluates all options including VERIFY_ADDRESS and ALLOW_COD,
    # its total loss must be <= ANY binary policy's total loss (since it takes the min point-wise).
    # We can just pick an arbitrary threshold or the optimized one. We'll use 0.5.
    _, losses_multi = eval_multi_action(val_rep_cod, proba_cod, engine)
    _, losses_verify = eval_binary_policy(val_rep_cod, proba_cod, engine, "VERIFY_ADDRESS", 0.5)
    
    total_multi = np.sum(losses_multi)
    total_verify = np.sum(losses_verify)
    
    assert total_multi <= total_verify + 1e-6

def test_stage4_closed_form_baseline(engine):
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    proba = tree_cal.predict_proba(val_rep.drop(columns=['rto_label']))[:, 1]
    
    val_rep_cod, proba_cod = get_cod_subset(val_rep, proba)
    
    _, losses_allow = eval_binary_policy(val_rep_cod, proba_cod, engine, "ALLOW_COD", 1.0)
    total_allow = np.sum(losses_allow)
    
    closed_form_total = 0.0
    for i, row in val_rep_cod.iterrows():
        p_i = proba_cod[i]
        margin_i = row['order_value'] * engine.average_margin_pct
        expected = p_i * engine.rto_logistics_cost - (1 - p_i) * margin_i
        closed_form_total += expected
        
    assert np.allclose(total_allow, closed_form_total, rtol=1e-9)

def test_stage4_prepaid_never_argmin(engine):
    margins = [100.0, 200.0, 500.0, 1000.0]
    p_rtos = np.arange(0.0, 1.01, 0.01)
    
    # We construct a mock order_value by dividing margin by average_margin_pct
    
    for margin in margins:
        order_value = margin / engine.average_margin_pct
        for p in p_rtos:
            losses = engine.evaluate_interventions(order_value, p)
            best_action = min(losses, key=losses.get)
            assert best_action != "PREPAID_ONLY", f"PREPAID_ONLY was argmin at P={p}, margin={margin}"
