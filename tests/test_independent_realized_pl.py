"""
tests/test_independent_realized_pl.py
Verifies realized P&L directly from raw predictions and actual labels.
"""

import pytest
import numpy as np
import pandas as pd
import joblib

def compute_realized_savings(test_df, proba, rto_cost=150.0, margin_pct=0.20, verify_cost=2.0):
    cod_mask = (test_df['payment_method'] == 'COD').values
    cod_df = test_df[cod_mask].copy().reset_index(drop=True)
    p_cod = proba[cod_mask]
    y_actual = cod_df['rto_label'].values
    V = cod_df['order_value'].values
    margin = V * margin_pct

    # Baseline: Always Allow COD
    base_losses = np.where(y_actual == 1, rto_cost, -margin)
    total_baseline_loss = np.sum(base_losses)

    # Multi-Action: Pointwise minimum expected loss
    el_allow = p_cod * rto_cost - (1.0 - p_cod) * margin
    el_verify = verify_cost + (p_cod * (1.0 - 0.30)) * rto_cost - (1.0 - p_cod) * (1.0 - 0.05) * margin
    el_deposit = (p_cod * (1.0 - 0.80)) * rto_cost - (1.0 - p_cod) * (1.0 - 0.40) * margin
    el_prepaid = (p_cod * (1.0 - 0.55)) * rto_cost - (1.0 - p_cod) * (1.0 - 0.70) * margin

    el_matrix = np.vstack([el_allow, el_verify, el_deposit, el_prepaid])
    chosen_actions = np.argmin(el_matrix, axis=0)

    frics = np.array([0.0, 2.0, 0.0, 0.0])
    succ_drops = np.array([0.0, 0.05, 0.40, 0.70])
    rto_reds = np.array([0.0, 0.30, 0.80, 0.55])

    primary_losses = np.where(
        y_actual == 1,
        frics[chosen_actions] + (1.0 - rto_reds[chosen_actions]) * rto_cost,
        frics[chosen_actions] - (1.0 - succ_drops[chosen_actions]) * margin
    )

    total_primary_loss = np.sum(primary_losses)
    realized_savings = total_baseline_loss - total_primary_loss
    return realized_savings

def test_independent_realized_pl_calculation():
    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    model = joblib.load("models/tree_model_calibrated.pkl")
    X = test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
    proba = model.predict_proba(X)[:, 1]

    savings = compute_realized_savings(test, proba)
    assert np.isclose(savings, 69786.08, atol=1.0), f"Expected 69786.08 realized savings, got {savings}"
