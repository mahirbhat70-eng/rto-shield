"""
tests/test_independent_calibration.py
Validates model calibration on the test set directly.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss

def test_independent_calibration_metrics():
    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    model = joblib.load("models/tree_model_calibrated.pkl")
    X = test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
    proba = model.predict_proba(X)[:, 1]
    y = test['rto_label'].values

    brier = brier_score_loss(y, proba)
    assert np.isclose(brier, 0.1476, atol=0.005), f"Expected Brier score ~0.1476, got {brier:.4f}"

    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)

    bin_maes = []
    weights = []
    for i in range(10):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_p = np.mean(proba[mask])
            bin_y = np.mean(y[mask])
            bin_maes.append(abs(bin_p - bin_y))
            weights.append(np.sum(mask))

    weighted_bin_mae = np.average(bin_maes, weights=weights)
    assert weighted_bin_mae < 0.02, f"Expected weighted bin MAE < 0.02, got {weighted_bin_mae:.4f}"
