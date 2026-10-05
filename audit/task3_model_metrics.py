"""
audit/task3_model_metrics.py — Single script to recompute PR-AUC, ROC-AUC, Brier,
LogLoss, and ECE from raw test predictions for all 4 models:
1. Logistic Regression
2. LightGBM (uncalibrated)
3. LightGBM (isotonic-calibrated)
4. LightGBM (Platt-calibrated)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

def ece_score(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.any():
            error += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return error

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test = pd.read_csv(os.path.join(repo_root, "data/processed/test.csv"), dtype={"pincode": str})
    val_cal = pd.read_csv(os.path.join(repo_root, "data/processed/val_cal.csv"), dtype={"pincode": str})

    drop_cols = ["rto_label", "timestamp", "order_id"]
    X_test = test.drop(columns=drop_cols, errors="ignore")
    y_test = test["rto_label"].values

    X_val_cal = val_cal.drop(columns=drop_cols, errors="ignore")
    y_val_cal = val_cal["rto_label"].values

    # Load frozen models
    lr = joblib.load(os.path.join(repo_root, "models/logistic_baseline.pkl"))
    lgbm_uncal = joblib.load(os.path.join(repo_root, "models/tree_model.pkl"))
    lgbm_iso = joblib.load(os.path.join(repo_root, "models/tree_model_calibrated.pkl"))

    # Fit Platt scaling on val_cal uncalibrated predictions
    p_val_cal_uncal = lgbm_uncal.predict_proba(X_val_cal)[:, 1]
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(p_val_cal_uncal.reshape(-1, 1), y_val_cal)

    # Predict on test set
    p_lr = lr.predict_proba(X_test)[:, 1]
    p_uncal = lgbm_uncal.predict_proba(X_test)[:, 1]
    p_iso = lgbm_iso.predict_proba(X_test)[:, 1]
    p_platt = platt.predict_proba(p_uncal.reshape(-1, 1))[:, 1]

    models = {
        "Logistic Regression": p_lr,
        "LightGBM (uncalibrated)": p_uncal,
        "LightGBM (isotonic)": p_iso,
        "LightGBM (Platt)": p_platt,
    }

    results = []
    for name, p in models.items():
        p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
        results.append({
            "Model": name,
            "PR-AUC": float(average_precision_score(y_test, p)),
            "ROC-AUC": float(roc_auc_score(y_test, p)),
            "Brier": float(brier_score_loss(y_test, p)),
            "LogLoss": float(log_loss(y_test, p_safe)),
            "ECE": float(ece_score(y_test, p, n_bins=10)),
        })

    df = pd.DataFrame(results)
    print("=" * 80)
    print("RAW TEST METRICS (N=14,980 Holdout Test Orders):")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("=" * 80)

if __name__ == "__main__":
    main()
