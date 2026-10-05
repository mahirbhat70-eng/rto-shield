import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, ".")
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action

test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
cal_model = joblib.load("models/tree_model_calibrated.pkl")
uncal_model = joblib.load("models/tree_model.pkl")

X_test = test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
y_test = test['rto_label'].values
p_cal = cal_model.predict_proba(X_test)[:, 1]
p_uncal = uncal_model.predict_proba(X_test)[:, 1]

def calc_calibration(y_true, proba, n_bins=10):
    if len(y_true) == 0:
        return 0.0, 0.0, None
    # Equal-count bins (quantiles)
    try:
        bins = pd.qcut(proba, q=n_bins, duplicates='drop')
    except Exception:
        bins = pd.cut(proba, bins=n_bins)
    df = pd.DataFrame({'y': y_true, 'p': proba, 'bin': bins})
    grouped = df.groupby('bin', observed=False).agg(
        count=('y', 'count'),
        mean_p=('p', 'mean'),
        empirical_rate=('y', 'mean')
    ).reset_index()
    grouped['abs_diff'] = np.abs(grouped['mean_p'] - grouped['empirical_rate'])
    ece = np.average(grouped['abs_diff'], weights=grouped['count'])
    max_error = grouped['abs_diff'].max()
    return ece, max_error, grouped

print("=== 1. RELIABILITY TABLES & ECE ===")
ece_all, max_all, g_all = calc_calibration(y_test, p_cal)
print(f"Overall Test (N={len(y_test)}): ECE={ece_all:.4f}, Max Bin Error={max_all:.4f}")

# COD only
cod_mask = (test['payment_method'] == 'COD').values
ece_cod, max_cod, g_cod = calc_calibration(y_test[cod_mask], p_cal[cod_mask])
print(f"COD Only (N={np.sum(cod_mask)}): ECE={ece_cod:.4f}, Max Bin Error={max_cod:.4f}")

# By recommended action
engine = CostEngine()
test_cod, proba_cod = get_cod_subset(test, p_cal)
actions, _ = eval_multi_action(test_cod, proba_cod, engine)
y_cod = test_cod['rto_label'].values

for act in ["ALLOW_COD", "VERIFY_ADDRESS", "REQUIRE_DEPOSIT"]:
    mask = (actions == act)
    ece_act, max_act, _ = calc_calibration(y_cod[mask], proba_cod[mask], n_bins=5)
    print(f"Action {act:16s} (N={np.sum(mask):5d}): ECE={ece_act:.4f}, Max Bin Error={max_act:.4f}")

# By pincode tier
for tier in [1, 2, 3]:
    mask = (test['pincode_tier'] == tier).values
    ece_tier, max_tier, _ = calc_calibration(y_test[mask], p_cal[mask])
    print(f"Pincode Tier {tier} (N={np.sum(mask):5d}): ECE={ece_tier:.4f}, Max Bin Error={max_tier:.4f}")

# 2. Chronological chunks
print("\n=== 2. CHRONOLOGICAL DRIFT (4 QUARTERS OF TEST SET) ===")
chunk_size = len(test) // 4
for q in range(4):
    start = q * chunk_size
    end = (q + 1) * chunk_size if q < 3 else len(test)
    y_q = y_test[start:end]
    p_q = p_cal[start:end]
    ece_q, max_q, _ = calc_calibration(y_q, p_q)
    print(f"Chunk Q{q+1} ({start}:{end}): Base RTO={y_q.mean():.4f}, Mean P={p_q.mean():.4f}, ECE={ece_q:.4f}, Max Error={max_q:.4f}")

# 3. Isotonic calibrator inspection
print("\n=== 3. ISOTONIC CALIBRATION OVERFITTING & STEP COUNT ===")
calibrator = cal_model.calibrated_classifiers_[0].calibrators[0]
print("Calibrator type:", type(calibrator))
if hasattr(calibrator, 'b_'):
    print("IsotonicRegression steps count:", len(calibrator.b_))
    print(f"X bounds: [{calibrator.a_.min():.4f}, {calibrator.a_.max():.4f}]")
    print(f"Y bounds: [{calibrator.b_.min():.4f}, {calibrator.b_.max():.4f}]")

# Fine grid check
fine_grid = np.linspace(0.01, 0.99, 100)
grid_pred = calibrator.predict(fine_grid)
flat_steps = np.diff(grid_pred) == 0
pct_flat = np.mean(flat_steps) * 100
print(f"Fine grid flatness: {pct_flat:.1f}% of consecutive grid points are on flat steps")

# 4. Calibrated vs Uncalibrated comparison
print("\n=== 4. CALIBRATED VS UNCALIBRATED LIGHTGBM ===")
brier_uncal = brier_score_loss(y_test, p_uncal)
brier_cal = brier_score_loss(y_test, p_cal)
ll_uncal = log_loss(y_test, p_uncal)
ll_cal = log_loss(y_test, p_cal)
ece_uncal, _, _ = calc_calibration(y_test, p_uncal)

print(f"Uncalibrated: Brier={brier_uncal:.4f}, LogLoss={ll_uncal:.4f}, ECE={ece_uncal:.4f}")
print(f"Calibrated:   Brier={brier_cal:.4f}, LogLoss={ll_cal:.4f}, ECE={ece_all:.4f}")
