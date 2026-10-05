import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, brier_score_loss,
    average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
)
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_binary_policy, eval_multi_action, calc_operational_metrics

# 1. Load data and model
test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
model = joblib.load("models/tree_model_calibrated.pkl")

X_test = test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
y_test = test['rto_label'].values
proba = model.predict_proba(X_test)[:, 1]

# 2. Statistical metrics
pr_auc = average_precision_score(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)
brier = brier_score_loss(y_test, proba)

# Calibration weighted bin MAE (10 deciles)
bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(proba, bins) - 1
bin_indices = np.clip(bin_indices, 0, 9)
bin_maes = []
weights = []
for i in range(10):
    mask = (bin_indices == i)
    if np.sum(mask) > 0:
        bin_p = np.mean(proba[mask])
        bin_y = np.mean(y_test[mask])
        bin_maes.append(abs(bin_p - bin_y))
        weights.append(np.sum(mask))
weighted_bin_mae = np.average(bin_maes, weights=weights)

print("=== 1. TEST SET STATISTICAL METRICS (N=15,000) ===")
print(f"PR-AUC:           {pr_auc:.4f} (target: 0.3313)")
print(f"ROC-AUC:          {roc_auc:.4f} (target: 0.6729)")
print(f"Brier Score:      {brier:.4f} (target: 0.1476)")
print(f"Weighted Bin MAE: {weighted_bin_mae:.4f}")

# 3. COD Subset evaluation (7,174 orders)
engine = CostEngine()
test_cod, proba_cod = get_cod_subset(test, proba)
y_cod = test_cod['rto_label'].values

# Multi-action policy
actions, policy_losses = eval_multi_action(test_cod, proba_cod, engine)
baseline_actions, baseline_losses = eval_binary_policy(test_cod, proba_cod, engine, "ALLOW_COD", 1.0)

baseline_total = np.sum(baseline_losses)
policy_total = np.sum(policy_losses)
expected_savings = baseline_total - policy_total

# Operating point classification: Intervene vs Allow
# ALLOW_COD is negative (0), any intervention is positive (1)
pred_intervene = (actions != "ALLOW_COD").astype(int)
tn, fp, fn, tp = confusion_matrix(y_cod, pred_intervene).ravel()
prec = precision_score(y_cod, pred_intervene)
rec = recall_score(y_cod, pred_intervene)
f1 = f1_score(y_cod, pred_intervene)

print("\n=== 2. COD SUBSET OPERATING POINT CLASSIFICATION (N=7,174) ===")
print("Definition: Intervention Triggered (VERIFY/DEPOSIT) vs ALLOW_COD")
print(f"TN: {tn:,}  (target: 1,025)")
print(f"FP: {fp:,}  (target: 4,121)")
print(f"FN: {fn:,}    (target: 293)")
print(f"TP: {tp:,}  (target: 1,735)")
print(f"Precision: {prec:.4f} (target: 0.2963)")
print(f"Recall:    {rec:.4f} (target: 0.8555)")
print(f"F1-Score:  {f1:.4f}")

# 4. Action Distribution
print("\n=== 3. POLICY ACTION MIX ===")
action_counts = pd.Series(actions).value_counts()
for act in ["ALLOW_COD", "VERIFY_ADDRESS", "DEPOSIT_PARTIAL", "PREPAID_ONLY"]:
    cnt = action_counts.get(act, 0)
    pct = cnt / len(actions) * 100
    print(f"  {act:16s}: {cnt:5d} ({pct:5.1f}%)")

# 5. Operational Metrics
orders_touched, rtos_prev_exp, drops_exp, friction_spend = calc_operational_metrics(test_cod, proba_cod, actions, engine)

# Realized P&L
realized_baseline = np.sum(y_cod * 150.0)
realized_policy_losses = []
for i, row in test_cod.iterrows():
    act = actions[i]
    y = y_cod[i]
    V = row['order_value']
    p = proba_cod[i]
    if act == "ALLOW_COD":
        realized_policy_losses.append(y * 150.0)
    elif act == "VERIFY_ADDRESS":
        if y == 1:
            loss = 0.85 * 0.80 * 150.0 + 1.50
        else:
            loss = 0.15 * 0.25 * V + 1.50
        realized_policy_losses.append(loss)
    elif act == "DEPOSIT_PARTIAL":
        if y == 1:
            loss = 0.60 * 0.20 * 150.0 + 0.50
        else:
            loss = 0.40 * 0.25 * V + 0.50
        realized_policy_losses.append(loss)
    elif act == "PREPAID_ONLY":
        loss = 0.85 * 0.25 * V
        realized_policy_losses.append(loss)

realized_policy_total = np.sum(realized_policy_losses)
realized_savings = realized_baseline - realized_policy_total

print("\n=== 4. POLICY FINANCIAL RESULTS ===")
print(f"Baseline Expected Loss:  Rs {baseline_total:10.2f}")
print(f"Policy Expected Loss:    Rs {policy_total:10.2f}")
print(f"Expected Savings:        Rs {expected_savings:10.2f} (target: Rs 71,741.02)")
print(f"Baseline Realized Loss:  Rs {realized_baseline:10.2f}")
print(f"Policy Realized Loss:    Rs {realized_policy_total:10.2f}")
print(f"Realized Savings:        Rs {realized_savings:10.2f} (target: Rs 69,786.08)")
print(f"Orders Touched:          {orders_touched} (target: 5,856)")
print(f"Expected RTOs Prevented: {rtos_prev_exp:.2f} (target: 951.16)")
print(f"Good Customer Drops:     {drops_exp:.2f} (target: 820.08)")
print(f"Friction Spend (OTP+PG): Rs {friction_spend:10.2f} (target: Rs 6,488.00)")

# 6. Derived Claims Arithmetic
baseline_rtos = np.sum(y_cod)
rto_eliminated_pct = rtos_prev_exp / baseline_rtos * 100
effective_rto_rate = (baseline_rtos - rtos_prev_exp) / (len(test_cod) - rtos_prev_exp - drops_exp) * 100
uplift_pct = (expected_savings / baseline_total) * 100
binary_savings = 35919.14
ratio_vs_binary = expected_savings / binary_savings
savings_per_cod = expected_savings / len(test_cod)

print("\n=== 5. DERIVED CLAIMS ARITHMETIC ===")
print(f"Baseline COD RTOs:             {baseline_rtos}")
print(f"RTOs Eliminated:               {rtos_prev_exp:.2f} / {baseline_rtos} = {rto_eliminated_pct:.1f}% (headline: 46.9%)")
print(f"Effective Post-Policy RTO Rate: ({baseline_rtos} - {rtos_prev_exp:.2f}) / ({len(test_cod)} - {rtos_prev_exp:.2f} - {drops_exp:.2f}) = {effective_rto_rate:.2f}% (headline: 19.9%)")
print(f"Baseline RTO Rate (pre-policy): {baseline_rtos / len(test_cod) * 100:.2f}% (28.3%)")
print(f"Expected Uplift:               Rs {expected_savings:.2f} / Rs {baseline_total:.2f} = {uplift_pct:.2f}% (headline: 13.1%)")
print(f"Ratio vs Single Binary Policy: Rs {expected_savings:.2f} / Rs {binary_savings:.2f} = {ratio_vs_binary:.2f}x (headline: 2.0x)")
print(f"Savings per COD order:         Rs {expected_savings:.2f} / {len(test_cod)} = Rs {savings_per_cod:.2f} (headline: ~Rs 10)")
