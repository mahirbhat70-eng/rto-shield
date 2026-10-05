import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from audit.independent_engine import IndependentCostEngine
from src.policy.cost_engine import CostEngine, is_cod

indep_engine = IndependentCostEngine()
proj_engine = CostEngine()

print("="*80)
print("PROMPT 4 AUDIT: INDEPENDENT RECOMPUTATION OF COST ENGINE")
print("="*80)

test_cases = [
    (500.0, 0.05, "Low risk, modest value (ALLOW expected)"),
    (500.0, 0.25, "Moderate risk, modest value (VERIFY expected)"),
    (500.0, 0.45, "High risk, modest value (DEPOSIT expected)"),
    (2000.0, 0.05, "High value, low risk (ALLOW expected)"),
    (2000.0, 0.25, "High value, moderate risk"),
    (2000.0, 0.55, "High value, high risk"),
    (100.0, 0.80, "Very low value, high risk (DEPOSIT expected)"),
    (15000.0, 0.35, "Luxury value, moderate risk (Margin protects ALLOW)"),
    (50.0, 0.15, "Edge case: minimum catalog value 50"),
    (1.0, 0.10, "Edge case: order value 1"),
    (0.0, 0.10, "Edge case: order value 0"),
    (100000.0, 0.50, "Edge case: extreme order value 100,000"),
    (800.0, 0.00, "Edge case: P_RTO = 0.0 (Zero risk)"),
    (800.0, 0.50, "Edge case: P_RTO = 0.5 (Toss-up)"),
    (800.0, 1.00, "Edge case: P_RTO = 1.0 (Certain RTO)"),
]

print(f"\n--- STEP 3 & 4: 15 DIVERSE ORDERS HAND VS PROJECT COMPARISON ---")
print(f"{'#':<2} | {'V (INR)':<8} | {'P_RTO':<5} | {'ALLOW':<9} | {'VERIFY':<9} | {'DEPOSIT':<9} | {'PREPAID':<9} | {'Best Action':<15} | {'Match?'}")
print("-" * 90)

mismatch_count = 0
for i, (v, p, desc) in enumerate(test_cases, 1):
    indep_losses, indep_action = indep_engine.evaluate_order(v, p)
    proj_losses = proj_engine.evaluate_interventions(v, p)
    proj_action = min(proj_losses, key=proj_losses.get)

    diffs = [abs(indep_losses[k] - proj_losses[k]) for k in indep_losses]
    max_diff = max(diffs)
    action_match = (indep_action == proj_action)
    val_match = (max_diff <= 1e-9)

    if not (action_match and val_match):
        mismatch_count += 1
        status = f"MISMATCH (diff={max_diff:.2e})"
    else:
        status = "MATCH"

    print(f"{i:<2} | {v:<8.1f} | {p:<5.2f} | {indep_losses['ALLOW_COD']:<9.2f} | {indep_losses['VERIFY_ADDRESS']:<9.2f} | {indep_losses['REQUIRE_DEPOSIT']:<9.2f} | {indep_losses['PREPAID_ONLY']:<9.2f} | {indep_action:<15} | {status}")

print(f"\n15 Test Cases Result: {15 - mismatch_count}/15 matches (max diff <= 1e-9).")

# Step 5: Run both engines on full held-out test set
print("\n--- STEP 5: FULL HELD-OUT TEST SET EVALUATION ---")
test_df = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
model = joblib.load("models/tree_model_calibrated.pkl")
X_test = test_df.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
proba = model.predict_proba(X_test)[:, 1]

cod_mask = test_df['payment_method'].map(is_cod).to_numpy()
test_cod = test_df[cod_mask].copy().reset_index(drop=True)
proba_cod = proba[cod_mask]

N_cod = len(test_cod)
V_cod = test_cod['order_value'].values

indep_actions, indep_min_losses, indep_mat = indep_engine.evaluate_batch(V_cod, proba_cod)
indep_allow_losses = indep_mat[0, :]
indep_total_baseline = np.sum(indep_allow_losses)
indep_total_primary = np.sum(indep_min_losses)
indep_savings = indep_total_baseline - indep_total_primary

proj_actions, proj_min_losses = proj_engine.get_optimal_policy(test_cod, proba_cod)
proj_names, proj_mat = proj_engine.evaluate_interventions_vectorized(V_cod, proba_cod)
proj_total_baseline = np.sum(proj_mat[0, :])
proj_total_primary = np.sum(proj_min_losses)
proj_savings = proj_total_baseline - proj_total_primary

action_agreements = np.sum(np.array(indep_actions) == np.array(proj_actions))
loss_diffs = np.abs(indep_min_losses - proj_min_losses)
max_loss_diff = np.max(loss_diffs)

print(f"Total Held-Out COD Orders: {N_cod}")
print(f"Action Agreement Rate: {action_agreements}/{N_cod} ({action_agreements/N_cod*100:.4f}%)")
print(f"Max Individual Loss Difference: {max_loss_diff:.2e} INR")
print(f"Independent Total Baseline EL: {indep_total_baseline:,.2f} INR")
print(f"Project Total Baseline EL:     {proj_total_baseline:,.2f} INR")
print(f"Independent Total Primary EL:  {indep_total_primary:,.2f} INR")
print(f"Project Total Primary EL:      {proj_total_primary:,.2f} INR")
print(f"Independent Expected Savings:  {indep_savings:,.2f} INR")
print(f"Project Expected Savings:      {proj_savings:,.2f} INR")
print(f"Savings Discrepancy:           {abs(indep_savings - proj_savings):.2e} INR")

# Step 7: Closed-form threshold analysis across order values
print("\n--- STEP 7: CLOSED-FORM THRESHOLD SENSITIVITY TO ORDER VALUE ---")
def solve_t_verify(V):
    m = V * 0.20
    return (2.0 + 0.05 * m) / (0.30 * 150.0 + 0.05 * m)

def solve_t_prepaid(V):
    m = V * 0.20
    return (0.70 * m) / (0.55 * 150.0 + 0.70 * m)

sample_values = [50, 100, 300, 607.57, 826.89, 1500, 3000, 5000, 10000, 25000]
print(f"{'Order Value (V)':<16} | {'Margin (20%)':<12} | {'t_VERIFY':<10} | {'t_PREPAID':<10}")
print("-" * 55)
for v in sample_values:
    tv = solve_t_verify(v)
    tp = solve_t_prepaid(v)
    print(f"INR {v:<12.2f} | INR {v*0.20:<8.2f} | {tv:<10.4f} | {tp:<10.4f}")
