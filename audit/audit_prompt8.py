import os
import sys
import itertools
import numpy as np
import pandas as pd
import joblib

# Setup path
sys.path.insert(0, ".")
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset

test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
model = joblib.load("models/tree_model_calibrated.pkl")
X_test = test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
proba = model.predict_proba(X_test)[:, 1]
test_cod, proba_cod = get_cod_subset(test, proba)
y_cod = test_cod['rto_label'].values
V_cod = test_cod['order_value'].values

def evaluate_custom_engine(rto_cost, margin_pct, v_drop, d_drop, v_eff, d_eff=0.80):
    # Base loss
    base_losses = np.where(y_cod == 1, rto_cost, -margin_pct * V_cod)
    tot_base_realized = np.sum(base_losses)
    
    # Expected losses for decision routing
    margin = margin_pct * V_cod
    el_allow = proba_cod * rto_cost - (1.0 - proba_cod) * margin
    el_verify = 2.0 + (proba_cod * (1.0 - v_eff)) * rto_cost - (1.0 - proba_cod) * (1.0 - v_drop) * margin
    el_deposit = 0.50 + (proba_cod * (1.0 - d_eff)) * rto_cost - (1.0 - proba_cod) * (1.0 - d_drop) * margin
    el_prepaid = (proba_cod * (1.0 - 0.55)) * rto_cost - (1.0 - proba_cod) * (1.0 - 0.70) * margin
    
    # Policy choices
    el_matrix = np.vstack([el_allow, el_verify, el_deposit, el_prepaid])
    chosen = np.argmin(el_matrix, axis=0) # 0: allow, 1: verify, 2: deposit, 3: prepaid
    
    frics = np.array([0.0, 2.0, 0.50, 0.0])
    drops = np.array([0.0, v_drop, d_drop, 0.70])
    reds = np.array([0.0, v_eff, d_eff, 0.55])
    
    realized_policy_losses = np.where(
        y_cod == 1,
        frics[chosen] + (1.0 - reds[chosen]) * rto_cost,
        frics[chosen] - (1.0 - drops[chosen]) * margin
    )
    tot_policy_realized = np.sum(realized_policy_losses)
    savings_vs_allow = tot_base_realized - tot_policy_realized
    
    # Single threshold benchmark (e.g. verify if p >= 0.20 else allow)
    single_chosen = np.where(proba_cod >= 0.20, 1, 0)
    single_losses = np.where(
        y_cod == 1,
        frics[single_chosen] + (1.0 - reds[single_chosen]) * rto_cost,
        frics[single_chosen] - (1.0 - drops[single_chosen]) * margin
    )
    tot_single = np.sum(single_losses)
    savings_single = tot_base_realized - tot_single
    
    beats_allow = savings_vs_allow > 0
    beats_single = savings_vs_allow > savings_single
    
    return {
        'rto_cost': rto_cost, 'margin_pct': margin_pct, 
        'v_drop': v_drop, 'd_drop': d_drop, 'v_eff': v_eff,
        'realized_savings': round(savings_vs_allow, 2),
        'single_savings': round(savings_single, 2),
        'beats_allow': beats_allow,
        'beats_single': beats_single
    }

# Grid over parameter variations
rto_costs = [75.0, 150.0, 300.0]
margins = [0.10, 0.20, 0.35]
v_drops = [0.025, 0.05, 0.10]
d_drops = [0.20, 0.40, 0.80, 1.00]
v_effs = [0.15, 0.30]

rows = []
for r, m, vd, dd, ve in itertools.product(rto_costs, margins, v_drops, d_drops, v_effs):
    res = evaluate_custom_engine(r, m, vd, dd, ve)
    rows.append(res)

sens_df = pd.DataFrame(rows)
sens_df.to_csv("audit/sensitivity.csv", index=False)
print(f"Generated audit/sensitivity.csv with {len(sens_df)} scenarios.")

# Summary statistics of sensitivity
print("Scenarios beating Allow COD:", sens_df['beats_allow'].mean() * 100, "%")
print("Scenarios beating Single Threshold:", sens_df['beats_single'].mean() * 100, "%")

# Find Deposit Drop-off break-even (holding other defaults fixed)
print("\n--- DEPOSIT DROP-OFF BREAK-EVEN SCAN ---")
for dd in np.linspace(0.40, 1.00, 25):
    res = evaluate_custom_engine(150.0, 0.20, 0.05, dd, 0.30)
    print(f"Deposit Drop={dd:.2f}: Realized Savings=Rs {res['realized_savings']:8.2f}, Beats Single={res['beats_single']}, Beats Allow={res['beats_allow']}")
    if not res['beats_allow']:
        print(f"--> Breaks even vs Allow COD at Deposit Drop = {dd:.2f}")
        break
