import os
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.policy.cost_engine import CostEngine

def get_cod_subset(df, proba):
    mask = df['payment_method'] == "COD"
    return df[mask].reset_index(drop=True), proba[mask]

def eval_binary_policy(df, proba, engine, action_name, threshold):
    losses = []
    actions = []
    
    for i, row in df.iterrows():
        p_rto = proba[i]
        order_value = row['order_value']
        
        chosen_action = action_name if p_rto > threshold else "ALLOW_COD"
        losses.append(engine.evaluate_interventions(order_value, p_rto)[chosen_action])
        actions.append(chosen_action)
        
    return np.array(actions), np.array(losses)

def grid_search_threshold(val_cal, proba_cal, engine, action_name):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_loss = float('inf')
    best_t = 0.5
    
    losses_per_t = []
    for t in thresholds:
        _, losses = eval_binary_policy(val_cal, proba_cal, engine, action_name, t)
        total_loss = np.sum(losses)
        losses_per_t.append(total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            best_t = t
            
    return best_t, thresholds, losses_per_t

def eval_multi_action(df, proba, engine):
    losses = []
    actions = []
    for i, row in df.iterrows():
        p_rto = proba[i]
        order_value = row['order_value']
        intervention_losses = engine.evaluate_interventions(order_value, p_rto)
        best_action = min(intervention_losses, key=intervention_losses.get)
        losses.append(intervention_losses[best_action])
        actions.append(best_action)
    return np.array(actions), np.array(losses)

def calc_operational_metrics(df, proba, actions, engine):
    orders_touched = sum(a != "ALLOW_COD" for a in actions)
    expected_rtos_prevented = 0.0
    expected_drops = 0.0
    friction_spend = 0.0
    
    for i, row in df.iterrows():
        a = actions[i]
        if a == "ALLOW_COD": continue
        p_rto = proba[i]
        params = engine.interventions[a]
        
        expected_rtos_prevented += p_rto * params['rto_reduction_pct']
        expected_drops += (1.0 - p_rto) * params['success_drop_pct']
        friction_spend += params['friction_cost']
        
    return orders_touched, expected_rtos_prevented, expected_drops, friction_spend

def main():
    os.makedirs('reports/stage4', exist_ok=True)
    engine = CostEngine()
    
    # Load data
    val_cal = pd.read_csv("data/processed/val_cal.csv", dtype={'pincode': str})
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    
    # Load models
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    tree_uncal = joblib.load('models/tree_model.pkl')
    
    # Grid search on val_cal COD subset
    val_cal_cod, proba_cal_cod = get_cod_subset(val_cal, tree_cal.predict_proba(val_cal.drop(columns=['rto_label']))[:, 1])
    
    t_prepaid, thresholds, losses_prepaid = grid_search_threshold(val_cal_cod, proba_cal_cod, engine, "PREPAID_ONLY")
    t_verify, _, losses_verify = grid_search_threshold(val_cal_cod, proba_cal_cod, engine, "VERIFY_ADDRESS")
    
    # Primary total loss on val_cal for the plot horizontal line
    _, primary_losses_cal = eval_multi_action(val_cal_cod, proba_cal_cod, engine)
    primary_total_cal = np.sum(primary_losses_cal)
    
    # Plot Threshold vs Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, losses_prepaid, label="Binary PREPAID Block")
    plt.plot(thresholds, losses_verify, label="Binary VERIFY Block")
    plt.axhline(y=primary_total_cal, color='r', linestyle='--', label="Multi-Action (Primary)")
    plt.title("Threshold vs Expected Loss (val_cal COD subset)")
    plt.xlabel("Threshold")
    plt.ylabel("Total Expected Loss (INR)")
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/stage4/threshold_vs_loss_curve.png')
    plt.close()
    
    # Evaluate on val_rep COD subset
    val_rep_cod, _ = get_cod_subset(val_rep, np.zeros(len(val_rep))) # Dummy proba
    
    proba_rep_cal = tree_cal.predict_proba(val_rep.drop(columns=['rto_label']))[:, 1]
    proba_rep_uncal = tree_uncal.predict_proba(val_rep.drop(columns=['rto_label']))[:, 1]
    
    _, p_cal_cod = get_cod_subset(val_rep, proba_rep_cal)
    _, p_uncal_cod = get_cod_subset(val_rep, proba_rep_uncal)
    p_clipped_cod = np.clip(p_cal_cod, 0.02, 0.85)
    
    # 1. Baseline
    baseline_actions, baseline_losses = eval_binary_policy(val_rep_cod, p_cal_cod, engine, "ALLOW_COD", 1.0)
    baseline_total = np.sum(baseline_losses)
    
    # Generate noisy probabilities early to evaluate each properly
    rng = np.random.default_rng(42)
    noise_cal = rng.normal(0, 0.04, size=len(p_cal_cod))
    noise_uncal = rng.normal(0, 0.04, size=len(p_uncal_cod))
    
    p_cal_noisy = np.clip(p_cal_cod + noise_cal, 0.0, 1.0)
    p_uncal_noisy = np.clip(p_uncal_cod + noise_uncal, 0.0, 1.0)
    p_clipped_noisy = np.clip(p_clipped_cod + noise_cal, 0.0, 1.0)
    
    _, baseline_noisy_losses = eval_binary_policy(val_rep_cod, p_cal_noisy, engine, "ALLOW_COD", 1.0)
    baseline_noisy_total = np.sum(baseline_noisy_losses)
    
    results = []
    
    def process_strategy(name, proba_clean, proba_noisy, eval_func):
        actions_clean, losses_clean = eval_func(proba_clean)
        total_loss = np.sum(losses_clean)
        savings = baseline_total - total_loss
        
        _, losses_noisy = eval_func(proba_noisy)
        total_loss_noisy = np.sum(losses_noisy)
        savings_noisy = baseline_noisy_total - total_loss_noisy
        delta_savings = savings_noisy - savings
        
        orders_touched, rtos_prevented, expected_drops, friction_spend = calc_operational_metrics(val_rep_cod, proba_clean, actions_clean, engine)
        
        # Action distribution
        dist = pd.Series(actions_clean).value_counts(normalize=True) * 100
        
        # Empirical P-Bands
        bands = {}
        for act in set(actions_clean):
            mask = actions_clean == act
            p_subset = proba_clean[mask]
            if len(p_subset) > 0:
                bands[act] = (np.min(p_subset), np.median(p_subset), np.max(p_subset))
                
        return {
            'Strategy': name,
            'Total Loss (INR)': total_loss,
            'Margin Saved vs Baseline': savings,
            'Orders Touched': orders_touched,
            'Expected RTOs Prevented': rtos_prevented,
            'Expected Good-Customer Drops': expected_drops,
            'Friction Spend': friction_spend,
            'Delta Savings (Noise)': delta_savings,
            'Action Dist': dist.to_dict(),
            'P-Bands': bands
        }
    
    # Add Baseline
    results.append({
        'Strategy': '1. Always Allow (Baseline)',
        'Total Loss (INR)': baseline_total,
        'Margin Saved vs Baseline': 0.0,
        'Orders Touched': 0,
        'Expected RTOs Prevented': 0.0,
        'Expected Good-Customer Drops': 0.0,
        'Friction Spend': 0.0,
        'Delta Savings (Noise)': 0.0,
        'Action Dist': {'ALLOW_COD': 100.0},
        'P-Bands': {'ALLOW_COD': (np.min(p_cal_cod), np.median(p_cal_cod), np.max(p_cal_cod))}
    })
    
    # 2. Binary PREPAID Block
    results.append(process_strategy(
        '2. Binary PREPAID Block', p_cal_cod, p_cal_noisy,
        lambda p: eval_binary_policy(val_rep_cod, p, engine, "PREPAID_ONLY", t_prepaid)
    ))
    
    # 3. Binary VERIFY Block
    results.append(process_strategy(
        '3. Binary VERIFY Block', p_cal_cod, p_cal_noisy,
        lambda p: eval_binary_policy(val_rep_cod, p, engine, "VERIFY_ADDRESS", t_verify)
    ))
    
    # 4. Multi-Action AI Policy (Primary)
    results.append(process_strategy(
        '4. Multi-Action AI Policy (Primary)', p_cal_cod, p_cal_noisy,
        lambda p: eval_multi_action(val_rep_cod, p, engine)
    ))
    
    # 5. Multi-Action AI Policy (Sensitivity)
    results.append(process_strategy(
        '5. Multi-Action AI Policy (Sensitivity)', p_uncal_cod, p_uncal_noisy,
        lambda p: eval_multi_action(val_rep_cod, p, engine)
    ))
    
    # 6. Multi-Action AI Policy (Constraint-Clipped Sensitivity)
    results.append(process_strategy(
        '6. Multi-Action AI Policy (Constraint-Clipped Sensitivity)', p_clipped_cod, p_clipped_noisy,
        lambda p: eval_multi_action(val_rep_cod, p, engine)
    ))
    
    # Write report
    with open('reports/stage4_financial_results.md', 'w') as f:
        f.write("# Stage 4 Financial Results (COD Subset Only)\n\n")
        f.write(f"**Tuned Thresholds (val_cal)**:\n")
        f.write(f"- PREPAID_ONLY Threshold: {t_prepaid:.2f}\n")
        f.write(f"- VERIFY_ADDRESS Threshold: {t_verify:.2f}\n\n")
        
        for r in results:
            f.write(f"### {r['Strategy']}\n")
            f.write(f"- **Total Loss (INR)**: {r['Total Loss (INR)']:.2f}\n")
            f.write(f"- **Total Margin Saved**: {r['Margin Saved vs Baseline']:.2f}\n")
            f.write(f"- **Delta Savings (Noise Test)**: {r['Delta Savings (Noise)']:.2f}\n")
            f.write(f"- **Operational Counts**:\n")
            f.write(f"  - Orders Touched: {r['Orders Touched']}\n")
            f.write(f"  - Expected RTOs Prevented: {r['Expected RTOs Prevented']:.2f}\n")
            f.write(f"  - Expected Good-Customer Drops: {r['Expected Good-Customer Drops']:.2f}\n")
            f.write(f"  - Friction Spend: {r['Friction Spend']:.2f}\n")
            
            f.write(f"- **Action Distribution & P-Bands**:\n")
            for act, pct in r['Action Dist'].items():
                band = r['P-Bands'].get(act)
                if band:
                    f.write(f"  - {act}: {pct:.1f}% (P-band: {band[0]:.4f} - {band[2]:.4f}, median {band[1]:.4f})\n")
                else:
                    f.write(f"  - {act}: {pct:.1f}%\n")
            f.write("\n")
            
    print("Stage 4 evaluation complete.")
    for r in results:
        print(f"{r['Strategy']}: Loss = {r['Total Loss (INR)']:.2f}, Saved = {r['Margin Saved vs Baseline']:.2f}")

if __name__ == "__main__":
    main()
