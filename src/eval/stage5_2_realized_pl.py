import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_binary_policy, eval_multi_action

def compute_realized_loss(df, actions, config):
    # realized loss per action given the true rto_label
    total_loss = 0.0
    for idx, row in df.iterrows():
        # idx might not align if df is filtered but iterrows() idx is original index
        # Let's use position
        pass

    # Better to vectorize
    v = df['order_value'].values
    labels = df['rto_label'].values
    
    act_to_idx = {'ALLOW_COD': 0, 'VERIFY_ADDRESS': 1, 'REQUIRE_DEPOSIT': 2, 'PREPAID_ONLY': 3}
    acts_idx = np.array([act_to_idx[a] for a in actions])
    
    frics = np.array([0, 2, 0, 0])
    succ_drops = np.array([0.0, 0.05, 0.40, 0.70])
    rto_reds = np.array([0.0, 0.30, 0.80, 0.55])
    
    log_c = config['rto_logistics_cost']
    margins = v * config['average_margin_pct']
    
    # Expected realized cost
    losses = np.where(labels == 1,
        frics[acts_idx] + (1 - rto_reds[acts_idx]) * log_c,
        frics[acts_idx] - (1 - succ_drops[acts_idx]) * margins)
        
    return np.sum(losses)

def run():
    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    tree_uncal = joblib.load('models/tree_model.pkl')
    
    p_cal = tree_cal.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
    p_uncal = tree_uncal.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
    
    test_cod, p_cal_cod = get_cod_subset(test, p_cal)
    _, p_uncal_cod = get_cod_subset(test, p_uncal)
    
    engine = CostEngine()
    
    primary_actions, _ = eval_multi_action(test_cod, p_cal_cod, engine)
    
    y_true = test_cod['rto_label'].values
    y_pred_binary = np.isin(primary_actions, ['VERIFY_ADDRESS', 'REQUIRE_DEPOSIT']).astype(int)
    
    print("1. POLICY OPERATING-POINT P/R")
    print(f"Precision: {precision_score(y_true, y_pred_binary):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_binary):.4f}")
    print(f"F1: {f1_score(y_true, y_pred_binary):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred_binary)}")
    
    print("\nPer-Action Table:")
    print(f"{'Action':<16} | {'n':<5} | {'Mean P':<8} | {'Emp RTO':<8} | {'|Diff|':<8}")
    print("-" * 55)
    for act in ['REQUIRE_DEPOSIT', 'VERIFY_ADDRESS', 'ALLOW_COD', 'PREPAID_ONLY']:
        mask = (primary_actions == act)
        n = mask.sum()
        if n == 0:
            continue
        mean_p = p_cal_cod[mask].mean()
        emp_rto = y_true[mask].mean()
        diff = abs(mean_p - emp_rto)
        print(f"{act:<16} | {n:<5} | {mean_p:.4f}   | {emp_rto:.4f}   | {diff:.4f}")
        
    cfg = yaml.safe_load(open("configs/cost_config.yaml"))
    
    # 2. REALIZED-LABEL PORTFOLIO P&L
    baseline_actions, baseline_el = eval_binary_policy(test_cod, p_cal_cod, engine, "ALLOW_COD", 1.0)
    prepaid_actions, prepaid_el = eval_binary_policy(test_cod, p_cal_cod, engine, "PREPAID_ONLY", 0.48)
    verify_actions, verify_el = eval_binary_policy(test_cod, p_cal_cod, engine, "VERIFY_ADDRESS", 0.20)
    primary_actions, primary_el = eval_multi_action(test_cod, p_cal_cod, engine)
    uncal_actions, uncal_el = eval_multi_action(test_cod, p_uncal_cod, engine)
    clipped_actions, clipped_el = eval_multi_action(test_cod, np.clip(p_cal_cod, 0.02, 0.85), engine)
    
    base_real = compute_realized_loss(test_cod, baseline_actions, cfg)
    
    print("\n2. REALIZED-LABEL PORTFOLIO P&L")
    for name, acts, el_losses in [
        ("1. Baseline", baseline_actions, baseline_el),
        ("2. Binary PREPAID", prepaid_actions, prepaid_el),
        ("3. Binary VERIFY", verify_actions, verify_el),
        ("4. Primary Multi-Action", primary_actions, primary_el),
        ("5. Sensitivity: Uncal", uncal_actions, uncal_el),
        ("6. Sensitivity: Clipped", clipped_actions, clipped_el)
    ]:
        real_loss = compute_realized_loss(test_cod, acts, cfg)
        real_sav = base_real - real_loss
        el_sav = np.sum(baseline_el) - np.sum(el_losses)
        print(f"{name:<25}: Realized Savings: {real_sav:,.2f} | EL Savings: {el_sav:,.2f} | Delta: {real_sav - el_sav:,.2f}")
        
    # Monte Carlo 
    print("\nMonte Carlo Simulation (5,000 draws) for Primary Multi-Action Savings:")
    rng = np.random.default_rng(42)
    mc_savings = []
    
    acts = primary_actions
    vals = test_cod['order_value'].values
    labels = test_cod['rto_label'].values
    margins = vals * cfg['average_margin_pct']
    log_c = cfg['rto_logistics_cost']
    
    act_to_idx = {'ALLOW_COD': 0, 'VERIFY_ADDRESS': 1, 'REQUIRE_DEPOSIT': 2, 'PREPAID_ONLY': 3}
    acts_idx = np.array([act_to_idx[a] for a in acts])
    
    frics = np.array([0, 2, 0, 0])
    succ_drops = np.array([0.0, 0.05, 0.40, 0.70])
    rto_reds = np.array([0.0, 0.30, 0.80, 0.55])
    
    for i in range(5000):
        # Bernoulli for drop / red
        drops = rng.binomial(1, succ_drops[acts_idx])
        reds = rng.binomial(1, rto_reds[acts_idx])
        
        # Losses for Primary
        losses = np.where(labels == 1,
            frics[acts_idx] + (1 - reds) * log_c,
            frics[acts_idx] - (1 - drops) * margins)
            
        # Baseline is always ALLOW (idx 0)
        base_losses = np.where(labels == 1,
            log_c,
            -margins)
            
        mc_savings.append(np.sum(base_losses) - np.sum(losses))
        
    mc_savings = np.array(mc_savings)
    print(f"Mean: {np.mean(mc_savings):,.2f}")
    print(f"P5: {np.percentile(mc_savings, 5):,.2f}")
    print(f"P95: {np.percentile(mc_savings, 95):,.2f}")
    print(f"P(savings > 0): {np.mean(mc_savings > 0) * 100:.1f}%")

if __name__ == "__main__":
    run()
