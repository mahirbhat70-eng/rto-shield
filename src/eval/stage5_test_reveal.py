import os
import sys
import hashlib
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_binary_policy, eval_multi_action
from src.eval.verify_calibration import compute_weighted_bin_mae

def compute_uniform_bin_mae(proba, y_true):
    bins = np.linspace(0.0, 1.0, 11)
    df = pd.DataFrame({'proba': proba, 'y': y_true})
    df['p_bin'] = pd.cut(df['proba'], bins=bins, include_lowest=True)
    stats = df.groupby('p_bin', observed=False).agg(
        n_i=('y', 'count'),
        mean_pred=('proba', 'mean'),
        empirical_rate=('y', 'mean')
    ).dropna()
    dev_i = np.abs(stats['mean_pred'] - stats['empirical_rate'])
    return np.mean(dev_i)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_violation(condition, message):
    if not condition:
        print(f"VIOLATION: {message}")
        sys.exit(1)

def main():
    test_csv_path = "data/processed/test.csv"
    initial_sha = sha256_file(test_csv_path)
    
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    test = pd.read_csv(test_csv_path, dtype={'pincode': str})
    
    # 1. SPLIT INTEGRITY & PROFILE
    check_violation(val_rep['timestamp'].max() < test['timestamp'].min(), "Strict temporal ordering violated.")
    
    test_rto_rate = test['rto_label'].mean()
    print("=" * 60)
    print("1. TEST SPLIT PROFILE")
    print(f"Row count: {len(test)}")
    print(f"Date range: {test['timestamp'].min()} to {test['timestamp'].max()}")
    print(f"RTO rate: {test_rto_rate:.4f}")
    
    check_violation(0.18 <= test_rto_rate <= 0.22, f"Test RTO rate {test_rto_rate:.4f} outside [0.18, 0.22]")

    # 2. BAYES CEILING ON TEST
    from src.eval.bayes_ceiling import get_true_p
    p_reconstructed = get_true_p(test)
    ceiling_pr_auc = average_precision_score(test['rto_label'], p_reconstructed)
    
    # Validate deciles
    test['p_recon'] = p_reconstructed
    test['decile'] = pd.qcut(test['p_recon'], q=10, duplicates='drop')
    decile_stats = test.groupby('decile', observed=False).agg(
        emp_rate=('rto_label', 'mean'),
        mean_p=('p_recon', 'mean'),
        n=('rto_label', 'count')
    ).reset_index()
    max_dev = np.max(np.abs(decile_stats['emp_rate'] - decile_stats['mean_p']))
    
    print("\n" + "=" * 60)
    print("2. BAYES CEILING ON TEST")
    print(f"Max decile deviation: {max_dev:.4f}")
    print(f"Ceiling PR-AUC: {ceiling_pr_auc:.4f}")
    
    check_violation(max_dev <= 0.035, f"Bayes ceiling decile deviation {max_dev:.4f} exceeds ~0.03") # Slight buffer for noise
    check_violation(0.31 <= ceiling_pr_auc <= 0.39, f"Ceiling PR-AUC {ceiling_pr_auc:.4f} outside [0.31, 0.39]")

    # 3. MODEL METRICS ON TEST
    X_test = test.drop(columns=['rto_label', 'p_recon', 'decile', 'timestamp', 'order_id'], errors='ignore')
    y_test = test['rto_label'].values
    from src.models.rule_baseline import derive_thresholds, predict_rules
    train_df = pd.read_csv("data/processed/train.csv", dtype={'pincode': str})
    rule_thresholds = derive_thresholds(train_df)
    rule_preds = predict_rules(test, rule_thresholds)
    rule_pr_auc = average_precision_score(y_test, rule_preds)
    
    lr = joblib.load('models/logistic_baseline.pkl')
    tree_uncal = joblib.load('models/tree_model.pkl')
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    
    p_lr = lr.predict_proba(X_test)[:, 1]
    p_uncal = tree_uncal.predict_proba(X_test)[:, 1]
    p_cal = tree_cal.predict_proba(X_test)[:, 1]
    
    metrics = {}
    for name, p in [('LR@0.5', p_lr), ('Uncal Tree', p_uncal), ('Cal Tree (PRIMARY)', p_cal)]:
        metrics[name] = {
            'PR-AUC': average_precision_score(y_test, p),
            'ROC-AUC': roc_auc_score(y_test, p),
            'Brier': brier_score_loss(y_test, p),
            'Unif_Bin_MAE': compute_uniform_bin_mae(p, y_test),
            'W_Bin_MAE': compute_weighted_bin_mae(p, y_test, name)
        }
        
    print("\n" + "=" * 60)
    print("3. MODEL METRICS ON TEST")
    print(f"Rule PR-AUC: {rule_pr_auc:.4f}")
    for k, v in metrics.items():
        print(f"{k}: PR-AUC={v['PR-AUC']:.4f}, Brier={v['Brier']:.4f}, W-Bin-MAE={v['W_Bin_MAE']:.4f}")
        
    cal_pr = metrics['Cal Tree (PRIMARY)']['PR-AUC']
    check_violation(0.29 <= cal_pr <= 0.37, f"Calibrated PR-AUC {cal_pr:.4f} outside [0.29, 0.37]")
    check_violation((cal_pr / ceiling_pr_auc) >= 0.88, f"Calibrated PR-AUC ratio to ceiling {(cal_pr / ceiling_pr_auc):.4f} < 0.88")

    # 4. POLICY EVALUATION
    test_cod, p_cal_cod = get_cod_subset(test, p_cal)
    _, p_uncal_cod = get_cod_subset(test, p_uncal)
    
    print("\n" + "=" * 60)
    print("4. POLICY EVALUATION (COD SUBSET)")
    print(f"COD subset size: {len(test_cod)}")
    print(f"Mean order_value: {test_cod['order_value'].mean():.2f}")
    print(f"Median order_value: {test_cod['order_value'].median():.2f}")
    print(f"Mean calibrated P: {p_cal_cod.mean():.4f}")
    
    engine = CostEngine()
    
    baseline_actions, baseline_losses = eval_binary_policy(test_cod, p_cal_cod, engine, "ALLOW_COD", 1.0)
    baseline_total = np.sum(baseline_losses)
    
    prepaid_actions, prepaid_losses = eval_binary_policy(test_cod, p_cal_cod, engine, "PREPAID_ONLY", 0.48)
    prepaid_total = np.sum(prepaid_losses)
    
    verify_actions, verify_losses = eval_binary_policy(test_cod, p_cal_cod, engine, "VERIFY_ADDRESS", 0.20)
    verify_total = np.sum(verify_losses)
    
    primary_actions, primary_losses = eval_multi_action(test_cod, p_cal_cod, engine)
    primary_total = np.sum(primary_losses)
    
    uncal_actions, uncal_losses = eval_multi_action(test_cod, p_uncal_cod, engine)
    uncal_total = np.sum(uncal_losses)
    
    p_clipped_cod = np.clip(p_cal_cod, 0.02, 0.85)
    clipped_actions, clipped_losses = eval_multi_action(test_cod, p_clipped_cod, engine)
    clipped_total = np.sum(clipped_losses)
    
    primary_savings = baseline_total - primary_total
    
    def report_policy(name, total, baseline, actions, p_arr):
        savings = baseline - total
        n_cod = len(test_cod)
        n_verify = np.sum(actions == 'VERIFY_ADDRESS')
        n_deposit = np.sum(actions == 'REQUIRE_DEPOSIT')
        n_prepaid = np.sum(actions == 'PREPAID_ONLY')
        n_allow = np.sum(actions == 'ALLOW_COD')
        
        touched = n_verify + n_deposit + n_prepaid
        friction = n_verify * 2
        
        print(f"\n{name}:")
        print(f"  Loss: {total:.2f}, Savings: {savings:.2f}")
        print(f"  Orders Touched (n_VERIFY+n_DEPOSIT): {touched} / {n_cod}")
        print(f"  Friction Spend (2 * n_VERIFY): {friction}")
        print(f"  Action Shares (Denom: {n_cod} COD rows):")
        print(f"    ALLOW_COD: {n_allow/n_cod*100:.1f}%")
        print(f"    VERIFY_ADDRESS: {n_verify/n_cod*100:.1f}%")
        print(f"    REQUIRE_DEPOSIT: {n_deposit/n_cod*100:.1f}%")
        print(f"    PREPAID_ONLY: {n_prepaid/n_cod*100:.1f}%")
        
        return savings, n_prepaid
        
    report_policy("1. Baseline", baseline_total, baseline_total, baseline_actions, p_cal_cod)
    prepaid_sav, prepaid_share = report_policy("2. Binary PREPAID", prepaid_total, baseline_total, prepaid_actions, p_cal_cod)
    report_policy("3. Binary VERIFY", verify_total, baseline_total, verify_actions, p_cal_cod)
    primary_sav, primary_prepaid_share = report_policy("4. Multi-Action Primary", primary_total, baseline_total, primary_actions, p_cal_cod)
    report_policy("5. Multi-Action Sensitivity (Uncal)", uncal_total, baseline_total, uncal_actions, p_uncal_cod)
    report_policy("6. Multi-Action Clipped", clipped_total, baseline_total, clipped_actions, p_clipped_cod)
    
    check_violation(primary_prepaid_share == 0, "PREPAID_ONLY share in Primary is not 0%")
    check_violation(primary_total <= verify_total <= prepaid_total, "Loss ordering violated")
    check_violation(np.abs(prepaid_sav) <= 0.02 * np.abs(baseline_total), "Prepaid block >2% diff from baseline")
    check_violation(0.08 * np.abs(baseline_total) <= primary_sav <= 0.18 * np.abs(baseline_total), "Primary savings not in [8%, 18%] of baseline loss")

    # 5. NOISE SENSITIVITY
    rng = np.random.default_rng(42)
    p_cal_noisy = np.clip(p_cal_cod + rng.normal(0, 0.04, size=len(p_cal_cod)), 0.0, 1.0)
    
    _, base_noisy_loss = eval_binary_policy(test_cod, p_cal_noisy, engine, "ALLOW_COD", 1.0)
    _, prim_noisy_loss = eval_multi_action(test_cod, p_cal_noisy, engine)
    
    noisy_savings = np.sum(base_noisy_loss) - np.sum(prim_noisy_loss)
    
    print("\n" + "=" * 60)
    print("5. NOISE SENSITIVITY")
    print(f"Clean Savings: {primary_sav:.2f}, Noisy Savings: {noisy_savings:.2f}")
    
    check_violation(np.abs(noisy_savings - primary_sav) <= 0.10 * primary_sav, "Noisy savings outside ±10% of clean savings")
    
    final_sha = sha256_file(test_csv_path)
    check_violation(initial_sha == final_sha, "test.csv SHA-256 changed!")

    print("\nALL PRE-REGISTERED CHECKS PASSED.")

if __name__ == "__main__":
    main()
