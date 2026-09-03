import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve
)

def evaluate_model(model_name, proba, y_true, rule_flags=None, threshold=0.5):
    preds = (proba >= threshold).astype(int)
    
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    flag_rate = preds.mean()
    
    prec_at_rule = np.nan
    if rule_flags is not None and proba is not None:
        rule_recall = recall_score(y_true, rule_flags, zero_division=0)
        p, r, t = precision_recall_curve(y_true, proba)
        prec_at_rule = np.interp(rule_recall, r[:-1][::-1], p[:-1][::-1])
        
    return {
        'Model': model_name,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Brier': brier,
        'Flag Rate': flag_rate,
        'Prec@Rule Recall': prec_at_rule
    }

def main():
    print("=" * 60)
    print("Stage 3: Comprehensive Evaluation (val_rep)")
    print("=" * 60)

    df = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    y_true = df['rto_label'].values
    
    # 1. Rule Baseline
    rule_flags = (
        ((df['historical_pincode_rto_rate'] > 0.25) & (df['category'] == 'Electronics')) |
        (df['orders_last_24h'] >= 3) |
        (df['prior_rto_count'] >= 2)
    ).astype(int)
    
    rule_prec = precision_score(y_true, rule_flags, zero_division=0)
    rule_rec = recall_score(y_true, rule_flags, zero_division=0)
    rule_f1 = f1_score(y_true, rule_flags, zero_division=0)
    rule_brier = brier_score_loss(y_true, rule_flags) # MSE of 0/1 flags
    rule_flag_rate = rule_flags.mean()
    
    prevalence = y_true.mean()
    rule_pr_auc = rule_prec * rule_rec + prevalence * (1 - rule_rec)
    
    results = []
    results.append({
        'Model': 'Rule Baseline',
        'Precision': rule_prec,
        'Recall': rule_rec,
        'F1': rule_f1,
        'PR-AUC': rule_pr_auc,
        'ROC-AUC': np.nan,
        'Brier': rule_brier,
        'Flag Rate': rule_flag_rate,
        'Prec@Rule Recall': rule_prec # trivially itself
    })
    
    # 2. Logistic Regression
    lr_pipeline = joblib.load('models/logistic_baseline.pkl')
    X_lr = df.drop(columns=['rto_label'])
    proba_lr = lr_pipeline.predict_proba(X_lr)[:, 1]
    results.append(evaluate_model('Logistic Regression', proba_lr, y_true, rule_flags))
    
    # 3. Uncalibrated Tree
    tree_pipeline = joblib.load('models/tree_model.pkl')
    proba_tree = tree_pipeline.predict_proba(X_lr)[:, 1]
    results.append(evaluate_model('Uncalibrated Tree (LGBM)', proba_tree, y_true, rule_flags))
    
    # 4. Calibrated Tree (if exists)
    if os.path.exists('models/tree_model_calibrated.pkl'):
        calibrated_tree = joblib.load('models/tree_model_calibrated.pkl')
        proba_cal = calibrated_tree.predict_proba(X_lr)[:, 1]
        results.append(evaluate_model('Calibrated Tree (LGBM)', proba_cal, y_true, rule_flags))
        
    results_df = pd.DataFrame(results)
    
    os.makedirs('reports/stage3', exist_ok=True)
    markdown_table = results_df.to_markdown(index=False, floatfmt=".4f")
    
    with open('reports/stage3/stage3_results.md', 'w') as f:
        f.write("# Stage 3 Results on `val_rep`\n\n")
        f.write(markdown_table)
        f.write("\n")
        
    print(markdown_table)
    
    # Noise Stress Test on Tree Model
    print("\n" + "=" * 60)
    print("Noise Stress Test on Uncalibrated Tree")
    print("=" * 60)
    
    df_noisy = df.copy()
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.04, size=len(df_noisy))
    df_noisy['historical_pincode_rto_rate'] = (df_noisy['historical_pincode_rto_rate'] + noise).clip(0, 1)
    
    X_noisy = df_noisy.drop(columns=['rto_label'])
    proba_noisy = tree_pipeline.predict_proba(X_noisy)[:, 1]
    pr_auc_noisy = average_precision_score(y_true, proba_noisy)
    
    clean_pr_auc = results_df.loc[results_df['Model'] == 'Uncalibrated Tree (LGBM)', 'PR-AUC'].values[0]
    delta = pr_auc_noisy - clean_pr_auc
    
    print(f"Clean PR-AUC: {clean_pr_auc:.4f}")
    print(f"Noisy PR-AUC: {pr_auc_noisy:.4f}")
    print(f"Delta PR-AUC: {delta:.4f}")

if __name__ == "__main__":
    main()
