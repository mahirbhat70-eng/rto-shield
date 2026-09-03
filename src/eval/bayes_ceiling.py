import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def compute_z_det(df):
    beta_0 = -2.7

    cat_risk_map = {
        "Apparel": 0.30, "Electronics": 0.20, "Footwear": 0.15,
        "Beauty": 0.05, "Home": -0.10, "Jewelry": 0.08
    }
    cat_risk = df['category'].map(cat_risk_map).fillna(0).values

    courier_risk_map = {
        "Courier_A": 0.0, "Courier_B": 0.05, "Courier_C": -0.05,
        "Courier_D": 0.10, "Courier_E": 0.15
    }
    courier_risk = df['courier_id'].map(courier_risk_map).fillna(0).values

    tier_risk = np.where(df['pincode_tier'] == 3, 0.30,
                np.where(df['pincode_tier'] == 2, 0.10, -0.15))

    is_cod = (df['payment_method'] == "COD").astype(float).values

    z = (
        beta_0
        + 1.10 * is_cod
        + 0.80 * np.log1p(df['prior_rto_count'].values)
        + 2.50 * df['historical_pincode_rto_rate'].values
        + 0.25 * np.log1p(df['orders_last_24h'].values)
        + 0.20 * np.log1p(df['device_cluster_size'].values)
        - 0.12 * np.log1p(df['account_age_days'].values)
        - 0.15 * np.log1p(df['prior_orders'].values)
        + cat_risk
        + courier_risk
        + tier_risk
        + 0.005 * df['discount_pct'].values
    )
    return z

def get_true_p(df):
    z_det = compute_z_det(df)
    
    # We must account for the stochastic noise + rng.normal(0, 0.80) in generator.
    # The true probability P(RTO=1 | x) = E[ 1 / (1 + exp(-(z_det + e))) ]
    # We can approximate this via Monte Carlo
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.80, size=(1000, len(z_det)))
    z_noisy = z_det.reshape(1, -1) + noise
    p_noisy = 1.0 / (1.0 + np.exp(-z_noisy))
    p_expected = np.mean(p_noisy, axis=0)
    
    # The prompt specified a "final clip ([0.02, 0.85] per Stage 1 contract)"
    p_final = np.clip(p_expected, 0.02, 0.85)
    
    return p_final

def main():
    print("=" * 60)
    print("Stage 3.5: Bayes Ceiling Evaluation")
    print("=" * 60)

    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    y_true = val_rep['rto_label'].values
    
    p_expected = get_true_p(val_rep)
    
    # Validation
    val_rep['p_expected'] = p_expected
    val_rep['decile'] = pd.qcut(val_rep['p_expected'], q=10, duplicates='drop')
    bin_stats = val_rep.groupby('decile').agg(
        mean_p=('p_expected', 'mean'),
        empirical_rate=('rto_label', 'mean'),
        count=('rto_label', 'count')
    ).reset_index()
    
    bin_stats['abs_diff'] = np.abs(bin_stats['mean_p'] - bin_stats['empirical_rate'])
    print("Reconstruction Validation (Deciles):")
    print(bin_stats.to_string(index=False))
    
    max_dev = bin_stats['abs_diff'].max()
    print(f"\nMax systematic deviation: {max_dev:.4f}")
    if max_dev > 0.03:
        print("WARNING: Reconstruction deviation exceeds sampling noise (~0.015-0.03).")
    
    bayes_ceiling_pr_auc = average_precision_score(y_true, p_expected)
    print(f"\nBayes Ceiling Proxy PR-AUC: {bayes_ceiling_pr_auc:.4f}")
    
    # Compare with Tree Model
    calibrated_tree = joblib.load('models/tree_model_calibrated.pkl')
    X_val_rep = val_rep.drop(columns=['rto_label', 'p_expected', 'decile'])
    proba_tree = calibrated_tree.predict_proba(X_val_rep)[:, 1]
    
    tree_pr_auc = average_precision_score(y_true, proba_tree)
    ratio = tree_pr_auc / bayes_ceiling_pr_auc
    print(f"Calibrated Tree PR-AUC: {tree_pr_auc:.4f}")
    print(f"Model achieves {ratio*100:.1f}% of the theoretical maximum (Bayes Ceiling).")

if __name__ == "__main__":
    main()
