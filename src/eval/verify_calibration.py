import joblib
import pandas as pd
import numpy as np

def compute_weighted_bin_mae(proba, y_true, model_name=""):
    # Create uniform bins from 0.0 to 1.0 (10 bins)
    bins = np.linspace(0.0, 1.0, 11)
    
    df = pd.DataFrame({'proba': proba, 'y': y_true})
    df['p_bin'] = pd.cut(df['proba'], bins=bins, include_lowest=True)
    
    stats = df.groupby('p_bin', observed=False).agg(
        n_i=('y', 'count'),
        mean_pred=('proba', 'mean'),
        empirical_rate=('y', 'mean')
    ).dropna() # drop empty bins
    
    stats['dev_i'] = np.abs(stats['mean_pred'] - stats['empirical_rate'])
    weighted_mae = np.sum(stats['n_i'] * stats['dev_i']) / np.sum(stats['n_i'])
    
    print(f"{model_name} Row-Weighted bin-MAE: {weighted_mae:.4f}")
    return weighted_mae

def main():
    print("=" * 60)
    print("Row-Weighted Bin-MAE Calculation")
    print("=" * 60)

    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    X = val_rep.drop(columns=['rto_label'])
    y = val_rep['rto_label'].values

    # Load models
    lr = joblib.load('models/logistic_baseline.pkl')
    tree_uncal = joblib.load('models/tree_model.pkl')
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')

    proba_lr = lr.predict_proba(X)[:, 1]
    proba_uncal = tree_uncal.predict_proba(X)[:, 1]
    proba_cal = tree_cal.predict_proba(X)[:, 1]

    compute_weighted_bin_mae(proba_lr, y, "Logistic Regression")
    compute_weighted_bin_mae(proba_uncal, y, "Uncalibrated Tree")
    compute_weighted_bin_mae(proba_cal, y, "Calibrated Tree")

if __name__ == "__main__":
    main()
