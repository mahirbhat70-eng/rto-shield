import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

def evaluate(model_name, pipeline, df):
    X = df.drop(columns=['rto_label'])
    y_true = df['rto_label'].values
    
    proba = pipeline.predict_proba(X)[:, 1]
    
    pr_auc = average_precision_score(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    
    frac_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)
    bin_mae = np.mean(np.abs(prob_pred - frac_true))
    
    print(f"\n{model_name}:")
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Brier:    {brier:.4f}")
    print(f"  bin-MAE:  {bin_mae:.4f}")

def main():
    print("=" * 60)
    print("Apples-to-Apples Evaluation (val_rep)")
    print("=" * 60)

    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    
    # 1. Logistic Baseline
    lr = joblib.load('models/logistic_baseline.pkl')
    evaluate("Logistic Regression", lr, val_rep)
    
    # 2. Calibrated Tree
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    evaluate("Calibrated Tree", tree_cal, val_rep)

if __name__ == "__main__":
    main()
