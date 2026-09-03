import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def main():
    print("=" * 60)
    print("Stage 3: Calibration & Reliability")
    print("=" * 60)

    val_df = pd.read_csv("data/processed/val.csv", dtype={'pincode': str})
    val_cal_df = pd.read_csv("data/processed/val_cal.csv", dtype={'pincode': str})
    val_rep_df = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})

    tree_model = joblib.load('models/tree_model.pkl')
    lr_model = joblib.load('models/logistic_baseline.pkl')

    y_val = val_df['rto_label']
    X_val = val_df.drop(columns=['rto_label'])

    proba_tree = tree_model.predict_proba(X_val)[:, 1]
    proba_lr = lr_model.predict_proba(X_val)[:, 1]

    brier_tree = brier_score_loss(y_val, proba_tree)
    brier_lr = brier_score_loss(y_val, proba_lr)

    print(f"Brier Score (LR on full val): {brier_lr:.4f}")
    print(f"Brier Score (Tree on full val): {brier_tree:.4f}")

    frac_true_tree, prob_pred_tree = calibration_curve(y_val, proba_tree, n_bins=10)
    frac_true_lr, prob_pred_lr = calibration_curve(y_val, proba_lr, n_bins=10)

    mean_abs_err = np.mean(np.abs(prob_pred_tree - frac_true_tree))
    print(f"Tree Mean |Pred - Obs| (10 bins): {mean_abs_err:.4f}")

    os.makedirs('reports/stage3', exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated")
    plt.plot(prob_pred_lr, frac_true_lr, 's-', label=f"Logistic Regression (Brier: {brier_lr:.4f})")
    plt.plot(prob_pred_tree, frac_true_tree, 'o-', label=f"Uncalibrated Tree (Brier: {brier_tree:.4f})")

    if brier_tree > brier_lr or mean_abs_err > 0.03:
        print("\nCalibrating Tree Model...")
        
        y_val_cal = val_cal_df['rto_label']
        X_val_cal = val_cal_df.drop(columns=['rto_label'])
        
        # In sklearn >=1.6, we should use FrozenEstimator
        try:
            from sklearn.utils.estimator_checks import check_estimator
            from sklearn.utils import estimator_html_repr
            # Hack to check if FrozenEstimator exists
            from sklearn.frozen import FrozenEstimator
            calibrator = CalibratedClassifierCV(FrozenEstimator(tree_model), method='isotonic', cv=None)
        except ImportError:
            calibrator = CalibratedClassifierCV(estimator=tree_model, method='isotonic', cv='prefit')
            
        calibrator.fit(X_val_cal, y_val_cal)
        
        joblib.dump(calibrator, 'models/tree_model_calibrated.pkl')
        print("Saved models/tree_model_calibrated.pkl")
        
        # Evaluate calibrated tree on val_rep for the plot
        y_val_rep = val_rep_df['rto_label']
        X_val_rep = val_rep_df.drop(columns=['rto_label'])
        proba_cal = calibrator.predict_proba(X_val_rep)[:, 1]
        
        brier_cal = brier_score_loss(y_val_rep, proba_cal)
        print(f"Brier Score (Calibrated Tree on val_rep): {brier_cal:.4f}")
        
        frac_true_cal, prob_pred_cal = calibration_curve(y_val_rep, proba_cal, n_bins=10)
        plt.plot(prob_pred_cal, frac_true_cal, '^-', label=f"Calibrated Tree on val_rep (Brier: {brier_cal:.4f})")
    else:
        print("\nCalibration not required based on thresholds.")
    
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('reports/stage3/calibration_curve.png')
    print("Saved reports/stage3/calibration_curve.png")

if __name__ == "__main__":
    main()
