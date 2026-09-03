import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def stress_test_noise():
    print("=" * 60)
    print("Stage 3 Preparation: Production-Realism Stress Test")
    print("Adding empirical estimation noise (sigma=0.04) to oracle-grade historical_pincode_rto_rate")
    print("=" * 60)

    # Load model
    pipeline = joblib.load('models/logistic_baseline.pkl')
    
    # Evaluate on val and test
    for split_name in ['val', 'test']:
        df = pd.read_csv(f'data/processed/{split_name}.csv', dtype={'pincode': str})
        
        # Original evaluation
        X_clean = df.drop(columns=['rto_label'])
        y_true = df['rto_label'].values
        proba_clean = pipeline.predict_proba(X_clean)[:, 1]
        pr_auc_clean = average_precision_score(y_true, proba_clean)
        
        # Noisy evaluation
        df_noisy = df.copy()
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.04, size=len(df_noisy))
        df_noisy['historical_pincode_rto_rate'] = (df_noisy['historical_pincode_rto_rate'] + noise).clip(0, 1)
        
        X_noisy = df_noisy.drop(columns=['rto_label'])
        proba_noisy = pipeline.predict_proba(X_noisy)[:, 1]
        pr_auc_noisy = average_precision_score(y_true, proba_noisy)
        
        delta = pr_auc_noisy - pr_auc_clean
        
        print(f"\n{split_name.upper()} Set:")
        print(f"  Clean PR-AUC: {pr_auc_clean:.4f}")
        print(f"  Noisy PR-AUC: {pr_auc_noisy:.4f}")
        print(f"  Delta PR-AUC    : {delta:.4f}")

if __name__ == "__main__":
    stress_test_noise()
