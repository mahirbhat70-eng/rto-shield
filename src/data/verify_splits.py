import pandas as pd
import numpy as np
import joblib

def verify_splits():
    train = pd.read_csv('data/processed/train.csv', dtype={'pincode': str})
    val = pd.read_csv('data/processed/val.csv', dtype={'pincode': str})
    test = pd.read_csv('data/processed/test.csv', dtype={'pincode': str})

    train_ts = pd.to_datetime(train['timestamp'])
    val_ts = pd.to_datetime(val['timestamp'])
    test_ts = pd.to_datetime(test['timestamp'])

    print("--- 1a. Temporal Boundary Check ---")
    assert train_ts.max() <= val_ts.min(), "Train overlaps Val!"
    assert val_ts.max() <= test_ts.min(), "Val overlaps Test!"
    print(f"Train max : {train_ts.max()}")
    print(f"Val min   : {val_ts.min()}")
    print(f"Val max   : {val_ts.max()}")
    print(f"Test min  : {test_ts.min()}")
    print("Strict inequality holds:", train_ts.max() < val_ts.min(), val_ts.max() < test_ts.min())

    print("\n--- 1b. Feature Leakage Check ---")
    # Is historical_pincode_rto_rate just the full-period average?
    full_dataset = pd.concat([train, val, test]).sort_values('timestamp').reset_index(drop=True)
    
    # Calculate full-period actual RTO rate per pincode
    full_actual = full_dataset.groupby('pincode')['rto_label'].mean()
    
    # Compare stored historical_pincode_rto_rate with full-period actual rate
    sample = full_dataset.sample(300, random_state=42)
    mismatches = 0
    max_diff = 0
    for _, row in sample.iterrows():
        pincode = row['pincode']
        stored = row['historical_pincode_rto_rate']
        actual_full = full_actual[pincode]
        diff = abs(stored - actual_full)
        if diff > 1e-4:
            mismatches += 1
        max_diff = max(max_diff, diff)
        
    print(f"Mismatches with full-period actual: {mismatches}/300 (Max abs diff: {max_diff:.4f})")
    if mismatches == 300:
        print("GOOD: stored values DO NOT systematically match full-period rates (no naive full-period leakage).")
    
    print("\n--- 2. LR Underfitting Check ---")
    pipeline = joblib.load('models/logistic_baseline.pkl')
    clf = pipeline.named_steps['classifier']
    X_val = val.drop(columns=['rto_label'])
    proba = pipeline.predict_proba(X_val)[:, 1]
    
    print(f"LR max predict_proba (val) : {proba.max():.4f}")
    print(f"LR n_iter_                 : {clf.n_iter_[0]}")
    print(f"LR max_iter                : {clf.max_iter}")

if __name__ == '__main__':
    verify_splits()
