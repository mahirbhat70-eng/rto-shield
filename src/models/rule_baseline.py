"""
RTO Shield -- Stage 2: Rule-Based Baseline

A simple deterministic if/else policy that a merchant ops team could write today.
Thresholds are derived from the TRAIN set only (top quartile / percentiles).
No learned model -- purely hand-crafted rules.
"""

import os
import pandas as pd
import numpy as np


def derive_thresholds(train_df):
    """
    Derive rule thresholds from TRAIN set only.
    Returns dict of thresholds used.
    """
    # threshold_A: top quartile of historical_pincode_rto_rate in train
    threshold_A = train_df['historical_pincode_rto_rate'].quantile(0.75)

    # threshold_B: prior_rto_count >= 2 flags repeat offenders
    threshold_B = 2

    # threshold_C: new-ish accounts (bottom 25th percentile of account_age_days)
    threshold_C = train_df['account_age_days'].quantile(0.25)

    thresholds = {
        'hist_rate_threshold': round(float(threshold_A), 4),
        'prior_rto_threshold': int(threshold_B),
        'account_age_threshold': int(threshold_C),
    }
    return thresholds


def predict_rules(df, thresholds):
    """
    Predict RTO=1 if:
      historical_pincode_rto_rate > threshold_A
      OR prior_rto_count >= threshold_B
      OR (payment_method == "COD" AND account_age_days < threshold_C)

    Returns numpy array of binary predictions (no access to rto_label).
    """
    cond1 = df['historical_pincode_rto_rate'] > thresholds['hist_rate_threshold']
    cond2 = df['prior_rto_count'] >= thresholds['prior_rto_threshold']
    cond3 = (df['payment_method'] == 'COD') & \
            (df['account_age_days'] < thresholds['account_age_threshold'])

    preds = (cond1 | cond2 | cond3).astype(int).values
    return preds


def main():
    """Derive thresholds from train, predict on val and test, print summary."""
    train_df = pd.read_csv("data/processed/train.csv", dtype={'pincode': str})
    val_df = pd.read_csv("data/processed/val.csv", dtype={'pincode': str})
    test_df = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})

    # Derive thresholds from TRAIN only
    thresholds = derive_thresholds(train_df)
    print("Rule Baseline Thresholds (derived from TRAIN set only)")
    print("-" * 50)
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    # Predict
    for name, df in [("Val", val_df), ("Test", test_df)]:
        preds = predict_rules(df, thresholds)
        print(f"\n{name} Set: predicted {preds.sum()} RTO out of {len(preds)}")

    print("\nRule baseline predictions complete.")


if __name__ == "__main__":
    main()
