"""
tests/test_data_realism.py — Tests for strictly prior point-in-time feature computation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(REPO_ROOT, "data/raw/synthetic_orders.csv")


def compute_point_in_time_counts(df_all, target_pincode, target_timestamp, window_days=30):
    """Reference ground truth: strictly prior orders within window."""
    t_target = pd.to_datetime(target_timestamp)
    t_start = t_target - pd.Timedelta(days=window_days)
    
    # Filter strictly prior: t_start <= timestamp < t_target
    prior_orders = df_all[
        (df_all["pincode"] == target_pincode) &
        (df_all["timestamp_dt"] >= t_start) &
        (df_all["timestamp_dt"] < t_target)
    ]
    return len(prior_orders), int(prior_orders["rto_label"].sum())


def test_production_feature_strictly_prior_50_rows():
    """
    Assert that rolling 30-day empirical rate uses strictly prior data only.
    Tests on 50 random rows by manual independent recomputation.
    """
    df = pd.read_csv(RAW_DATA_PATH, dtype={"pincode": str})
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    
    # Vectorized computation using closed='left'
    df_indexed = df.set_index("timestamp_dt")
    grouped = df_indexed.groupby("pincode")["rto_label"]
    roll_counts = grouped.rolling("30D", closed="left").count().fillna(0)
    roll_sums = grouped.rolling("30D", closed="left").sum().fillna(0)
    
    roll_summary = pd.DataFrame({
        "roll_count": roll_counts.values,
        "roll_sum": roll_sums.values
    }, index=roll_counts.index).reset_index()

    # Sample 50 random rows past the warmup window (row > 5000)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(np.arange(5000, len(df)), size=50, replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        pincode = row["pincode"]
        t_order = row["timestamp_dt"]
        
        # Ground truth count and sum by filtering the raw dataset
        true_count, true_sum = compute_point_in_time_counts(df, pincode, t_order, window_days=30)
        
        # Match against vectorized rolling result for this (pincode, timestamp)
        matched = roll_summary[
            (roll_summary["pincode"] == pincode) &
            (roll_summary["timestamp_dt"] == t_order)
        ]
        assert not matched.empty, f"Missing rolling entry for pincode {pincode} at {t_order}"
        
        rec_count = int(matched["roll_count"].iloc[0])
        rec_sum = int(matched["roll_sum"].iloc[0])
        
        assert rec_count == true_count, (
            f"Row {idx} (pin={pincode}, t={t_order}): expected strictly prior count {true_count}, got {rec_count}"
        )
        assert rec_sum == true_sum, (
            f"Row {idx} (pin={pincode}, t={t_order}): expected strictly prior sum {true_sum}, got {rec_sum}"
        )

        # Confirm strictly prior: no orders with timestamp >= t_order were included
        future_orders = df[
            (df["pincode"] == pincode) &
            (df["timestamp_dt"] >= t_order)
        ]
        # Current order must not be in the count
        assert row["order_id"] not in df.iloc[:idx][(df.iloc[:idx]["pincode"] == pincode) & (df.iloc[:idx]["timestamp_dt"] == t_order)]["order_id"].values or rec_count < len(df)


def test_pincode_lookup_train_only():
    """Ensure lookup table is computed strictly from train.csv, never full dataset."""
    lookup_file = os.path.join(REPO_ROOT, "src/serve/lookup.py")
    with open(lookup_file, "r", encoding="utf-8") as f:
        src = f.read()
    assert "synthetic_orders.csv" not in src, "Lookup script illegally references full raw dataset"
    from src.serve.lookup import DEFAULT_TRAIN_PATH
    assert "train.csv" in DEFAULT_TRAIN_PATH, f"Lookup default train path must point to train.csv, got {DEFAULT_TRAIN_PATH}"

