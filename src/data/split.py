"""
src/data/split.py
WHY: Chronological train/val/test split to prevent data leakage.
     The test window must represent the 'future' — any time-based aggregation
     (pincode RTO rates, customer history) must be computed ONLY from rows
     whose timestamps precede the observation.

     Split ratios: 64% train / 16% val / 20% test (by row count after sorting).
     We also produce two val subsets:
       - val_cal: used to fit isotonic calibration (first half of val)
       - val_rep: used for policy evaluation (second half of val)
     This prevents calibration from contaminating the evaluation signal.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROCESSED_DIR = "data/processed"

TRAIN_FRAC = 0.64
VAL_FRAC   = 0.16
# test_frac = 1 - TRAIN_FRAC - VAL_FRAC = 0.20


def split(input_path: str = "data/raw/synthetic_orders.csv") -> dict:
    """
    Load synthetic orders, sort chronologically, and produce splits.

    Returns:
        dict with keys: train, val, val_cal, val_rep, test — each a DataFrame.
    """
    df = pd.read_csv(input_path, parse_dates=["order_date"], dtype={"pincode": str})
    df = df.sort_values("order_date").reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train  = df.iloc[:n_train].copy()
    val    = df.iloc[n_train : n_train + n_val].copy()
    test   = df.iloc[n_train + n_val :].copy()

    # Split val further for calibration vs representative evaluation
    mid = len(val) // 2
    val_cal = val.iloc[:mid].copy()
    val_rep = val.iloc[mid:].copy()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(f"{PROCESSED_DIR}/train.csv",   index=False)
    val.to_csv(  f"{PROCESSED_DIR}/val.csv",     index=False)
    val_cal.to_csv(f"{PROCESSED_DIR}/val_cal.csv", index=False)
    val_rep.to_csv(f"{PROCESSED_DIR}/val_rep.csv", index=False)
    test.to_csv( f"{PROCESSED_DIR}/test.csv",    index=False)

    print(f"Split complete  (total={n:,})")
    print(f"  train  : {len(train):,}  rows  ({len(train)/n:.1%})")
    print(f"  val    : {len(val):,}   rows  ({len(val)/n:.1%})")
    print(f"    val_cal : {len(val_cal):,}")
    print(f"    val_rep : {len(val_rep):,}")
    print(f"  test   : {len(test):,}   rows  ({len(test)/n:.1%})")
    print(f"  date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")

    return dict(train=train, val=val, val_cal=val_cal, val_rep=val_rep, test=test)


if __name__ == "__main__":
    split()
