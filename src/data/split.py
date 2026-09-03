"""
RTO Shield — Stage 2: Temporal Train/Val/Test Split

Splits data/raw/synthetic_orders.csv by TIMESTAMP (not row count):
  - Train: first 70% of the TIME RANGE
  - Validation: next 15% of the TIME RANGE
  - Test: final 15% of the TIME RANGE

Cutoffs are calendar-based — row counts per split may not be exactly 70/15/15.
No shuffling. No sklearn train_test_split. No leakage across splits.
"""

import os
import argparse
import pandas as pd


def temporal_split(input_path="data/raw/synthetic_orders.csv",
                   output_dir="data/processed"):
    """
    Split CSV by timestamp into train/val/test based on time range percentages.
    Returns (train_df, val_df, test_df) and prints summary.
    """
    df = pd.read_csv(input_path, dtype={'pincode': str})
    df['_ts'] = pd.to_datetime(df['timestamp'])

    ts_min = df['_ts'].min()
    ts_max = df['_ts'].max()
    total_seconds = (ts_max - ts_min).total_seconds()

    # Cutoff timestamps: 70% / 15% / 15% of the time range
    train_cutoff = ts_min + pd.Timedelta(seconds=total_seconds * 0.70)
    val_cutoff = ts_min + pd.Timedelta(seconds=total_seconds * 0.85)

    train_df = df[df['_ts'] <= train_cutoff].copy()
    
    val_cal_cutoff = train_cutoff + (val_cutoff - train_cutoff) / 2
    val_cal_df = df[(df['_ts'] > train_cutoff) & (df['_ts'] <= val_cal_cutoff)].copy()
    val_rep_df = df[(df['_ts'] > val_cal_cutoff) & (df['_ts'] <= val_cutoff)].copy()
    
    test_df = df[df['_ts'] > val_cutoff].copy()

    # Drop helper column
    for split_df in [train_df, val_cal_df, val_rep_df, test_df]:
        split_df.drop(columns=['_ts'], inplace=True)

    # ── Assertions ────────────────────────────────────────────────
    assert len(train_df) + len(val_cal_df) + len(val_rep_df) + len(test_df) == len(df), \
        "Row count mismatch after split"

    # No temporal leakage: max(train) < min(val_cal) < min(val_rep) < min(test)
    train_ts_max = pd.to_datetime(train_df['timestamp']).max()
    val_cal_ts_min = pd.to_datetime(val_cal_df['timestamp']).min()
    val_cal_ts_max = pd.to_datetime(val_cal_df['timestamp']).max()
    val_rep_ts_min = pd.to_datetime(val_rep_df['timestamp']).min()
    val_rep_ts_max = pd.to_datetime(val_rep_df['timestamp']).max()
    test_ts_min = pd.to_datetime(test_df['timestamp']).min()

    assert train_ts_max < val_cal_ts_min, \
        f"Temporal leakage: train max {train_ts_max} >= val_cal min {val_cal_ts_min}"
    assert val_cal_ts_max < val_rep_ts_min, \
        f"Temporal leakage: val_cal max {val_cal_ts_max} >= val_rep min {val_rep_ts_min}"
    assert val_rep_ts_max < test_ts_min, \
        f"Temporal leakage: val_rep max {val_rep_ts_max} >= test min {test_ts_min}"

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_cal_df.to_csv(os.path.join(output_dir, "val_cal.csv"), index=False)
    val_rep_df.to_csv(os.path.join(output_dir, "val_rep.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────
    print("Temporal Split Summary")
    print("=" * 60)
    print(f"Source: {input_path}  ({len(df):,} rows)")
    print(f"Time range: {ts_min} -> {ts_max}")
    print(f"Train cutoff:  {train_cutoff}")
    print(f"Val cutoff:    {val_cutoff}")
    print()

    for name, sdf in [("Train", train_df), ("Val_Cal", val_cal_df), ("Val_Rep", val_rep_df), ("Test", test_df)]:
        sts = pd.to_datetime(sdf['timestamp'])
        rto_rate = sdf['rto_label'].mean()
        print(f"  {name:7s}: {len(sdf):>6,} rows | "
              f"{sts.min()} -> {sts.max()} | "
              f"RTO rate: {rto_rate:.4f} ({rto_rate*100:.1f}%)")

    print()
    print(f"Saved to: {output_dir}/{{train,val_cal,val_rep,test}}.csv")
    print("Temporal leakage check: PASSED")

    return train_df, val_cal_df, val_rep_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal split for RTO Shield")
    parser.add_argument("--input", default="data/raw/synthetic_orders.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    temporal_split(input_path=args.input, output_dir=args.output_dir)
