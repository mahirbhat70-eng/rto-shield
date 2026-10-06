"""
audit/run_c1_realism.py — Phase C Data Realism Check.

Evaluates pipeline under 4 feature regimes:
(a) Current feature (latent pincode + N(0, 0.03))
(b) Noise sigma 0.10 added
(c) Noise sigma 0.20 added
(d) Realistic rolling empirical rate computed from strictly prior 30 days with shrinkage
"""

import os
import sys
import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit"))

from reconcile import realized_pl, frozen_actions, CFG
from src.models.tree_model import get_preprocessor
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

DROP_COLS = ["rto_label", "timestamp", "order_id", "customer_id"]

def build_tree_pipeline():
    return Pipeline([
        ('preprocessor', get_preprocessor()),
        ('classifier', LGBMClassifier(random_state=42, verbose=-1, n_estimators=300, learning_rate=0.05, num_leaves=31))
    ])


def train_calibrate_eval(train_df, val_cal_df, test_df, feature_name="historical_pincode_rto_rate"):
    # 1. Fit uncalibrated tree
    pipeline = build_tree_pipeline()
    X_train = train_df.drop(columns=DROP_COLS, errors="ignore")
    y_train = train_df["rto_label"].values
    pipeline.fit(X_train, y_train)

    # 2. Fit Isotonic Calibrator on val_cal
    X_val = val_cal_df.drop(columns=DROP_COLS, errors="ignore")
    y_val = val_cal_df["rto_label"].values
    p_val_uncal = pipeline.predict_proba(X_val)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_uncal, y_val)

    # 3. Predict on test
    X_test = test_df.drop(columns=DROP_COLS, errors="ignore")
    y_test = test_df["rto_label"].values
    p_test_uncal = pipeline.predict_proba(X_test)[:, 1]
    p_test_cal = iso.predict(p_test_uncal)

    # Statistical PR-AUC
    pr_auc = float(average_precision_score(y_test, p_test_cal))

    # COD subset savings
    cod_mask = (test_df["payment_method"] == "COD").values
    tc = test_df[cod_mask].reset_index(drop=True)
    yc = tc["rto_label"].values
    Vc = tc["order_value"].values
    pc = p_test_cal[cod_mask]

    # Model savings
    acts_model = frozen_actions(tc, pc)
    sav_model = float(realized_pl(acts_model, yc, Vc, CFG))

    # Pincode-only rule savings
    p_pin = tc[feature_name].values.astype(float)
    acts_pin = frozen_actions(tc, p_pin)
    sav_pin = float(realized_pl(acts_pin, yc, Vc, CFG))

    return {
        "pr_auc": pr_auc,
        "model_savings_Rs": sav_model,
        "pincode_rule_savings_Rs": sav_pin,
        "model_vs_pincode_pct": ((sav_model - sav_pin) / sav_pin) * 100.0 if sav_pin != 0 else 0.0,
        "action_mix": pd.Series(acts_model).value_counts().to_dict()
    }


def compute_rolling_30d_feature(full_df, shrinkage_m=10.0):
    """
    Computes strictly prior 30-day rolling empirical rate per pincode.
    Uses closed='left' so the current order is NEVER counted.
    Shrinkage toward global training mean rate.
    """
    df = full_df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp_dt", "order_id"]).reset_index(drop=True)
    
    # Global training rate for shrinkage (from first 70k orders)
    train_slice = df.iloc[:70000]
    global_rate = float(train_slice["rto_label"].mean())

    # Set timestamp index for rolling group
    df_indexed = df.set_index("timestamp_dt")
    
    # Rolling 30 days strictly prior (closed='left')
    grouped = df_indexed.groupby("pincode")["rto_label"]
    roll_counts = grouped.rolling("30D", closed="left").count().fillna(0).values
    roll_sums = grouped.rolling("30D", closed="left").sum().fillna(0).values

    # m-estimate shrinkage
    empirical_rates = (roll_sums + shrinkage_m * global_rate) / (roll_counts + shrinkage_m)
    
    # Put back to original dataframe order
    # Note: groupby.rolling returns grouped by pincode order, so map by (pincode, timestamp_dt)
    roll_df = df_indexed.groupby("pincode")["rto_label"].rolling("30D", closed="left").agg(["count", "sum"]).reset_index()
    roll_df["count"] = roll_df["count"].fillna(0)
    roll_df["sum"] = roll_df["sum"].fillna(0)
    roll_df["rolling_rate"] = (roll_df["sum"] + shrinkage_m * global_rate) / (roll_df["count"] + shrinkage_m)

    # Merge back to df to ensure exact row correspondence
    merged = df.merge(
        roll_df[["pincode", "timestamp_dt", "rolling_rate"]],
        on=["pincode", "timestamp_dt"],
        how="left"
    )
    # Deduplicate in case of duplicate timestamps in same pincode
    merged = merged.drop_duplicates(subset=["order_id"]).sort_values("timestamp_dt").reset_index(drop=True)
    return merged["rolling_rate"].values


def main():
    print("Loading data splits...")
    train = pd.read_csv(os.path.join(ROOT, "data/processed/train.csv"), dtype={"pincode": str})
    val_cal = pd.read_csv(os.path.join(ROOT, "data/processed/val_cal.csv"), dtype={"pincode": str})
    test = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv"), dtype={"pincode": str})

    # (a) Current feature
    print("\nEvaluating (a) Current feature (latent + N(0, 0.03))...")
    res_a = train_calibrate_eval(train, val_cal, test)

    # (b) Noise sigma 0.10
    print("Evaluating (b) Noise sigma 0.10...")
    rng = np.random.default_rng(42)
    train_b = train.copy()
    val_cal_b = val_cal.copy()
    test_b = test.copy()
    for d in [train_b, val_cal_b, test_b]:
        noise = rng.normal(0, 0.10, size=len(d))
        d["historical_pincode_rto_rate"] = np.clip(d["historical_pincode_rto_rate"] + noise, 0.0, 1.0)
    res_b = train_calibrate_eval(train_b, val_cal_b, test_b)

    # (c) Noise sigma 0.20
    print("Evaluating (c) Noise sigma 0.20...")
    train_c = train.copy()
    val_cal_c = val_cal.copy()
    test_c = test.copy()
    for d in [train_c, val_cal_c, test_c]:
        noise = rng.normal(0, 0.20, size=len(d))
        d["historical_pincode_rto_rate"] = np.clip(d["historical_pincode_rto_rate"] + noise, 0.0, 1.0)
    res_c = train_calibrate_eval(train_c, val_cal_c, test_c)

    # (d) Realistic rolling 30-day empirical rate
    print("Computing (d) Realistic 30-day rolling empirical rate from strictly prior orders...")
    full_raw = pd.read_csv(os.path.join(ROOT, "data/raw/synthetic_orders.csv"), dtype={"pincode": str})
    rolling_rates = compute_rolling_30d_feature(full_raw, shrinkage_m=10.0)
    full_raw["historical_pincode_rto_rate"] = rolling_rates

    train_d = full_raw.iloc[:len(train)].copy().reset_index(drop=True)
    val_cal_d = full_raw.iloc[len(train):len(train)+len(val_cal)].copy().reset_index(drop=True)
    # Test is the last 14,980 rows
    test_d = full_raw.iloc[-len(test):].copy().reset_index(drop=True)

    print("Evaluating (d) Realistic rolling feature regime...")
    res_d = train_calibrate_eval(train_d, val_cal_d, test_d)

    results = [
        {"Regime": "(a) Current feature (latent + N(0, 0.03))", "PR-AUC": res_a["pr_auc"],
         "Model Savings (Rs)": res_a["model_savings_Rs"], "Pincode Rule (Rs)": res_a["pincode_rule_savings_Rs"],
         "Model Lift over Pincode": f"+{res_a['model_vs_pincode_pct']:.1f}%"},
        {"Regime": "(b) Noise sigma 0.10", "PR-AUC": res_b["pr_auc"],
         "Model Savings (Rs)": res_b["model_savings_Rs"], "Pincode Rule (Rs)": res_b["pincode_rule_savings_Rs"],
         "Model Lift over Pincode": f"+{res_b['model_vs_pincode_pct']:.1f}%"},
        {"Regime": "(c) Noise sigma 0.20", "PR-AUC": res_c["pr_auc"],
         "Model Savings (Rs)": res_c["model_savings_Rs"], "Pincode Rule (Rs)": res_c["pincode_rule_savings_Rs"],
         "Model Lift over Pincode": f"+{res_c['model_vs_pincode_pct']:.1f}%"},
        {"Regime": "(d) Realistic rolling 30D empirical rate", "PR-AUC": res_d["pr_auc"],
         "Model Savings (Rs)": res_d["model_savings_Rs"], "Pincode Rule (Rs)": res_d["pincode_rule_savings_Rs"],
         "Model Lift over Pincode": f"+{res_d['model_vs_pincode_pct']:.1f}%"},
    ]

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print("PHASE C: DATA REALISM COMPARISON (HOLD-OUT TEST SPLIT)")
    print("=" * 95)
    print(df_res.to_string(index=False, float_format=lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v)))
    print("=" * 95)

    headline = 69786.08
    surv_pct = (res_d["model_savings_Rs"] / headline) * 100.0
    print(f"\nSavings surviving in realistic regime (d): Rs {res_d['model_savings_Rs']:,.2f} / Rs {headline:,.2f} = {surv_pct:.1f}%")
    if res_d['model_savings_Rs'] < headline:
        print(f"CONCLUSION: The headline Rs 69,786 IS AN UPPER BOUND enabled by oracle latent pincode knowledge.")
    else:
        print(f"CONCLUSION: The headline Rs 69,786 is robust even under realistic empirical rolling features.")

if __name__ == "__main__":
    main()
