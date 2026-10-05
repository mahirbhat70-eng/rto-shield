import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")

print("="*80)
print("PROMPT 5 AUDIT: DATA LEAKAGE AUDIT")
print("="*80)

# Load splits
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), dtype={'pincode': str})
val_cal_df = pd.read_csv(os.path.join(DATA_DIR, "val_cal.csv"), dtype={'pincode': str})
val_rep_df = pd.read_csv(os.path.join(DATA_DIR, "val_rep.csv"), dtype={'pincode': str})
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), dtype={'pincode': str})

print(f"Loaded splits: train={len(train_df)}, val_cal={len(val_cal_df)}, val_rep={len(val_rep_df)}, test={len(test_df)}")

# Task 4: Split timestamp monotonicity
ts_train_max = pd.to_datetime(train_df['timestamp']).max()
ts_val_cal_min = pd.to_datetime(val_cal_df['timestamp']).min()
ts_val_cal_max = pd.to_datetime(val_cal_df['timestamp']).max()
ts_val_rep_min = pd.to_datetime(val_rep_df['timestamp']).min()
ts_val_rep_max = pd.to_datetime(val_rep_df['timestamp']).max()
ts_test_min = pd.to_datetime(test_df['timestamp']).min()

print("\n--- TASK 4 & 5: SPLIT MONOTONICITY & CALIBRATION SPLITS ---")
print(f"Train max timestamp:    {ts_train_max}")
print(f"Val_cal min timestamp:  {ts_val_cal_min} (Train < Val_cal? {ts_train_max < ts_val_cal_min})")
print(f"Val_cal max timestamp:  {ts_val_cal_max}")
print(f"Val_rep min timestamp:  {ts_val_rep_min} (Val_cal < Val_rep? {ts_val_cal_max < ts_val_rep_min})")
print(f"Val_rep max timestamp:  {ts_val_rep_max}")
print(f"Test min timestamp:     {ts_test_min} (Val_rep < Test? {ts_val_rep_max < ts_test_min})")

# Overlap checks
train_custs = set(train_df['customer_id'])
val_custs = set(val_cal_df['customer_id']).union(set(val_rep_df['customer_id']))
test_custs = set(test_df['customer_id'])

print(f"\nUnique customers: Train={len(train_custs)}, Val={len(val_custs)}, Test={len(test_custs)}")
print(f"Customer overlap Train & Test: {len(train_custs.intersection(test_custs))} customers (expected in repeat-buyer e-commerce)")
print(f"Duplicate order_ids across splits: {len(set(train_df['order_id']).intersection(set(test_df['order_id'])))}")

# Task 7: "Too good to be true" single feature PR-AUC
print("\n--- TASK 7: SINGLE-FEATURE PR-AUC AUDIT ---")
num_cols = [
    'order_value', 'quantity', 'discount_pct', 'cod_charge',
    'account_age_days', 'prior_orders', 'prior_rto_count',
    'historical_pincode_rto_rate', 'orders_last_24h', 'device_cluster_size'
]
cat_cols = ['category', 'payment_method', 'courier_id', 'pincode_tier']

y_train = train_df['rto_label'].values
y_test = test_df['rto_label'].values
base_rate = np.mean(y_test)
print(f"Test Set Base RTO Rate (baseline PR-AUC): {base_rate:.4f}\n")

single_feature_results = []
for col in num_cols + cat_cols:
    if col in num_cols:
        ct = ColumnTransformer([('num', StandardScaler(), [col])])
    else:
        ct = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), [col])])
    pipe = Pipeline([('prep', ct), ('clf', LogisticRegression(max_iter=500))])
    pipe.fit(train_df[[col]], y_train)
    proba = pipe.predict_proba(test_df[[col]])[:, 1]
    score = average_precision_score(y_test, proba)
    is_suspicious = score > 0.60
    single_feature_results.append((col, score, "LEAK SUSPECT (>0.60)" if is_suspicious else "SAFE"))

print(f"{'Feature':<30} | {'PR-AUC (alone)':<15} | {'Verdict'}")
print("-" * 55)
for col, score, v in sorted(single_feature_results, key=lambda x: x[1], reverse=True):
    print(f"{col:<30} | {score:<15.4f} | {v}")

# Task 8: Label-shuffle check
print("\n--- TASK 8: LABEL-SHUFFLE CHECK ---")
rng = np.random.default_rng(42)
y_train_shuffled = rng.permutation(y_train)

ct = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
pipe_shuffle = Pipeline([('prep', ct), ('clf', LogisticRegression(max_iter=1000))])

X_train = train_df[num_cols + cat_cols]
X_test = test_df[num_cols + cat_cols]

pipe_shuffle.fit(X_train, y_train_shuffled)
proba_shuffled = pipe_shuffle.predict_proba(X_test)[:, 1]
score_shuffled = average_precision_score(y_test, proba_shuffled)

# COD subset score
cod_mask = (test_df['payment_method'] == 'COD').values
cod_base_rate = np.mean(y_test[cod_mask])
score_shuffled_cod = average_precision_score(y_test[cod_mask], proba_shuffled[cod_mask])

print(f"Overall Test PR-AUC with shuffled y_train:     {score_shuffled:.4f} (Base rate: {base_rate:.4f})")
print(f"COD Subset Test PR-AUC with shuffled y_train:   {score_shuffled_cod:.4f} (COD Base rate: {cod_base_rate:.4f})")
if abs(score_shuffled - base_rate) < 0.02:
    print("RESULT: PASS. Model learns zero signal when labels are shuffled (PR-AUC collapses to base rate).")
else:
    print("RESULT: FAIL. Model still has predictive power with shuffled labels!")
