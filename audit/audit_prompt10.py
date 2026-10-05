import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, ".")
from src.serve.scorer import score_order, validate_and_clean, MODEL_VERSION
from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action
from src.policy.cost_engine import CostEngine

# 1. Equivalence: 1,000 random test orders
print("=== 1. SERVING PATH EQUIVALENCE (1,000 ORDERS) ===")
test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(test), size=1000, replace=False)
sample_test = test.iloc[sample_idx].reset_index(drop=True)

# Batch scoring via model
cal_model = joblib.load("models/tree_model_calibrated.pkl")
X_sample = sample_test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
p_batch = cal_model.predict_proba(X_sample)[:, 1]

# Pointwise serving scoring
diffs = []
action_mismatches = 0
engine = CostEngine()

for i, row in sample_test.iterrows():
    order_dict = row.to_dict()
    # Scorer expects raw incoming order without pipeline-derived pincode lookups
    order_dict.pop('historical_pincode_rto_rate', None)
    order_dict.pop('pincode_tier', None)
    order_dict.pop('rto_label', None)
    order_dict.pop('timestamp', None)
    order_dict.pop('order_id', None)
    
    res = score_order(order_dict)
    p_serve = res['probability']
    diff = abs(p_batch[i] - p_serve)
    diffs.append(diff)
    
    # Action check
    if row['payment_method'] == 'COD':
        losses = engine.evaluate_interventions(row['order_value'], p_batch[i])
        expected_act = min(losses, key=losses.get)
        if res['recommended_action'] != expected_act:
            action_mismatches += 1
    else:
        if res['recommended_action'] != "ALLOW_COD":
            action_mismatches += 1

max_diff = max(diffs)
print(f"Max absolute probability difference: {max_diff:.2e}")
print(f"Action mismatches: {action_mismatches} / 1,000")

# 2. Robustness inputs
print("\n=== 2. ROBUSTNESS INPUTS CHECK ===")
robust_cases = [
    ("Missing fields", {"order_value": 500.0}),
    ("Null/None values", {"order_value": None, "payment_method": "COD"}),
    ("Wrong types (str order_value)", {"order_value": "five hundred", "payment_method": "COD"}),
    ("Negative order value", {"order_value": -50.0, "category": "Apparel", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "560001", "courier_id": "Courier_A"}),
    ("Zero order value", {"order_value": 0.0, "category": "Apparel", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "560001", "courier_id": "Courier_A"}),
    ("Huge order value 1e9", {"order_value": 1e9, "category": "Apparel", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "560001", "courier_id": "Courier_A"}),
    ("Unseen pincode (cold-start)", {"order_value": 500.0, "category": "Apparel", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "999999", "courier_id": "Courier_A"}),
    ("Malformed pincode (5 digits)", {"order_value": 500.0, "category": "Apparel", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "12345", "courier_id": "Courier_A"}),
    ("Unseen category", {"order_value": 500.0, "category": "Automotive", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "560001", "courier_id": "Courier_A"}),
    ("Unicode / Emoji in category", {"order_value": 500.0, "category": "Apparel Fire", "payment_method": "COD", "quantity": 1, "discount_pct": 0, "cod_charge": 49, "account_age_days": 100, "prior_orders": 2, "prior_rto_count": 0, "orders_last_24h": 0, "device_cluster_size": 1, "pincode": "560001", "courier_id": "Courier_A"})
]

for name, payload in robust_cases:
    try:
        res = score_order(payload)
        print(f"Case '{name}': SUCCESS -> Action={res['recommended_action']}, P={res['probability']:.4f}, Warnings={res['warnings']}")
    except ValueError as ve:
        first_err = str(ve).split('\n')[1] if '\n' in str(ve) else str(ve)
        print(f"Case '{name}': REJECTED cleanly with ValueError -> {first_err[:70]}...")
    except Exception as e:
        print(f"Case '{name}': CRASHED with {type(e).__name__} -> {e}")

# 3. Determinism check: 100 calls
print("\n=== 3. DETERMINISM CHECK (100 CALLS) ===")
sample_order = sample_test.iloc[0].to_dict()
sample_order.pop('historical_pincode_rto_rate', None)
sample_order.pop('pincode_tier', None)
sample_order.pop('rto_label', None)
sample_order.pop('timestamp', None)
sample_order.pop('order_id', None)

first_res = score_order(sample_order)
deterministic = True
for _ in range(100):
    cur_res = score_order(sample_order)
    if cur_res['probability'] != first_res['probability'] or cur_res['recommended_action'] != first_res['recommended_action']:
        deterministic = False
        break
print(f"Repeated scorings identical: {deterministic}")

# 4. Pickle security & artifact hashes
print("\n=== 4. SECURITY & ARTIFACT HASHES ===")
with open("models/artifact_hashes.json", "r") as f:
    hashes = json.load(f)

for fn, expected_h in hashes.items():
    fp = os.path.join("models", fn)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            actual_h = hashlib.sha256(f.read()).hexdigest()
        match = actual_h == expected_h
        print(f"  {fn:28s}: {'MATCH' if match else 'MUTATED'} (hash={actual_h[:12]}...)")
    else:
        print(f"  {fn:28s}: MISSING")
