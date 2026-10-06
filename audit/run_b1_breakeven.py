"""
audit/run_b1_breakeven.py — Precise bisection of break-even RTO rates.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.policy.cost_engine import CostEngine
from audit.reconcile import realized_pl, expected_pl, bisect_breakeven

CFG = yaml.safe_load(open(os.path.join(ROOT, "configs/cost_config.yaml")))
DROP = ["rto_label", "timestamp", "order_id"]

test = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv"), dtype={"pincode": str})
cod_mask = (test["payment_method"] == "COD").values
tc = test[cod_mask].reset_index(drop=True)
yc = tc["rto_label"].values
Vc = tc["order_value"].values

import joblib
model = joblib.load(os.path.join(ROOT, "models/tree_model_calibrated.pkl"))
X_test = test.drop(columns=DROP, errors="ignore")
p_cal = model.predict_proba(X_test)[:, 1]
pc = p_cal[cod_mask]

eng = CostEngine(os.path.join(ROOT, "configs/cost_config.yaml"))
acts, _ = eng.get_optimal_policy(tc.assign(payment_method="COD"), pc)
acts = np.asarray(acts)

# 1. COD RTO sweep (Method A: label resampling)
pos_cod = np.where(yc == 1)[0]
neg_cod = np.where(yc == 0)[0]

def cod_resampled_savings(r, reps=50):
    out = []
    for s in range(reps):
        rng = np.random.default_rng(s)
        n_pos = int(round(len(neg_cod) * r / (1.0 - r)))
        ip = rng.choice(pos_cod, n_pos, replace=True)
        idx = np.concatenate([neg_cod, ip])
        out.append(realized_pl(acts[idx], yc[idx], Vc[idx], CFG))
    return float(np.mean(out))

be_cod_a = bisect_breakeven(lambda r: cod_resampled_savings(r, reps=30), 0.10, 0.25, tol=0.0005)

# 2. Overall platform RTO sweep
pos_all = np.where(test["rto_label"] == 1)[0]
neg_all = np.where(test["rto_label"] == 0)[0]

# Precompute actions for all test rows (only COD orders get actions; others are None)
all_acts = np.full(len(test), None, dtype=object)
all_acts[cod_mask] = acts

def overall_resampled_savings(r_all, reps=30):
    out = []
    for s in range(reps):
        rng = np.random.default_rng(s)
        n_pos = int(round(len(neg_all) * r_all / (1.0 - r_all)))
        ip = rng.choice(pos_all, n_pos, replace=True)
        idx = np.concatenate([neg_all, ip])
        
        # Filter to COD orders within the resampled population
        idx_cod = [i for i in idx if cod_mask[i]]
        if not idx_cod:
            out.append(0.0)
            continue
        idx_cod = np.array(idx_cod)
        cod_acts = all_acts[idx_cod]
        cod_y = test["rto_label"].values[idx_cod]
        cod_v = test["order_value"].values[idx_cod]
        out.append(realized_pl(cod_acts, cod_y, cod_v, CFG))
    return float(np.mean(out))

for test_r in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
    sav = overall_resampled_savings(test_r, reps=10)
    print(f"Overall RTO {test_r*100:4.1f}% -> Savings Rs {sav:10.2f}")

be_overall = bisect_breakeven(lambda r: overall_resampled_savings(r, reps=30), 0.08, 0.16, tol=0.0005)

print(f"EXACT COD RTO Break-even Crossover: {be_cod_a * 100:.1f}% ({be_cod_a:.4f})")
print(f"EXACT Overall Platform Break-even Crossover: {be_overall * 100:.1f}% ({be_overall:.4f})")
