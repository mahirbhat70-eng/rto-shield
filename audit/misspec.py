"""
audit/misspec.py — Fixed-policy misspecification across ALL behavioural assumptions.

The policy (actions per order) is chosen ONCE under the config's assumptions and
frozen. We then evaluate realized savings when the TRUE world differs, one
parameter at a time, plus a joint "pessimistic" scenario. Break-even value for each
parameter is found by bisection.
Run: uv run python audit/misspec.py
"""
import copy
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit"))
os.chdir(ROOT)
from reconcile import realized_pl, frozen_actions, bisect_breakeven, CFG  # noqa: E402

SWEEPS = {
    ("REQUIRE_DEPOSIT", "rto_reduction_pct"): [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
    ("REQUIRE_DEPOSIT", "success_drop_pct"):  [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    ("VERIFY_ADDRESS", "rto_reduction_pct"):  [0.30, 0.20, 0.10, 0.05, 0.00],
    ("VERIFY_ADDRESS", "success_drop_pct"):   [0.05, 0.10, 0.15, 0.20, 0.30],
    ("VERIFY_ADDRESS", "friction_cost"):      [2, 5, 10, 20],
}


def load():
    test = pd.read_csv("data/processed/test.csv", dtype={"pincode": str})
    p = joblib.load("models/tree_model_calibrated.pkl").predict_proba(
        test.drop(columns=["rto_label", "timestamp", "order_id"], errors="ignore"))[:, 1]
    cod = (test["payment_method"] == "COD").values
    tc = test[cod].reset_index(drop=True)
    return frozen_actions(tc, p[cod]), tc["rto_label"].values, tc["order_value"].values


def with_param(action, key, val, base=CFG):
    c = copy.deepcopy(base)
    if action is None:
        c[key] = val
    else:
        c["interventions"][action][key] = val
    return c


def main():
    acts, y, V = load()
    print(f"frozen policy: n_COD={len(acts)}  design-point savings Rs {realized_pl(acts, y, V, CFG):,.2f}\n")
    summary = []
    for (a, k), grid in SWEEPS.items():
        print(f"--- TRUE {a}.{k} (policy assumed {CFG['interventions'][a][k]}) ---")
        for v in grid:
            s = realized_pl(acts, y, V, with_param(a, k, v))
            print(f"  {v:>6}: Rs {s:>12,.2f}  {'PROFIT' if s > 0 else 'LOSS'}")
        lo, hi = min(grid), max(grid)
        be = bisect_breakeven(lambda x: realized_pl(acts, y, V, with_param(a, k, x)), lo, hi * 1.0 if hi > lo else lo)
        summary.append((f"{a}.{k}", CFG["interventions"][a][k], be))
    for key, grid in [("rto_logistics_cost", [150, 120, 100, 80, 60]),
                      ("average_margin_pct", [0.20, 0.25, 0.30, 0.40])]:
        print(f"--- TRUE {key} (policy assumed {CFG[key]}) ---")
        for v in grid:
            s = realized_pl(acts, y, V, with_param(None, key, v))
            print(f"  {v:>6}: Rs {s:>12,.2f}  {'PROFIT' if s > 0 else 'LOSS'}")
        be = bisect_breakeven(lambda x: realized_pl(acts, y, V, with_param(None, key, x)), min(grid), max(grid))
        summary.append((key, CFG[key], be))

    # joint pessimistic scenario
    pess = copy.deepcopy(CFG)
    pess["interventions"]["REQUIRE_DEPOSIT"].update(rto_reduction_pct=0.60, success_drop_pct=0.50)
    pess["interventions"]["VERIFY_ADDRESS"].update(rto_reduction_pct=0.15, success_drop_pct=0.10)
    print(f"\nJOINT PESSIMISTIC (deposit red .60 / drop .50, verify red .15 / drop .10): "
          f"Rs {realized_pl(acts, y, V, pess):,.2f}")

    print("\nBREAK-EVEN (true value at which frozen policy savings hit 0):")
    for name, assumed, be in summary:
        print(f"  {name:<36} assumed={assumed:<6} break-even={'none in range' if be is None else f'{be:.4f}'}")


if __name__ == "__main__":
    main()
