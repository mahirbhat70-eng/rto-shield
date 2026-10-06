"""
audit/run_b5_misspecification.py — 108-cell joint misspecification grid with policy frozen.
"""

import copy
import os
import sys
import itertools
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit"))

from reconcile import realized_pl, frozen_actions, CFG

# Grid parameters
DEP_REDUCTIONS = [0.80, 0.60, 0.40, 0.20]
DEP_DROPS = [0.40, 0.60, 0.80]
VER_REDUCTIONS = [0.30, 0.15, 0.00]
VER_DROPS = [0.05, 0.10, 0.20]

def main():
    test = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv"), dtype={"pincode": str})
    cod_mask = (test["payment_method"] == "COD").values
    tc = test[cod_mask].reset_index(drop=True)
    y_cod = tc["rto_label"].values
    V_cod = tc["order_value"].values

    # Frozen primary policy actions
    model = joblib.load(os.path.join(ROOT, "models/tree_model_calibrated.pkl"))
    drop_cols = ["rto_label", "timestamp", "order_id"]
    p_model = model.predict_proba(test.drop(columns=drop_cols, errors="ignore"))[:, 1][cod_mask]
    acts_model = frozen_actions(tc, p_model)

    # Pincode-only rule actions
    p_pin = tc["historical_pincode_rto_rate"].values.astype(float)
    acts_pin = frozen_actions(tc, p_pin)

    rows = []
    for dep_red, dep_dr, ver_red, ver_dr in itertools.product(DEP_REDUCTIONS, DEP_DROPS, VER_REDUCTIONS, VER_DROPS):
        cell_cfg = copy.deepcopy(CFG)
        cell_cfg["interventions"]["REQUIRE_DEPOSIT"]["rto_reduction_pct"] = dep_red
        cell_cfg["interventions"]["REQUIRE_DEPOSIT"]["success_drop_pct"] = dep_dr
        cell_cfg["interventions"]["VERIFY_ADDRESS"]["rto_reduction_pct"] = ver_red
        cell_cfg["interventions"]["VERIFY_ADDRESS"]["success_drop_pct"] = ver_dr

        sav_model = realized_pl(acts_model, y_cod, V_cod, cell_cfg)
        sav_pin = realized_pl(acts_pin, y_cod, V_cod, cell_cfg)

        beats_allow = (sav_model > 0.0)
        beats_pincode = (sav_model > sav_pin)

        rows.append({
            "dep_reduction": dep_red,
            "dep_drop": dep_dr,
            "ver_reduction": ver_red,
            "ver_drop": ver_dr,
            "policy_realized_savings_Rs": round(sav_model, 2),
            "pincode_realized_savings_Rs": round(sav_pin, 2),
            "beats_always_allow": beats_allow,
            "beats_pincode_rule": beats_pincode,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, "audit/misspecification_grid.csv")
    df.to_csv(csv_path, index=False)

    total_cells = len(df)
    n_beats_allow = int(df["beats_always_allow"].sum())
    n_beats_pin = int(df["beats_pincode_rule"].sum())
    frac_allow = n_beats_allow / total_cells
    frac_pin = n_beats_pin / total_cells

    print("=" * 80)
    print("JOINT MISSPECIFICATION GRID EVALUATION (108 CELLS)")
    print("=" * 80)
    print(f"Total Grid Cells:                  {total_cells}")
    print(f"Cells beating Always-Allow:        {n_beats_allow} / {total_cells} ({frac_allow*100:.1f}%)")
    print(f"Cells beating Pincode-Only Rule:   {n_beats_pin} / {total_cells} ({frac_pin*100:.1f}%)")
    print(f"Saved grid to:                     {csv_path}")
    print("=" * 80)

    # Show worst and best cells
    print("\nTop 3 Worst Cells (Extreme Degradation):")
    print(df.sort_values("policy_realized_savings_Rs").head(3).to_string(index=False))
    print("\nTop 3 Best Cells (Optimistic Reality):")
    print(df.sort_values("policy_realized_savings_Rs", ascending=False).head(3).to_string(index=False))

if __name__ == "__main__":
    main()
