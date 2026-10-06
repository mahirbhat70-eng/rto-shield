"""
audit/seed_sweep.py — 5-seed sweep of the FULL pipeline (generate -> split -> lookup ->
LR -> LGBM grid -> isotonic & Platt), each seed in an isolated temp copy of the repo.

Per seed reports: model PR-AUCs, OBSERVABLE Bayes ceiling E[p|x], % of ceiling,
and realized COD savings for:
- Logistic Regression
- Uncalibrated LightGBM
- Platt-calibrated LightGBM
- Isotonic-calibrated LightGBM (shipped primary)

Run: python audit/seed_sweep.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit"))
from reconcile import realized_pl, frozen_actions, CFG  # noqa: E402
from src.eval.bayes_ceiling import get_true_p  # noqa: E402

SEEDS = [42, 101, 2024, 777, 999]
IGNORE = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "audit")
DROP = ["rto_label", "timestamp", "order_id"]


def build(seed):
    work = tempfile.mkdtemp(prefix=f"rto_seed{seed}_")
    shutil.copytree(ROOT, work, ignore=IGNORE, dirs_exist_ok=True)
    env = dict(os.environ, PYTHONPATH=work, PYTHONHASHSEED="0")
    chain = [["src/data/generator.py", "--seed", str(seed)], ["src/data/split.py"],
             ["src/serve/lookup.py"], ["src/models/logistic_baseline.py"],
             ["src/models/tree_model.py"], ["src/eval/calibration.py"]]
    for c in chain:
        r = subprocess.run([sys.executable] + c, cwd=work, env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"seed {seed} failed at {c[0]}: {r.stderr[-800:]}")
    return work


def evaluate(work, seed):
    test = pd.read_csv(os.path.join(work, "data/processed/test.csv"), dtype={"pincode": str})
    val_cal = pd.read_csv(os.path.join(work, "data/processed/val_cal.csv"), dtype={"pincode": str})

    X_test = test.drop(columns=DROP, errors="ignore")
    y_test = test["rto_label"].values

    X_val_cal = val_cal.drop(columns=DROP, errors="ignore")
    y_val_cal = val_cal["rto_label"].values

    lr = joblib.load(os.path.join(work, "models/logistic_baseline.pkl"))
    tree_uncal = joblib.load(os.path.join(work, "models/tree_model.pkl"))
    tree_iso = joblib.load(os.path.join(work, "models/tree_model_calibrated.pkl"))

    # Platt calibration
    p_val_uncal = tree_uncal.predict_proba(X_val_cal)[:, 1]
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(p_val_uncal.reshape(-1, 1), y_val_cal)

    p_lr = lr.predict_proba(X_test)[:, 1]
    p_uncal = tree_uncal.predict_proba(X_test)[:, 1]
    p_iso = tree_iso.predict_proba(X_test)[:, 1]
    p_platt = platt.predict_proba(p_uncal.reshape(-1, 1))[:, 1]

    cod = (test["payment_method"] == "COD").values
    tc = test[cod].reset_index(drop=True)

    from reconcile import expected_pl

    acts_iso = frozen_actions(tc, p_iso[cod])
    sav_iso = realized_pl(acts_iso, tc["rto_label"].values, tc["order_value"].values, CFG)
    exp_iso = expected_pl(acts_iso, p_iso[cod], tc["order_value"].values, CFG)

    acts_uncal = frozen_actions(tc, p_uncal[cod])
    sav_uncal = realized_pl(acts_uncal, tc["rto_label"].values, tc["order_value"].values, CFG)
    exp_uncal = expected_pl(acts_uncal, p_uncal[cod], tc["order_value"].values, CFG)

    acts_platt = frozen_actions(tc, p_platt[cod])
    sav_platt = realized_pl(acts_platt, tc["rto_label"].values, tc["order_value"].values, CFG)
    exp_platt = expected_pl(acts_platt, p_platt[cod], tc["order_value"].values, CFG)

    acts_lr = frozen_actions(tc, p_lr[cod])
    sav_lr = realized_pl(acts_lr, tc["rto_label"].values, tc["order_value"].values, CFG)
    exp_lr = expected_pl(acts_lr, p_lr[cod], tc["order_value"].values, CFG)

    ceil = average_precision_score(y_test, get_true_p(test))

    return {
        "seed": seed,
        "test_rto": y_test.mean(),
        "cod_rto": tc["rto_label"].mean(),
        "n_cod": int(cod.sum()),
        "ceiling_PR": ceil,
        "LR_PR": average_precision_score(y_test, p_lr),
        "LGBM_uncal_PR": average_precision_score(y_test, p_uncal),
        "LGBM_platt_PR": average_precision_score(y_test, p_platt),
        "LGBM_iso_PR": average_precision_score(y_test, p_iso),
        "LR_expected_Rs": exp_lr,
        "LR_realized_Rs": sav_lr,
        "LR_gap_pct": ((exp_lr - sav_lr) / sav_lr) * 100.0,
        "LGBM_uncal_expected_Rs": exp_uncal,
        "LGBM_uncal_realized_Rs": sav_uncal,
        "LGBM_uncal_gap_pct": ((exp_uncal - sav_uncal) / sav_uncal) * 100.0,
        "LGBM_platt_expected_Rs": exp_platt,
        "LGBM_platt_realized_Rs": sav_platt,
        "LGBM_platt_gap_pct": ((exp_platt - sav_platt) / sav_platt) * 100.0,
        "LGBM_iso_expected_Rs": exp_iso,
        "LGBM_iso_realized_Rs": sav_iso,
        "LGBM_iso_gap_pct": ((exp_iso - sav_iso) / sav_iso) * 100.0,
    }


def main():
    rows = []
    for s in SEEDS:
        w = build(s)
        rows.append(evaluate(w, s))
        shutil.rmtree(w, ignore_errors=True)
        print(f"seed {s}: done", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nmean / std / min / max:")
    print(df.drop(columns=["seed"]).agg(["mean", "std", "min", "max"]).to_string(float_format=lambda v: f"{v:.4f}"))
    df.to_csv(os.path.join(ROOT, "audit", "seed_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
