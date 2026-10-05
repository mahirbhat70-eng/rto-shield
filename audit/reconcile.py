"""
audit/reconcile.py — Single source of truth for every disputed number.

All P&L here is computed by ONE function (`realized_pl`) whose parameters are read
from configs/cost_config.yaml. No hardcoded friction / margin / reduction constants.
Run:  uv run python audit/reconcile.py
"""
import os
import sys
import copy

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.policy.cost_engine import CostEngine  # noqa: E402
from src.eval.bayes_ceiling import get_true_p  # noqa: E402

CFG = yaml.safe_load(open("configs/cost_config.yaml"))
DROP = ["rto_label", "timestamp", "order_id"]


# ───────────────────────── canonical P&L ─────────────────────────
def realized_pl(actions, y, V, cfg):
    """Realized savings vs ALLOW_COD-for-everyone, using per-order outcome y (0/1).

    Expected-over-intervention-response accounting (the same convention used by
    tests/test_independent_realized_pl.py and the headline ₹69,786.08):
      RTO order     : friction + (1 - rto_reduction) * rto_cost
      delivered ord.: friction - (1 - success_drop) * margin
    """
    rto_cost, mpct, iv = cfg["rto_logistics_cost"], cfg["average_margin_pct"], cfg["interventions"]
    m = V * mpct
    base = np.where(y == 1, rto_cost, -m).sum()
    fr = np.array([iv[a]["friction_cost"] for a in actions], float)
    dr = np.array([iv[a]["success_drop_pct"] for a in actions], float)
    rr = np.array([iv[a]["rto_reduction_pct"] for a in actions], float)
    pol = np.where(y == 1, fr + (1 - rr) * rto_cost, fr - (1 - dr) * m).sum()
    return base - pol


def expected_pl(actions, p, V, cfg):
    """Same as realized_pl but with outcome probability p instead of label y."""
    rto_cost, mpct, iv = cfg["rto_logistics_cost"], cfg["average_margin_pct"], cfg["interventions"]
    m = V * mpct
    base = (p * rto_cost - (1 - p) * m).sum()
    fr = np.array([iv[a]["friction_cost"] for a in actions], float)
    dr = np.array([iv[a]["success_drop_pct"] for a in actions], float)
    rr = np.array([iv[a]["rto_reduction_pct"] for a in actions], float)
    pol = (fr + p * (1 - rr) * rto_cost - (1 - p) * (1 - dr) * m).sum()
    return base - pol


def frozen_actions(df_cod, p_cod, cfg_path=None):
    eng = CostEngine(cfg_path)
    acts, _ = eng.get_optimal_policy(df_cod.assign(payment_method="COD"), p_cod)
    return np.asarray(acts)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def flat_fraction(p):
    """Share of rows whose predicted value is shared with >=1% of all rows (plateaus)."""
    vc = pd.Series(np.round(p, 10)).value_counts()
    return vc[vc >= 0.01 * len(p)].sum() / len(p)


def bisect_breakeven(f, lo, hi, tol=1e-5):
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if np.sign(f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    test = pd.read_csv("data/processed/test.csv", dtype={"pincode": str})
    X = test.drop(columns=DROP, errors="ignore")
    y = test["rto_label"].values
    models = {
        "LogReg": joblib.load("models/logistic_baseline.pkl"),
        "LGBM uncal": joblib.load("models/tree_model.pkl"),
        "LGBM isotonic (PRIMARY)": joblib.load("models/tree_model_calibrated.pkl"),
    }
    P = {k: m.predict_proba(X)[:, 1] for k, m in models.items()}
    p_cal = P["LGBM isotonic (PRIMARY)"]

    cod = (test["payment_method"] == "COD").values
    tc = test[cod].reset_index(drop=True)
    yc, Vc, pc = tc["rto_label"].values, tc["order_value"].values, p_cal[cod]
    acts = frozen_actions(tc, pc)

    # ── C1 design point ──
    print("=" * 78)
    print("C1. DESIGN-POINT REALIZED SAVINGS (config params, frozen primary policy)")
    print("=" * 78)
    vals, cnts = np.unique(acts, return_counts=True)
    print("action counts:", dict(zip(vals, cnts.tolist())), " n_COD =", len(acts))
    canon = realized_pl(acts, yc, Vc, CFG)
    print(f"canonical realized savings            : Rs {canon:,.2f}")
    bug = copy.deepcopy(CFG)
    bug["interventions"]["REQUIRE_DEPOSIT"]["friction_cost"] = 0.50
    buggy = realized_pl(acts, yc, Vc, bug)
    n_dep = int((acts == "REQUIRE_DEPOSIT").sum())
    print(f"with Rs0.50 deposit friction (bug in run_step3_verifications.py): Rs {buggy:,.2f}")
    print(f"difference = Rs {canon - buggy:,.2f} = 0.50 x n_deposit ({n_dep}) = Rs {0.5 * n_dep:,.2f}")

    # ── C4 model table ──
    print()
    print("=" * 78)
    print("C4. MODEL TABLE (full test split, n=%d, recomputed from frozen artifacts)" % len(y))
    print("=" * 78)
    p_obs = get_true_p(test)
    rows = []
    for k, p in list(P.items()) + [("Observable Bayes ceiling (E[p|x])", p_obs)]:
        rows.append({"model": k,
                     "PR-AUC": average_precision_score(y, p),
                     "ROC-AUC": roc_auc_score(y, p),
                     "Brier": brier_score_loss(y, p),
                     "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                     "ECE10": ece(y, p),
                     "plateau_share": flat_fraction(p),
                     "n_unique_p": len(np.unique(np.round(p, 10)))})
    mt = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(mt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ceil = average_precision_score(y, p_obs)
    print(f"\nprimary / observable ceiling = {average_precision_score(y, p_cal):.4f} / {ceil:.4f} = "
          f"{average_precision_score(y, p_cal) / ceil:.3f}")

    # ── C2 Bayes: oracle vs observable ──
    print()
    print("=" * 78)
    print("C2. WHY THE SEED TABLE SAID ~0.46: latent ORACLE vs OBSERVABLE ceiling (seed 42)")
    print("=" * 78)
    sys.path.insert(0, os.path.join(ROOT, "audit"))
    from audit_prompt6 import get_latent_and_data
    df42, _, p_latent = get_latent_and_data(100000, 42)
    t42 = df42.iloc[85000:].reset_index(drop=True)
    y42 = t42["rto_label"].values
    print(f"latent p incl. per-order N(0,0.8) noise (UNOBSERVABLE oracle): PR-AUC = "
          f"{average_precision_score(y42, p_latent[85000:]):.4f}")
    print(f"E[p | observed x]  (true Bayes ceiling for any model)       : PR-AUC = "
          f"{average_precision_score(y42, get_true_p(t42)):.4f}")
    print("=> 0.46 is an oracle that sees the noise term; it is NOT an achievable ceiling.")

    # ── C5 break-even, ONE definition: COD-subset RTO rate, frozen policy ──
    print()
    print("=" * 78)
    print("C5. BREAK-EVEN — definition: COD-subset RTO rate; policy FROZEN; config costs")
    print("=" * 78)
    base_rate = yc.mean()
    print(f"observed COD RTO rate on test: {base_rate:.4f}")

    # Method A: label shift by resampling (keeps p(x|y); uses real labels)
    pos, neg = np.where(yc == 1)[0], np.where(yc == 0)[0]

    def resampled_savings(r, reps=20):
        out = []
        for s in range(reps):
            rng = np.random.default_rng(s)
            n_pos = int(round(len(neg) * r / (1 - r)))
            ip = rng.choice(pos, n_pos, replace=True)
            idx = np.concatenate([neg, ip])
            out.append(realized_pl(acts[idx], yc[idx], Vc[idx], CFG) / len(idx))
        return float(np.mean(out))

    # Method B: uniform odds rescaling of the model's probabilities (no labels)
    def odds_shift(p, r):
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + mid)))
            if q.mean() < r:
                lo = mid
            else:
                hi = mid
        return q

    pcl = np.clip(pc, 1e-4, 1 - 1e-4)

    def odds_savings(r):
        return expected_pl(acts, odds_shift(pcl, r), Vc, CFG) / len(acts)

    print(f"{'COD RTO':>8} | {'A: resampled labels, Rs/COD order':>34} | {'B: odds-shifted p, Rs/COD order':>32}")
    for r in [0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.24, round(base_rate, 4), 0.32]:
        print(f"{r:8.4f} | {resampled_savings(r):34.2f} | {odds_savings(r):32.2f}")
    beA = bisect_breakeven(lambda r: resampled_savings(r, reps=10), 0.05, 0.28, tol=1e-3)
    beB = bisect_breakeven(odds_savings, 0.05, 0.28, tol=1e-4)
    print(f"\nbreak-even COD RTO  A (label resampling): {beA:.4f}" if beA else "A: no crossing")
    print(f"break-even COD RTO  B (odds shift)      : {beB:.4f}" if beB else "B: no crossing")
    print("NOTE: valid ONLY for this synthetic basket + config (Rs150, 20% margin, 40%/80% deposit).")
    print("Earlier -Rs26,921 @15% came from audit/audit_prompt11.py, which used a different,")
    print("non-config loss model (25% margin, Rs1.50/Rs0.50 friction, no margin on ALLOW) on the")
    print("FULL test split. It is superseded by this table.")


if __name__ == "__main__":
    main()
