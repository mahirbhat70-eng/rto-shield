"""
stage6_uplift_ope.py — Uplift modeling + off-policy evaluation on the
structural simulator v2 (src/data/generator_v2.py).

THE QUESTION THIS ANSWERS: the frozen pipeline treats intervention effects
(rto_reduction_pct / success_drop_pct) as global constants from industry
priors. On v2 data — where the TRUE effects are heterogeneous and mean
deposit effectiveness is 0.50, not the assumed 0.80 — what happens if you
LEARN the effects from logged data instead of assuming them?

Pipeline:
  1. Train risk + T-learner outcome models on the TRAIN window (Mar–Aug)
     using only OBSERVED data (assigned_action, observed_outcome) — no peeking.
  2. Compare four policies on the TEST window (Oct–Nov, festive drift):
       P1 Constants   — frozen cost_config.yaml constants (the v1 approach)
       P2 Global-learned — mean learned effects (one r/d per action)
       P3 Heterogeneous  — per-order learned effects r_a(x), d_a(x) (uplift)
       P4 Oracle         — true simulated effects (upper bound)
     All evaluated on GROUND-TRUTH potential outcomes (no EL self-scoring).
  3. Off-policy evaluation on the VAL window (Sep): IPS / SNIPS / DR
     estimates of each policy's value vs ground truth — can you trust an
     offline estimate before deploying?
  4. Drift report: festive-season degradation of the risk model.

Run:  python src/data/generator_v2.py   (if data/raw_v2/ is missing)
      python src/eval/stage6_uplift_ope.py
Outputs: reports/stage6_uplift_ope_results.md,
         reports/stage6/policy_savings.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDERS_CSV = "data/raw_v2/orders_v2.csv"
TRUTH_CSV = "data/raw_v2/potential_outcomes_v2.csv"

# Cost-engine constants (frozen config — used by P1 and shared EL framing)
RTO_COST = 150.0
MARGIN_PCT = 0.20
VERIFY_FRICTION = 2.0
ACTIONS = ["ALLOW", "VERIFY", "DEPOSIT", "PREPAID"]
CLAIMED = {  # configs/cost_config.yaml, the v1 "industry prior" constants
    "VERIFY":   {"r": 0.30, "d": 0.05},
    "DEPOSIT":  {"r": 0.80, "d": 0.40},
    "PREPAID":  {"r": 0.55, "d": 0.70},
}
FRICTION = {"ALLOW": 0.0, "VERIFY": VERIFY_FRICTION, "DEPOSIT": 0.0, "PREPAID": 0.0}

FEATURES_NUM = ["order_value", "quantity", "discount_pct", "cod_charge",
                "account_age_days", "prior_orders", "prior_rto_count",
                "historical_pincode_rto_rate", "orders_last_24h", "device_cluster_size"]
FEATURES_CAT = ["category", "payment_method", "courier_id", "pincode_tier"]


def load_data():
    if not (os.path.exists(ORDERS_CSV) and os.path.exists(TRUTH_CSV)):
        print("v2 data missing — generating first …")
        from src.data.generator_v2 import generate, validate
        df, truth = generate(120000, 42)
        validate(df, truth, len(df))
        os.makedirs("data/raw_v2", exist_ok=True)
        df.to_csv(ORDERS_CSV, index=False)
        truth.to_csv(TRUTH_CSV, index=False)
    df = pd.read_csv(ORDERS_CSV, dtype={"pincode": str})
    truth = pd.read_csv(TRUTH_CSV)
    return df.merge(truth, on="order_id", how="left")


def encode(df):
    """One consistent encoding for the WHOLE dataset (column alignment across
    windows/subsets is guaranteed by encoding once and slicing rows)."""
    out = df[FEATURES_NUM].astype(float).copy()
    for c in FEATURES_CAT:
        d = pd.get_dummies(df[c].astype(str), prefix=c)
        out = pd.concat([out, d], axis=1)
    return out


def fit_small(X, y, sample_weight=None, seed=42, small=False):
    m = LGBMClassifier(
        n_estimators=200 if small else 250, learning_rate=0.06,
        num_leaves=15 if small else 31,
        min_child_samples=30 if small else 40, random_state=seed, verbose=-1)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def el_action(choice, p, V, r, d):
    """Expected loss of action with effect params (r, d) — the cost-engine formula."""
    return FRICTION[choice] + p * (1 - r) * RTO_COST - (1 - p) * (1 - d) * (V * MARGIN_PCT)


def argmin_action(p_hat, V, effects):
    """effects: dict action -> (r_array, d_array). Returns chosen action index array."""
    n = len(p_hat)
    best_el = np.full(n, np.inf)
    best_a = np.zeros(n, dtype=int)
    for j, a in enumerate(ACTIONS):
        if a == "ALLOW":
            el = p_hat * RTO_COST - (1 - p_hat) * (V * MARGIN_PCT)
        else:
            r, d = effects[a]
            el = el_action(a, p_hat, V, r, d)
        m = el < best_el
        best_el[m] = el[m]
        best_a[m] = j
    return best_a


def realized_loss(action_idx, V, y_rto, dropped):
    """Ground-truth realized loss for the chosen actions (potential outcomes)."""
    n = len(action_idx)
    loss = np.zeros(n)
    for j, a in enumerate(ACTIONS):
        m = action_idx == j
        if m.sum() == 0:
            continue
        fr = FRICTION[a]
        y = y_rto[a][m]
        dr = dropped[a][m]
        loss[m] = fr + np.where(dr, 0.0,
                                np.where(y, RTO_COST, -(V[m] * MARGIN_PCT)))
    return loss


def ope_estimates(choice, a_logged, prop, loss_logged, q_at_pi, q_at_logged):
    """IPS / SNIPS / doubly-robust value estimates (mean loss per order; lower is better).

    choice      action indices chosen by the TARGET policy            (n,)
    a_logged    action indices the LOGGING policy actually took       (n,)
    prop        logging propensity of the action that was taken       (n,)
    loss_logged realized loss of the logged action                    (n,)
    q_at_pi     Q-hat predicted loss of the target policy's choice    (n,)
    q_at_logged Q-hat predicted loss of the logged action             (n,)

    Returns (v_ips, v_snips, v_dr). IPS is unbiased under correctness of
    `prop` but high-variance under thin overlap; SNIPS trades a little bias
    for much lower variance; DR is consistent if EITHER `prop` or `q` is
    correct (doubly robust) and lowest-variance when both are decent.
    """
    match = choice == a_logged
    w = np.where(match, 1.0 / np.maximum(prop, 1e-6), 0.0)
    v_ips = float(np.mean(w * loss_logged))
    v_snips = float(np.sum(w * loss_logged) / max(np.sum(w), 1e-9))
    v_dr = float(np.mean(
        q_at_pi + np.where(match, (loss_logged - q_at_logged) / np.maximum(prop, 1e-6), 0.0)))
    return v_ips, v_snips, v_dr


Y_RTO_KEYS = {"ALLOW": "y_allow", "VERIFY": "y_rto_verify",
              "DEPOSIT": "y_rto_deposit", "PREPAID": "y_rto_prepaid"}
DROP_KEYS = {"ALLOW": None, "VERIFY": "dropped_verify",
             "DEPOSIT": "dropped_deposit", "PREPAID": "dropped_prepaid"}


def main():
    df = load_data()
    X_all = encode(df)  # single encoding, aligned columns everywhere
    ts = pd.to_datetime(df["timestamp"])
    train_m = (ts <= "2026-08-31").values
    val_m = ((ts >= "2026-09-01") & (ts <= "2026-09-30")).values
    test_m = (ts >= "2026-10-01").values
    train = df[train_m]
    val = df[val_m]
    test = df[test_m]
    print(f"splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── 1. Models on OBSERVED train data only (COD orders) ──────────
    # TWO fits per model: naive (unweighted) and IPW (weighted by
    # 1/propensity of the logged action). The naive T-learner is confounded
    # (actions were assigned to high-risk orders), IPW re-creates a
    # pseudo-randomized population — the logged data's exploration floors
    # are what make this possible. Interventions only exist on COD orders,
    # so all models are trained on the COD population.
    is_cod_train = (train["payment_method"] == "COD").values
    allow_mask = ((train["assigned_action"] == "ALLOW") & is_cod_train).values
    train_prop = train["propensity"].to_numpy(float)
    ipw = 1.0 / np.maximum(train_prop, 1e-6)

    p_model = fit_small(X_all[train_m][allow_mask],
                        train.loc[allow_mask, "rto_label"].values)
    p_model_ipw = fit_small(X_all[train_m][allow_mask],
                            train.loc[allow_mask, "rto_label"].values,
                            sample_weight=ipw[allow_mask])

    outcome_models, drop_models = {}, {}          # naive (for the bias lesson)
    outcome_models_ipw, drop_models_ipw = {}, {}  # IPW (used by policies)
    for a in ("VERIFY", "DEPOSIT", "PREPAID"):
        m = ((train["assigned_action"] == a) & is_cod_train).values
        y_r = train.loc[m, "rto_label"].values
        y_d = (train.loc[m, "observed_outcome"] == "DROPPED").astype(int).values
        outcome_models[a] = fit_small(X_all[train_m][m], y_r, small=True)
        drop_models[a] = fit_small(X_all[train_m][m], y_d, small=True)
        outcome_models_ipw[a] = fit_small(X_all[train_m][m], y_r, sample_weight=ipw[m], small=True)
        drop_models_ipw[a] = fit_small(X_all[train_m][m], y_d, sample_weight=ipw[m], small=True)

    # Hajek (marginal) cross-check: the classic ATE estimator without ML
    cod_train = train[is_cod_train]
    hajek = {}
    for a in ("ALLOW", "VERIFY", "DEPOSIT", "PREPAID"):
        sub = cod_train[cod_train["assigned_action"] == a]
        w = 1.0 / np.maximum(sub["propensity"].to_numpy(float), 1e-6)
        hajek[a] = float(np.sum(w * sub["rto_label"].to_numpy(float)) / np.sum(w))
    hajek_effects = {a: 1.0 - hajek[a] / hajek["ALLOW"] for a in ("VERIFY", "DEPOSIT", "PREPAID")}

    # ── 2. Effect estimates on each window (COD orders only) ──────────
    def window_effects(row_mask, weighted):
        X = X_all[row_mask]
        pm = p_model_ipw if weighted else p_model
        om = outcome_models_ipw if weighted else outcome_models
        dm = drop_models_ipw if weighted else drop_models
        p_allow = pm.predict_proba(X)[:, 1]
        p_allow_safe = np.clip(p_allow, 0.05, 1.0)
        eff = {}
        for a in ("VERIFY", "DEPOSIT", "PREPAID"):
            p_a = om[a].predict_proba(X)[:, 1]
            d_a = dm[a].predict_proba(X)[:, 1]
            eff[a] = (np.clip(1.0 - p_a / p_allow_safe, 0.0, 1.0), np.clip(d_a, 0.0, 1.0))
        return p_allow, eff

    train_cod_m = train_m & (df["payment_method"] == "COD").values
    val_cod_m = val_m & (df["payment_method"] == "COD").values
    test_cod_m = test_m & (df["payment_method"] == "COD").values
    train_cod = df[train_cod_m]
    val_cod = df[val_cod_m]
    test_cod = df[test_cod_m]

    p_train, eff_train = window_effects(train_cod_m, weighted=True)
    _, eff_naive = window_effects(train_cod_m, weighted=False)
    _, eff_val = window_effects(val_cod_m, weighted=True)
    p_test, eff_test = window_effects(test_cod_m, weighted=True)

    # Global learned effects (mean of per-order estimates, train window)
    glob = {a: (float(eff_train[a][0].mean()), float(eff_train[a][1].mean()))
            for a in ("VERIFY", "DEPOSIT", "PREPAID")}

    # True effects for the oracle policy + honesty table
    true_eff_test = {
        "VERIFY": (test_cod["r_verify"].values, test_cod["drop_verify"].values),
        "DEPOSIT": (test_cod["r_deposit"].values, test_cod["drop_deposit"].values),
        "PREPAID": (test_cod["r_prepaid"].values, test_cod["drop_prepaid"].values),
    }

    print("\nlearned (IPW) vs TRUE mean effects (test window), naive shown for the bias lesson:")
    for a in ("VERIFY", "DEPOSIT", "PREPAID"):
        t_r = true_eff_test[a][0].mean()
        t_d = true_eff_test[a][1].mean()
        n_r = float(eff_naive[a][0].mean())
        print(f"  {a:8s} r: naive {n_r:.3f} | IPW {glob[a][0]:.3f} | Hajek {hajek_effects[a]:.3f} | "
              f"true {t_r:.3f} || d: IPW {glob[a][1]:.3f} vs true {t_d:.3f}")

    # ── 3. Four policies on the test window (COD) ─────────────────────
    V_test = test_cod["order_value"].to_numpy(float)
    y_rto_test = {a: test_cod[Y_RTO_KEYS[a]].to_numpy(bool) for a in ACTIONS}
    dropped_test = {a: (test_cod[DROP_KEYS[a]].to_numpy(bool) if DROP_KEYS[a]
                        else np.zeros(len(test_cod), dtype=bool)) for a in ACTIONS}
    p0_true = test_cod["p0"].to_numpy(float)

    baseline_loss = realized_loss(np.zeros(len(test_cod), dtype=int),
                                  V_test, y_rto_test, dropped_test)
    baseline_total = float(baseline_loss.sum())

    def consts_effects(n):
        return {a: (np.full(n, CLAIMED[a]["r"]), np.full(n, CLAIMED[a]["d"]))
                for a in ("VERIFY", "DEPOSIT", "PREPAID")}

    def global_effects(n):
        return {a: (np.full(n, glob[a][0]), np.full(n, glob[a][1]))
                for a in ("VERIFY", "DEPOSIT", "PREPAID")}

    n_test = len(test_cod)
    policy_choices = {
        "P1 Constants (v1)": argmin_action(p_test, V_test, consts_effects(n_test)),
        "P2 Global-learned": argmin_action(p_test, V_test, global_effects(n_test)),
        "P3 Uplift (heterogeneous)": argmin_action(p_test, V_test, eff_test),
        "P4 Oracle (true effects)": argmin_action(p0_true, V_test, true_eff_test),
    }

    results = {}
    for name, choice in policy_choices.items():
        loss = realized_loss(choice, V_test, y_rto_test, dropped_test)
        sav = baseline_total - float(loss.sum())
        mix = {ACTIONS[j]: float(np.mean(choice == j)) for j in range(len(ACTIONS))}
        results[name] = {"savings": sav, "mix": mix}
        print(f"{name:28s} savings ₹{sav:,.0f} | mix "
              f"{ {k: round(v, 3) for k, v in mix.items()} }")

    oracle_sav = results["P4 Oracle (true effects)"]["savings"]
    for name, r in results.items():
        r["pct_of_oracle"] = (r["savings"] / oracle_sav * 100.0) if oracle_sav > 0 else float("nan")

    # ── 4. Off-policy evaluation on the VAL window ────────────────────
    V_val = val_cod["order_value"].to_numpy(float)
    y_rto_val = {a: val_cod[Y_RTO_KEYS[a]].to_numpy(bool) for a in ACTIONS}
    dropped_val = {a: (val_cod[DROP_KEYS[a]].to_numpy(bool) if DROP_KEYS[a]
                       else np.zeros(len(val_cod), dtype=bool)) for a in ACTIONS}
    p0_true_val = val_cod["p0"].to_numpy(float)
    true_eff_val = {
        "VERIFY": (val_cod["r_verify"].values, val_cod["drop_verify"].values),
        "DEPOSIT": (val_cod["r_deposit"].values, val_cod["drop_deposit"].values),
        "PREPAID": (val_cod["r_prepaid"].values, val_cod["drop_prepaid"].values),
    }
    n_val = len(val_cod)

    # Logged data for OPE: chosen action of the LOGGING policy + its propensity
    a_logged = val_cod["assigned_action"].map({a: j for j, a in enumerate(ACTIONS)}).to_numpy(int)
    prop = val_cod["propensity"].to_numpy(float)
    loss_logged = realized_loss(a_logged, V_val, y_rto_val, dropped_val)

    # Q-hat for DR: predicted realized loss per action from the IPW models
    p_allow_val = np.clip(p_model_ipw.predict_proba(X_all[val_cod_m])[:, 1], 0.0, 1.0)
    Qhat = {}
    for j, a in enumerate(ACTIONS):
        if a == "ALLOW":
            Qhat[a] = p_allow_val * RTO_COST - (1 - p_allow_val) * (V_val * MARGIN_PCT)
        else:
            r_arr, d_arr = eff_val[a]
            # expected realized loss consistent with the EL formula
            Qhat[a] = (FRICTION[a] + (1 - d_arr) * (
                (p_allow_val * (1 - r_arr)) * RTO_COST
                - (1 - p_allow_val * (1 - r_arr)) * (V_val * MARGIN_PCT)))

    ope_rows = []
    val_choices = {
        "P1 Constants (v1)": argmin_action(p_allow_val, V_val, consts_effects(n_val)),
        "P2 Global-learned": argmin_action(p_allow_val, V_val, global_effects(n_val)),
        "P3 Uplift (heterogeneous)": argmin_action(p_allow_val, V_val, eff_val),
        "P4 Oracle (true effects)": argmin_action(p0_true_val, V_val, true_eff_val),
    }
    base_val_total = float(realized_loss(np.zeros(n_val, dtype=int), V_val, y_rto_val, dropped_val).sum())
    for name, choice in val_choices.items():
        v_true = float(realized_loss(choice, V_val, y_rto_val, dropped_val).sum()) / n_val
        q_pi = np.array([Qhat[ACTIONS[j]] for j in range(4)]).T  # (n, 4)
        v_ips, v_snips, v_dr = ope_estimates(
            choice, a_logged, prop, loss_logged,
            q_pi[np.arange(n_val), choice], q_pi[np.arange(n_val), a_logged])
        ope_rows.append({
            "policy": name, "v_true": v_true, "v_ips": v_ips, "v_snips": v_snips, "v_dr": v_dr,
            "ips_bias_pct": (v_ips - v_true) / abs(v_true) * 100,
            "dr_bias_pct": (v_dr - v_true) / abs(v_true) * 100,
        })
        print(f"OPE {name:28s} true {v_true:8.2f} | IPS {v_ips:8.2f} ({ope_rows[-1]['ips_bias_pct']:+.1f}%) "
              f"| SNIPS {v_snips:8.2f} | DR {v_dr:8.2f} ({ope_rows[-1]['dr_bias_pct']:+.1f}%)")

    # Also: what would the logging policy itself have scored? (reference)
    v_logging = float(loss_logged.sum()) / n_val
    logging_sav = base_val_total - float(loss_logged.sum())

    # ── 5. Drift report ───────────────────────────────────────────────
    drift = {}
    for label, row_mask, part in (("train", train_cod_m, train_cod), ("val", val_cod_m, val_cod),
                                  ("test", test_cod_m, test_cod)):
        p_hat = p_model_ipw.predict_proba(X_all[row_mask])[:, 1]
        drift[label] = {
            "cod_rto_rate": float(part["rto_label"].mean()),
            "mean_p_hat": float(p_hat.mean()),
            "n": len(part),
        }
    prauc_train = average_precision_score(train_cod["rto_label"], p_train)
    prauc_test = average_precision_score(test_cod["rto_label"], p_test)
    cal_gap_test = abs(p_test.mean() - test_cod["rto_label"].mean())
    print(f"\ndrift: train RTO {drift['train']['cod_rto_rate']:.3f} -> test "
          f"{drift['test']['cod_rto_rate']:.3f} | PR-AUC {prauc_train:.3f} -> {prauc_test:.3f} "
          f"| test calibration gap {cal_gap_test:.3f}")

    # ── 6. Figure ─────────────────────────────────────────────────────
    os.makedirs("reports/stage6", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    names = list(results.keys())
    savs = [results[n]["savings"] for n in names]
    colors = ["#94a3b8", "#f59e0b", "#0052FF", "#059669"]
    axes[0].bar(range(len(names)), savs, color=colors)
    axes[0].set_xticks(range(len(names)), [n.split(" (")[0] for n in names], rotation=12)
    axes[0].axhline(0, color="#0F172A", linewidth=0.8)
    for i, s in enumerate(savs):
        axes[0].text(i, s + (max(savs) * 0.02 if s >= 0 else -max(savs) * 0.08),
                     f"₹{s:,.0f}\n({results[names[i]]['pct_of_oracle']:.0f}% of oracle)",
                     ha="center", fontsize=8.5)
    axes[0].set_title("Ground-truth savings on the festive test window (COD orders)", fontsize=10)
    axes[0].set_ylabel("Savings vs always-allow (₹)")

    x = np.arange(len(ope_rows))
    w = 0.27
    axes[1].bar(x - w, [r["v_true"] for r in ope_rows], w, label="ground truth", color="#0F172A")
    axes[1].bar(x, [r["v_ips"] for r in ope_rows], w, label="IPS", color="#0052FF")
    axes[1].bar(x + w, [r["v_dr"] for r in ope_rows], w, label="doubly robust", color="#f59e0b")
    axes[1].set_xticks(x, [r["policy"].split(" (")[0] for r in ope_rows], rotation=12)
    axes[1].axhline(v_logging, color="#dc2626", linestyle="--", linewidth=1.2,
                    label=f"logging policy (₹{v_logging:.0f}/order)")
    axes[1].set_title("Off-policy estimates on the validation window (loss ₹/order, lower is better)", fontsize=10)
    axes[1].legend(fontsize=8.5)
    fig.suptitle("RTO Shield Stage 6 — learned effects vs assumed constants, with off-policy evaluation", fontsize=11)
    fig.savefig("reports/stage6/policy_savings.png", dpi=160)
    plt.close(fig)

    # ── 7. Report ─────────────────────────────────────────────────────
    L = []
    L.append("# Stage 6 — Uplift Modeling & Off-Policy Evaluation (structural simulator v2)\n")
    L.append("> Generated by `src/eval/stage6_uplift_ope.py` on `src/data/generator_v2.py` data. "
             "Unlike the frozen v1 results, these numbers are measured against **simulated ground-truth "
             "potential outcomes** — the intervention effects are no longer assumptions.\n")
    L.append("## Setup\n")
    L.append("| window | rows | period | role |")
    L.append("|---|---|---|---|")
    L.append(f"| train | {len(train):,} | Mar 1 – Aug 31 | model fitting (observed data only) |")
    L.append(f"| val | {len(val):,} | Sep 1 – Sep 30 | off-policy evaluation |")
    L.append(f"| test | {len(test):,} | Oct 1 – Nov 30 | festive-drift final evaluation |")
    L.append(f"\nTrue (simulated) effects are **heterogeneous** with means: "
             f"VERIFY r≈{true_eff_test['VERIFY'][0].mean():.2f}, DEPOSIT r≈{true_eff_test['DEPOSIT'][0].mean():.2f} "
             f"(vs the frozen assumption 0.80), PREPAID r≈{true_eff_test['PREPAID'][0].mean():.2f}; "
             f"deposit drop d≈{true_eff_test['DEPOSIT'][1].mean():.2f} (vs assumed 0.40).\n")

    L.append("## 1. Learned vs true effects (T-learner, train window)\n")
    L.append("| action | naive r (confounded) | IPW r (learned) | true r | IPW d | true d | frozen assumption (r, d) |")
    L.append("|---|---|---|---|---|---|---|")
    for a in ("VERIFY", "DEPOSIT", "PREPAID"):
        L.append(f"| {a} | {float(eff_naive[a][0].mean()):.3f} | {glob[a][0]:.3f} | "
                 f"{true_eff_test[a][0].mean():.3f} | {glob[a][1]:.3f} | {true_eff_test[a][1].mean():.3f} | "
                 f"({CLAIMED[a]['r']:.2f}, {CLAIMED[a]['d']:.2f}) |")
    L.append("\n**The estimation lesson — read this twice**. BOTH the naive and the IPW T-learner sit "
             "far above the truth for DEPOSIT and VERIFY; propensity weighting fixes selection into "
             "treatment, not extrapolation. Each arm model is fit on a 2–5% sliver of orders and must "
             "then predict the whole population, and the ratio r = 1 − p̂_a/p̂_allow amplifies any "
             "under-prediction of p̂_allow (floor-clipped at 0.05). A merchant who 'validates' the "
             "0.80 prior against logged data would measure "
             f"{glob['DEPOSIT'][0]:.2f} and call it confirmed — the truth is "
             f"{true_eff_test['DEPOSIT'][0].mean():.2f}. Only ground truth reveals the gap, and in "
             "production there is none: that is exactly the case for DR-style corrections, larger "
             "exploration floors, and parametric effect priors. The one effect the data CAN see is "
             f"the PREPAID drop rate ({glob['PREPAID'][1]:.2f} learned vs "
             f"{true_eff_test['PREPAID'][1].mean():.2f} true) — a 55–70% signal is loud enough to "
             "survive 2% coverage.\n")

    L.append("## 2. Policy comparison — festive test window, ground-truth outcomes\n")
    L.append(f"Baseline (always allow) portfolio loss: ₹{baseline_total:,.0f} over {n_test:,} COD orders.\n")
    L.append("| policy | realized savings (₹) | % of oracle | ALLOW | VERIFY | DEPOSIT | PREPAID |")
    L.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        L.append(f"| {name} | ₹{r['savings']:,.0f} | {r['pct_of_oracle']:.0f}% | "
                 f"{r['mix']['ALLOW']:.1%} | {r['mix']['VERIFY']:.1%} | "
                 f"{r['mix']['DEPOSIT']:.1%} | {r['mix']['PREPAID']:.1%} |")
    L.append("\nReading: P1's frozen constants assume deposit r = 0.80 against a truth of 0.50, yet "
             f"still capture {results['P1 Constants (v1)']['pct_of_oracle']:.0f}% of oracle value — a "
             "wrong-but-centered constant is not fatal. P2 (one learned (r, d) per action) is the best "
             f"deployable policy at {results['P2 Global-learned']['pct_of_oracle']:.0f}%: "
             f"+₹{results['P2 Global-learned']['savings'] - results['P1 Constants (v1)']['savings']:,.0f} "
             "over P1 for learning six numbers. P3 (per-order heterogeneous effects) TRAILS P1 at "
             f"{results['P3 Uplift (heterogeneous)']['pct_of_oracle']:.0f}%: its order-level effect "
             "estimates are noisier than the constant's bias (§1), so mis-ranking effects order-by-order "
             "costs more than it saves. The oracle's headroom over P1 — "
             f"₹{oracle_sav - results['P1 Constants (v1)']['savings']:,.0f}, "
             f"{100 - results['P1 Constants (v1)']['pct_of_oracle']:.0f} points of value — is the "
             "combined price of assuming instead of estimating AND of estimating badly on 2–5% arm "
             "coverage. Uplift is not free; it needs data.\n")

    L.append("## 3. Off-policy evaluation — validation window\n")
    L.append(f"Reference: the legacy logging policy scores ₹{v_logging:.2f} loss/order "
             f"(savings vs always-allow: ₹{logging_sav:,.0f}). Can offline estimators rank the "
             "candidates before deployment?\n")
    L.append("| policy | true loss (₹/order) | IPS | SNIPS | doubly robust | IPS bias | DR bias |")
    L.append("|---|---|---|---|---|---|---|")
    for r in ope_rows:
        L.append(f"| {r['policy']} | {r['v_true']:.2f} | {r['v_ips']:.2f} | {r['v_snips']:.2f} | "
                 f"{r['v_dr']:.2f} | {r['ips_bias_pct']:+.1f}% | {r['dr_bias_pct']:+.1f}% |")
    dr_lo = min(r['dr_bias_pct'] for r in ope_rows)
    dr_hi = max(r['dr_bias_pct'] for r in ope_rows)
    ips_hi = max(r['ips_bias_pct'] for r in ope_rows)
    ips_lo = min(r['ips_bias_pct'] for r in ope_rows)
    worst = max(ope_rows, key=lambda r: abs(r['dr_bias_pct']))['policy']
    n_single = sum(1 for r in ope_rows if abs(r['dr_bias_pct']) < 10)
    L.append(f"\nNo estimator is a substitute for overlap. Raw IPS ran {ips_lo:+.0f}–{ips_hi:+.0f}% "
             "optimistic on every policy (thin DEPOSIT/PREPAID coverage → rare matches → exploding "
             f"weights). DR cut bias to a {dr_lo:+.0f}–{dr_hi:+.0f}% band — single digits on "
             f"{n_single} of {len(ope_rows)} policies, but {max(abs(dr_lo), abs(dr_hi)):.0f} pts on "
             f"{worst}, whose action menu diverges most from what the logging policy actually did. "
             "The deployable pattern: keep exploration floors in the router, run DR (never raw IPS) "
             "before any policy swap, and distrust candidates whose actions diverge far from the "
             "logged ones.\n")

    L.append("## 4. Festive drift (the v1 frozen policy never saw October)\n")
    L.append("| window | COD RTO rate | mean model P |")
    L.append("|---|---|---|")
    for k in ("train", "val", "test"):
        L.append(f"| {k} | {drift[k]['cod_rto_rate']:.3f} | {drift[k]['mean_p_hat']:.3f} |")
    L.append(f"\nPR-AUC train {prauc_train:.3f} → test {prauc_test:.3f}; mean-prediction calibration "
             f"gap on test {cal_gap_test:.3f}. The risk model degrades exactly when it matters most "
             "(festive spike) — the case for drift monitoring (PSI/KS on top features, trailing-window "
             "pincode rebuilds) and scheduled recalibration.\n")

    L.append("## 5. Takeaways\n")
    p1 = results['P1 Constants (v1)']
    p2 = results['P2 Global-learned']
    p3 = results['P3 Uplift (heterogeneous)']
    L.append(f"1. **A wrong constant survived — barely**: P1's assumed r=0.80 (truth 0.50) still took "
             f"{p1['pct_of_oracle']:.0f}% of oracle; the remaining "
             f"₹{oracle_sav - p1['savings']:,.0f} ({100 - p1['pct_of_oracle']:.0f} pts) is what not "
             "knowing the effects costs. Global learning (P2) recovered the first "
             f"₹{p2['savings'] - p1['savings']:,.0f} of that with six learned numbers.")
    L.append(f"2. **Heterogeneous uplift did NOT beat a good constant here**: P3 took only "
             f"{p3['pct_of_oracle']:.0f}% of oracle — below P1 ({p1['pct_of_oracle']:.0f}%) — because "
             "per-order effect estimates on 2–5% arm coverage are noisier than the bias they correct "
             "(§1). Uplift pays only with real coverage or stronger estimators (causal forests, "
             "DR-learners, monotone effect priors).")
    L.append(f"3. **Off-policy evaluation is a guardrail, not an oracle**: DR bias ranged "
             f"{dr_lo:+.1f}% to {dr_hi:+.1f}% (worst on {worst}), raw IPS {ips_lo:+.0f}–{ips_hi:+.0f}% "
             "everywhere. Exploration floors are not optional — without them none of these estimators "
             "even exist.")
    L.append("4. **The production recipe**: exploration floors → logged propensities → learn global "
             "(r, d) first → heterogeneous uplift only once arm coverage supports it → DR-based "
             "offline audit → staged rollout with drift monitoring (§4: PR-AUC fell exactly when RTO "
             "spiked).\n")

    os.makedirs("reports", exist_ok=True)
    with open("reports/stage6_uplift_ope_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("Wrote reports/stage6_uplift_ope_results.md + reports/stage6/policy_savings.png")

    with open("reports/stage6_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_baseline_loss": round(baseline_total, 0),
            "policy_savings": {k: round(v["savings"], 0) for k, v in results.items()},
            "pct_of_oracle": {k: round(v["pct_of_oracle"], 1) for k, v in results.items()},
            "drift": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in drift.items()},
            "ope_dr_bias_pct": {r["policy"]: round(r["dr_bias_pct"], 1) for r in ope_rows},
        }, f, indent=2)


if __name__ == "__main__":
    main()
