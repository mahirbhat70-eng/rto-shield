"""
audit/correction_econ.py — correction-pass economics (Tasks 2, 5a, 8, 10).
All P&L via audit/reconcile.realized_pl (one function). Frozen calibrated model, test split, COD only.
Run: python audit/correction_econ.py [section ...]   sections: cap breakeven grid costs (default all)
"""
import copy, os, sys, tempfile
import numpy as np, pandas as pd, yaml, joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "audit"))
from reconcile import realized_pl, frozen_actions, bisect_breakeven, CFG  # noqa: E402
from src.policy.cost_engine import CostEngine  # noqa: E402
from src.serve.guardrails import ProductionGuardrails  # noqa: E402

test = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv"), dtype={"pincode": str})
P_ALL = joblib.load(os.path.join(ROOT, "models/tree_model_calibrated.pkl")).predict_proba(
    test.drop(columns=["rto_label", "timestamp", "order_id"]))[:, 1]
COD = (test.payment_method == "COD").to_numpy()
TC = test[COD].reset_index(drop=True)
Y, V, P = TC.rto_label.to_numpy(), TC.order_value.to_numpy(), P_ALL[COD]
ACTS = frozen_actions(TC, P)


def lost_good(acts, cfg):
    drop = np.array([cfg["interventions"][a]["success_drop_pct"] for a in acts])
    g = Y == 0
    return (drop * g).sum() / g.sum(), (drop * V * g).sum() / (V * g).sum()


def served(cap, mode, cfg=CFG):
    eng = CostEngine()
    gr = ProductionGuardrails(max_intervention_rate=cap, cap_mode=mode, high_value_threshold=1e12,
                              min_base_rto_rate=0.0)
    out = []
    for i in range(len(TC)):  # arrival order = timestamp order (test.csv is chronological)
        el = eng.evaluate_interventions(V[i], P[i])
        d = {"recommended_action": ACTS[i], "probability": P[i], "el_table": el}
        out.append(gr.evaluate({"order_value": V[i]}, lambda f, d=d: dict(d))["recommended_action"])
    return np.array(out)


def sec_cap():
    print("=" * 100 + "\nTASK 2  INTERVENTION RATE AND CAP (frozen model, test COD n=%d)" % len(TC))
    iv = (ACTS != "ALLOW_COD")
    print(f"pure policy intervention rate: {iv.sum()}/{len(ACTS)} = {iv.mean():.2%}  "
          f"mix={dict(zip(*np.unique(ACTS, return_counts=True)))}")
    base = realized_pl(ACTS, Y, V, CFG)
    print(f"{'cap':>5} {'mode':>7} {'served Rs':>11} {'vs pure':>8} {'forced ALLOW':>12} {'interv %':>8} "
          f"{'lost good COD %':>15} {'good COD rev lost %':>19}")
    for cap in (0.4, 0.6, 0.8, 1.0):
        for mode in ("naive", "ranked"):
            a = served(cap, mode)
            s = realized_pl(a, Y, V, CFG); lg, lr = lost_good(a, CFG)
            forced = int(((ACTS != "ALLOW_COD") & (a == "ALLOW_COD")).sum())
            print(f"{cap:5.0%} {mode:>7} {s:11,.2f} {s/base-1:+8.2%} {forced:12d} {(a!='ALLOW_COD').mean():8.2%} "
                  f"{lg:15.2%} {lr:19.2%}")
    lg, lr = lost_good(ACTS, CFG)
    print(f"pure (no cap): Rs {base:,.2f}; expected good COD customers lost {lg:.2%} "
          f"(= {lg*(Y==0).sum():.0f} of {(Y==0).sum()} good COD orders); good-COD revenue lost {lr:.2%}")


def sec_breakeven():
    print("=" * 100 + "\nTASK 5a  BREAK-EVEN (frozen actions; label-shift by resampling, 20 seeds)")
    pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]

    def per_order(r, seed):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([neg, rng.choice(pos, int(round(len(neg) * r / (1 - r))), replace=True)])
        return realized_pl(ACTS[idx], Y[idx], V[idx], CFG) / len(idx)

    print("raw sweep, Rs per COD order (mean over 20 seeds):")
    for r in (0.12, 0.15, 0.17, 0.20, 0.24, Y.mean(), 0.32):
        print(f"  COD RTO {r:.4f}: {np.mean([per_order(r, s) for s in range(20)]):+8.3f}")
    xs = np.array([bisect_breakeven(lambda r, s=s: per_order(r, s), 0.05, 0.28, tol=1e-4) for s in range(20)])
    boot = np.random.default_rng(0).choice(xs, (5000, len(xs))).mean(1)
    print(f"per-seed COD crossovers: {np.round(xs, 4).tolist()}")
    print(f"COD-RTO crossover: mean {xs.mean():.4f}, seed sd {xs.std(ddof=1):.4f}, "
          f"bootstrap 95% CI of mean [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")
    # overall-RTO mapping: prepaid labels fixed, COD rate shifted
    pre = test[~COD]; n_c, n_p = COD.sum(), (~COD).sum(); r_p = pre.rto_label.mean()
    ov = (n_c * xs + n_p * r_p) / (n_c + n_p)
    print(f"overall-RTO crossover (prepaid rate fixed at {r_p:.4f}, COD share {n_c/(n_c+n_p):.3f}): "
          f"mean {ov.mean():.4f}, range [{ov.min():.4f}, {ov.max():.4f}]")
    s15, sb = np.mean([per_order(.15, s) for s in range(20)]), np.mean([per_order(Y.mean(), s) for s in range(20)])
    lin = .15 + (0 - s15) * (Y.mean() - .15) / (sb - s15)
    print(f"linear interpolation between (0.15, {s15:+.3f}) and ({Y.mean():.4f}, {sb:+.3f}) -> {lin:.4f} "
          f"(savings are convex in r because the policy is frozen, so straight-line interpolation is biased)")
    return xs


def cfg_with(dd, dr, vd, vr):
    c = copy.deepcopy(CFG); iv = c["interventions"]
    iv["REQUIRE_DEPOSIT"].update(success_drop_pct=dd, rto_reduction_pct=dr)
    iv["VERIFY_ADDRESS"].update(success_drop_pct=vd, rto_reduction_pct=vr)
    return c


def acts_under(cfg, p):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name
    try:
        return frozen_actions(TC, p, path)
    finally:
        os.unlink(path)


def sec_grid():
    print("=" * 100 + "\nTASK 8  MISSPECIFICATION GRID (pessimistic / neutral / optimistic; policy FROZEN at design)")
    pin_acts = frozen_actions(TC, TC.historical_pincode_rto_rate.to_numpy(float))
    rows = []
    for dd in (0.70, 0.50, 0.30):
        for dr in (0.20, 0.50, 0.80):
            for vd in (0.20, 0.10, 0.05):
                for vr in (0.0, 0.20, 0.40):
                    t = cfg_with(dd, dr, vd, vr)
                    rows.append(dict(dep_drop=dd, dep_red=dr, ver_drop=vd, ver_red=vr,
                                     policy=realized_pl(ACTS, Y, V, t), pincode=realized_pl(pin_acts, Y, V, t),
                                     reoptimized=realized_pl(acts_under(t, P), Y, V, t)))
    g = pd.DataFrame(rows); g.to_csv(os.path.join(ROOT, "audit/misspecification_grid.csv"), index=False)
    print(f"cells={len(g)}; design point (0.40,0.80,0.05,0.30) is NOT on this grid (dep_drop 0.40 is between levels)")
    for col in ("policy", "reoptimized"):
        s = g[col]
        print(f"{col:12s}: beats always-allow {(s>0).mean():.1%} ({(s>0).sum()}/{len(g)}), beats pincode rule "
              f"{(s>g.pincode).mean():.1%}; min {s.min():,.0f} median {s.median():,.0f} max {s.max():,.0f}")
    print(f"pincode rule: min {g.pincode.min():,.0f} median {g.pincode.median():,.0f} max {g.pincode.max():,.0f}")
    print("profitable share by parameter value (frozen policy):")
    for c in ("dep_drop", "dep_red", "ver_drop", "ver_red"):
        print(f"  {c}: " + ", ".join(f"{v}: {(g[g[c]==v].policy>0).mean():.0%}" for v in sorted(g[c].unique())))
    prof = g[g.policy > 0]
    pess = prof.assign(score=(prof.dep_drop - prof.dep_red + prof.ver_drop - prof.ver_red)).sort_values("score")
    print("most pessimistic profitable cells (frozen policy):")
    print(pess.tail(5).drop(columns="score").to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


ASSUMED_COSTS_NOTE = "see configs/cost_config.yaml friction_components (ASSUMED)"


def sec_costs():
    print("=" * 100 + "\nTASK 10  MISSING COSTS (" + ASSUMED_COSTS_NOTE + ")")
    old = copy.deepcopy(CFG)
    for a in old["interventions"]:
        old["interventions"][a]["friction_cost"] = {"ALLOW_COD": 0, "VERIFY_ADDRESS": 2}.get(a, 0)
    new = CFG
    for name, c in (("OLD (deposit friction Rs0, verify Rs2)", old), ("NEW (config as loaded)", new)):
        a = acts_under(c, P)
        print(f"{name:40s}: frictions={ {k: v['friction_cost'] for k, v in c['interventions'].items()} }")
        print(f"{'':40s}  realized Rs {realized_pl(a, Y, V, c):,.2f}; mix "
              f"{dict(zip(*np.unique(a, return_counts=True)))}")


if __name__ == "__main__":
    want = sys.argv[1:] or ["cap", "breakeven", "grid", "costs"]
    for s in want:
        globals()[f"sec_{s}"]()
