"""
audit/run_c1_realism.py — Data realism check (correction pass, Task 1).

Regimes, each with a freshly trained LGBM + isotonic (same recipe in every regime, so
regimes are comparable to each other; NOT identical to the frozen shipped model):
  (a) current feature: per-order latent theta + N(0, 0.03)  (the label's causal input)
  (b) feature + N(0, 0.10)    (c) feature + N(0, 0.20)
  (d) empirical rolling rate from strictly prior 30 days of orders in the pincode,
      m-estimate shrinkage toward the train-period global rate
  (d') as (d) but only labels at least LABEL_LAG_DAYS old (RTO labels arrive late)

Run:  python audit/run_c1_realism.py            (shipped data, uniform pincode volume)
      python audit/run_c1_realism.py --zipf 1.1 (long-tail pincode volume, regenerated)
"""
import argparse, os, sys, tempfile
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "audit"))
from reconcile import realized_pl, frozen_actions, CFG  # noqa: E402
from src.models.tree_model import get_preprocessor  # noqa: E402
from src.data.split import temporal_split  # noqa: E402
from src.data.generator import generate  # noqa: E402

DROP = ["rto_label", "timestamp", "order_id", "customer_id"]
SHRINK_M = 10.0
LABEL_LAG_DAYS = 10


def rolling_prior_counts(df, window_days=30, lag_days=0):
    """Point-in-time (count, sum of labels) of orders in the same pincode with
    timestamp in [t - lag - window, t - lag). Strictly prior: ties at t are excluded."""
    t = pd.to_datetime(df["timestamp"]).values.astype("datetime64[s]").astype(np.int64)
    y = df["rto_label"].to_numpy()
    cnt, sm = np.zeros(len(df)), np.zeros(len(df))
    lag, win = lag_days * 86400, window_days * 86400
    for idx in df.groupby("pincode").indices.values():
        idx = idx[np.argsort(t[idx], kind="stable")]
        tt, cs = t[idx], np.concatenate([[0], np.cumsum(y[idx])])
        hi = np.searchsorted(tt, tt - lag, "left")
        lo = np.searchsorted(tt, tt - lag - win, "left")
        cnt[idx], sm[idx] = hi - lo, cs[hi] - cs[lo]
    return cnt, sm


def shrunk_rate(cnt, sm, prior, m=SHRINK_M):
    return (sm + m * prior) / (cnt + m)


def fit_eval(tr, vc, te, feat="historical_pincode_rto_rate"):
    pipe = Pipeline([("preprocessor", get_preprocessor()),
                     ("classifier", LGBMClassifier(random_state=42, verbose=-1, n_estimators=300,
                                                   learning_rate=0.05, num_leaves=31))])
    pipe.fit(tr.drop(columns=DROP, errors="ignore"), tr["rto_label"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        pipe.predict_proba(vc.drop(columns=DROP, errors="ignore"))[:, 1], vc["rto_label"])
    p = iso.predict(pipe.predict_proba(te.drop(columns=DROP, errors="ignore"))[:, 1])
    cod = (te["payment_method"] == "COD").to_numpy()
    tc = te[cod].reset_index(drop=True)
    y, V = tc["rto_label"].to_numpy(), tc["order_value"].to_numpy()
    acts_m = frozen_actions(tc, p[cod])
    acts_p = frozen_actions(tc, tc[feat].to_numpy(float))
    sm, sp = float(realized_pl(acts_m, y, V, CFG)), float(realized_pl(acts_p, y, V, CFG))
    return dict(pr_auc=float(average_precision_score(te["rto_label"], p)), model=sm, pin=sp,
                lift=(sm - sp) / abs(sp) * 100, n_cod=int(cod.sum()),
                pin_feat_mean=float(tc[feat].mean()), pin_feat_std=float(tc[feat].std()),
                pin_intervene=float((acts_p != "ALLOW_COD").mean()),
                model_intervene=float((acts_m != "ALLOW_COD").mean()))


def splits_of(raw):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "raw.csv"); raw.to_csv(p, index=False)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            tr, vc, vr, te = temporal_split(p, os.path.join(d, "out"))
    return tr.reset_index(drop=True), vc.reset_index(drop=True), te.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zipf", type=float, default=None)
    a = ap.parse_args()
    if a.zipf is None:
        raw = pd.read_csv(os.path.join(ROOT, "data/raw/synthetic_orders.csv"), dtype={"pincode": str})
        label = "shipped data (uniform pincode volume)"
    else:
        raw = generate(100000, 42, pincode_zipf_a=a.zipf); label = f"Zipf a={a.zipf} pincode volume"
    raw = raw.reset_index(drop=True)
    tr, vc, te = splits_of(raw)
    print(f"DATA: {label}; rows={len(raw)}, pincodes={raw.pincode.nunique()}, "
          f"train/val_cal/test={len(tr)}/{len(vc)}/{len(te)}")

    rng = np.random.default_rng(42)
    res = {"(a) current feature (latent+N(0,.03))": fit_eval(tr, vc, te)}
    for s in (0.10, 0.20):
        noisy = []
        for d in (tr, vc, te):
            d = d.copy()
            d["historical_pincode_rto_rate"] = np.clip(d["historical_pincode_rto_rate"]
                                                      + rng.normal(0, s, len(d)), 0, 1)
            noisy.append(d)
        res[f"({'b' if s == .1 else 'c'}) feature + N(0,{s:.2f})"] = fit_eval(*noisy)

    train_end = pd.to_datetime(tr["timestamp"]).max()
    prior = float(raw.loc[pd.to_datetime(raw["timestamp"]) <= train_end, "rto_label"].mean())
    stats = {}
    for tag, lag in (("(d) rolling 30d, lag 0", 0), (f"(d') rolling 30d, label lag {LABEL_LAG_DAYS}d", LABEL_LAG_DAYS)):
        cnt, sm = rolling_prior_counts(raw, 30, lag)
        r2 = raw.copy(); r2["historical_pincode_rto_rate"] = shrunk_rate(cnt, sm, prior)
        r2["_cnt"] = cnt
        t2, v2, e2 = splits_of(r2)
        stats[tag] = e2.loc[e2.payment_method == "COD", "_cnt"].to_numpy()
        res[tag] = fit_eval(*(x.drop(columns="_cnt") for x in (t2, v2, e2)))

    print(f"\nshrinkage: rate = (sum_y + m*prior) / (count + m), m={SHRINK_M}, prior=train-period "
          f"global RTO rate={prior:.4f}; window [t-lag-30d, t-lag), ties at t excluded")
    for tag, c in stats.items():
        print(f"{tag}: prior-window orders per test COD order: p10={np.percentile(c,10):.0f} "
              f"median={np.median(c):.0f} p90={np.percentile(c,90):.0f}; share with <5 = {(c<5).mean():.1%}")
    t_all = pd.to_datetime(raw["timestamp"])
    months = (t_all.max() - t_all.min()).days / 30.0
    per_pin = raw.groupby("pincode").size() / months
    print(f"orders per pincode per 30 days (pincode level): p10={per_pin.quantile(.1):.1f} "
          f"median={per_pin.median():.1f} p90={per_pin.quantile(.9):.1f}; "
          f"share of pincodes with <5 = {(per_pin<5).mean():.1%}")

    print("\n" + "=" * 118)
    print(f"{'regime':44s} {'PR-AUC':>7} {'model Rs':>11} {'pincode Rs':>11} {'lift':>8} "
          f"{'pinfeat mean':>12} {'pin interv':>10} {'model interv':>12}")
    for k, r in res.items():
        print(f"{k:44s} {r['pr_auc']:7.4f} {r['model']:11,.2f} {r['pin']:11,.2f} {r['lift']:+7.1f}% "
              f"{r['pin_feat_mean']:12.4f} {r['pin_intervene']:10.1%} {r['model_intervene']:12.1%}")
    a_ = res["(a) current feature (latent+N(0,.03))"]
    print("=" * 118)
    n_cod = a_["n_cod"]
    for k in list(res)[1:]:
        r = res[k]; loss = a_["model"] - r["model"]
        print(f"survival {k:40s}: {r['model']:,.2f} / {a_['model']:,.2f} = {r['model']/a_['model']:.2%}; "
              f"loss Rs {loss:,.2f} over {n_cod} COD orders = Rs {loss/n_cod*1000:,.2f} per 1,000 COD orders")


if __name__ == "__main__":
    main()
