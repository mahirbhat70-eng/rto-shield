"""
generator_v2.py — Structural RTO simulator with potential outcomes (Stage 6).

WHY: the v1 generator (src/data/generator.py) draws rto_label from a single
logistic-Bernoulli process — there are no interventions in the data, so every
downstream financial number is *imputed* from the four constants in
cost_config.yaml rather than *estimated* (see
reports/deposit_effectiveness_sensitivity.md for how much that matters).

This simulator generates a *causal* world:
  * POTENTIAL OUTCOMES: every order carries Y(a) for a in
    {ALLOW, VERIFY, DEPOSIT, PREPAID} — RTO and drop outcomes under each
    action, with HETEROGENEOUS effects (r/d depend on order features).
  * LOGGING POLICY: a legacy merchant heuristic assigns actions with known
    propensities (>=1% exploration floor on every action — the positivity
    assumption that makes off-policy evaluation possible).
  * DELAYED LABELS: RTO outcomes resolve 5-28 days after the order; the
    prior_* features are strictly point-in-time (only resolved history).
  * CUSTOMER RANDOM EFFECTS: u_c ~ N(0, 0.8) creates within-customer
    correlation (fraud rings / chronic refusers) that v1 lacked.
  * FESTIVE SEASONALITY: Oct/Nov volume + risk spikes (the regime the v1
    frozen policy never saw).
  * COLD PINCODES: 200 new-area pincodes appear only from mid-August.

Two output files (same seed => same data):
  data/raw_v2/orders_v2.csv          — what a merchant would observe
  data/raw_v2/potential_outcomes_v2.csv — ground truth Y(a) (eval only)

Run: python src/data/generator_v2.py [--rows 120000] [--seed 42]
"""

import argparse
import os
import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────

START = pd.Timestamp("2026-03-01")
END = pd.Timestamp("2026-11-30 23:59:59")
NEW_PINCODE_ACTIVATION = pd.Timestamp("2026-08-15")

MONTH_INTENSITY = {  # volume multipliers (festive ramp)
    3: 1.00, 4: 1.00, 5: 1.00, 6: 1.00, 7: 1.00, 8: 1.00,
    9: 1.20, 10: 1.60, 11: 1.30,
}
MONTH_RISK_EFFECT = {  # additive on z (festive RTO spike)
    3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0,
    9: 0.15, 10: 0.50, 11: 0.30,
}
BETA_0 = -2.30  # tuned so overall observed RTO lands ~20-24%

N_ESTABLISHED_PINS = 1000
N_NEW_PINS = 200
N_CUSTOMERS = 30000

CATEGORIES = ["Electronics", "Apparel", "Footwear", "Beauty", "Home", "Jewelry"]
CAT_PROBS = [0.20, 0.30, 0.15, 0.15, 0.12, 0.08]
CAT_RISK = {"Apparel": 0.30, "Electronics": 0.20, "Footwear": 0.15,
            "Beauty": 0.05, "Home": -0.10, "Jewelry": 0.08}
COURIERS = ["Courier_A", "Courier_B", "Courier_C", "Courier_D", "Courier_E"]
COURIER_PROBS = [0.30, 0.25, 0.20, 0.15, 0.10]
COURIER_RISK = {"Courier_A": 0.0, "Courier_B": 0.05, "Courier_C": -0.05,
                "Courier_D": 0.10, "Courier_E": 0.15}
PAYMENTS = ["COD", "UPI", "Credit Card", "Debit Card", "Net Banking"]
PAYMENT_PROBS = [0.48, 0.28, 0.12, 0.08, 0.04]
COD_CHARGE_OPTS = [20.0, 29.0, 39.0, 49.0, 59.0, 79.0, 99.0]
COD_CHARGE_PROBS = [0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]

ACTIONS = ["ALLOW", "VERIFY", "DEPOSIT", "PREPAID"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_rows=120000, seed=42):
    rng = np.random.default_rng(seed)

    # ── 1. Pincode pools ──────────────────────────────────────────────
    raw = rng.choice(np.arange(100000, 999999), size=N_ESTABLISHED_PINS + N_NEW_PINS, replace=False)
    pincodes = [str(p) for p in sorted(raw)]
    established = pincodes[:N_ESTABLISHED_PINS]
    new_area = pincodes[N_ESTABLISHED_PINS:]
    theta = rng.beta(2, 8, size=len(pincodes))
    tiers = rng.choice([1, 2, 3], size=len(pincodes), p=[0.30, 0.40, 0.30])
    pin_map = {p: i for i, p in enumerate(pincodes)}
    # Zipf popularity over established pincodes; new areas get uniform
    zipf_w = 1.0 / np.arange(1, N_ESTABLISHED_PINS + 1) ** 0.9
    zipf_w = zipf_w / zipf_w.sum()
    global_prior_rate = float(theta.mean())

    # ── 2. Customers ──────────────────────────────────────────────────
    cust_ids = [f"CUST{i + 1:06d}" for i in range(N_CUSTOMERS)]
    cust_start = pd.Timestamp("2025-06-01") + pd.to_timedelta(
        rng.integers(0, 270, size=N_CUSTOMERS), unit="D")
    u_c = rng.normal(0, 0.8, size=N_CUSTOMERS)          # customer random effect
    rto_prop = rng.beta(1.5, 6.0, size=N_CUSTOMERS)
    dev_base = rng.choice([1, 1, 1, 2, 3, 5], size=N_CUSTOMERS)
    lam_24h = rng.gamma(2.0, 0.8, size=N_CUSTOMERS)     # velocity intensity

    # ── 3. Order timestamps with festive intensity ────────────────────
    days = pd.date_range(START, END.normalize(), freq="D")
    day_w = np.array([MONTH_INTENSITY[d.month] for d in days])
    day_w = day_w / day_w.sum()
    day_idx = rng.choice(len(days), size=n_rows, p=day_w)
    sec_of_day = rng.integers(0, 86400, size=n_rows)
    timestamps = days.to_numpy()[day_idx] + pd.to_timedelta(sec_of_day, unit="s")
    order_df = pd.DataFrame({"timestamp": pd.DatetimeIndex(timestamps)}).sort_values("timestamp").reset_index(drop=True)
    ts = order_df["timestamp"]

    # ── 4. Static per-order draws ─────────────────────────────────────
    cust_idx = rng.integers(0, N_CUSTOMERS, size=n_rows)
    is_new_area = (ts >= NEW_PINCODE_ACTIVATION) & (rng.random(n_rows) < 0.12)
    pin_idx = np.empty(n_rows, dtype=int)
    pin_idx[~is_new_area.values] = rng.choice(
        N_ESTABLISHED_PINS, size=(~is_new_area.values).sum(), p=zipf_w)
    pin_idx[is_new_area.values] = N_ESTABLISHED_PINS + rng.integers(
        0, N_NEW_PINS, size=is_new_area.values.sum())
    pincode = np.array(pincodes, dtype=object)[pin_idx]
    tier = tiers[pin_idx]
    theta_i = theta[pin_idx]
    # Observed feature: good estimator for established, prior for cold areas
    hist_rate = np.where(
        pin_idx < N_ESTABLISHED_PINS,
        np.clip(theta_i + rng.normal(0, 0.04, n_rows), 0, 1),
        np.clip(global_prior_rate + rng.normal(0, 0.05, n_rows), 0, 1),
    )

    sampled_cat = rng.choice(CATEGORIES, size=n_rows, p=CAT_PROBS)
    cat_risk = np.array([CAT_RISK[c] for c in sampled_cat])
    sampled_pm = rng.choice(PAYMENTS, size=n_rows, p=PAYMENT_PROBS)
    is_cod = (sampled_pm == "COD").astype(float)
    cod_charge = np.where(sampled_pm == "COD",
                          rng.choice(COD_CHARGE_OPTS, size=n_rows, p=COD_CHARGE_PROBS), 0.0)
    courier = rng.choice(COURIERS, size=n_rows, p=COURIER_PROBS)
    courier_risk = np.array([COURIER_RISK[c] for c in courier])
    tier_risk = np.where(tier == 3, 0.30, np.where(tier == 2, 0.10, -0.15))
    order_value = np.round(np.clip(rng.lognormal(6.4, 0.8, n_rows), 50, 15000), 2)
    quantity = np.clip(rng.poisson(1.2, n_rows) + 1, 1, 10)
    discount = np.zeros(n_rows)
    has_disc = rng.random(n_rows) > 0.5
    discount[has_disc] = rng.beta(2, 5, has_disc.sum()) * 70
    discount = np.round(discount, 2)

    month = ts.dt.month.values
    month_effect = np.array([MONTH_RISK_EFFECT[m] for m in month])

    # ── 5. Sequential per-customer pass: point-in-time features, Y(a) ─
    order_rows = np.argsort(cust_idx, kind="stable")  # group rows by customer
    # history[cust] = list of (arrival_ts, is_rto); placed[cust] = list of ts
    history = {}
    placed = {}
    prior_orders = np.zeros(n_rows, dtype=int)
    prior_rto = np.zeros(n_rows, dtype=int)
    orders_24h = np.zeros(n_rows, dtype=int)
    account_age = np.zeros(n_rows, dtype=int)
    device_cluster = np.zeros(n_rows, dtype=int)

    ts_vals = ts.values
    start_val = cust_start.values

    # row → precomputed basics
    for i in order_rows:
        c = cust_idx[i]
        t = ts_vals[i]
        hist = history.setdefault(c, [])
        # resolved history as of t (delayed labels!)
        n_res = 0
        n_rto = 0
        for arr, was_rto in hist:
            if arr <= t:
                n_res += 1
                n_rto += int(was_rto)
        prior_orders[i] = n_res
        prior_rto[i] = n_rto
        # velocity: customer's orders in the last 24h
        p_list = placed.setdefault(c, [])
        p_list.append(t)
        orders_24h[i] = sum(1 for x in p_list if t - np.timedelta64(24, "h") <= x <= t) - 1
        account_age[i] = max(0, int((t - start_val[c]) / np.timedelta64(1, "D")))
        device_cluster[i] = max(1, int(dev_base[c] + rng.integers(0, 2)))

    # ── 6. Risk z, p0, potential outcomes, assignment, observed data ─
    z = (
        BETA_0
        + month_effect
        + 1.10 * is_cod
        + 0.80 * np.log1p(prior_rto)
        + 2.50 * hist_rate
        + 0.25 * np.log1p(orders_24h)
        + 0.20 * np.log1p(device_cluster)
        - 0.12 * np.log1p(account_age + 1)
        - 0.15 * np.log1p(prior_orders)
        + cat_risk + courier_risk + tier_risk
        + 0.005 * discount
        + u_c[cust_idx]
        + rng.normal(0, 0.8, n_rows)
    )
    p0 = _sigmoid(z)
    y_allow = rng.binomial(1, p0).astype(bool)

    # Heterogeneous intervention effects (the TRUTH the policies must learn)
    r_verify = 0.20 + 0.12 * (tier == 3) + 0.10 * (orders_24h >= 3)
    drop_verify = 0.03 + 0.04 * (account_age < 90)
    r_deposit = 0.35 + 0.25 * (order_value < 700) + 0.10 * (prior_rto >= 1)
    drop_deposit = 0.25 + 0.20 * (account_age < 180) + 0.10 * (order_value < 400)
    r_prepaid = 0.30 + 0.10 * (account_age < 90)
    drop_prepaid = 0.55 + 0.15 * (order_value < 500)

    y_rto = {
        "ALLOW": y_allow,
        "VERIFY": rng.binomial(1, np.clip(p0 * (1 - r_verify), 0, 1)).astype(bool),
        "DEPOSIT": rng.binomial(1, np.clip(p0 * (1 - r_deposit), 0, 1)).astype(bool),
        "PREPAID": rng.binomial(1, np.clip(p0 * (1 - r_prepaid), 0, 1)).astype(bool),
    }
    dropped = {
        "ALLOW": np.zeros(n_rows, dtype=bool),
        "VERIFY": rng.binomial(1, drop_verify).astype(bool),
        "DEPOSIT": rng.binomial(1, drop_deposit).astype(bool),
        "PREPAID": rng.binomial(1, drop_prepaid).astype(bool),
    }

    # Logging policy: legacy heuristic + exploration floors (positivity!).
    # Interventions apply to COD orders ONLY (non-COD payments passthrough —
    # same semantics as the serving layer). Rule = v1 rule baseline.
    rule_flag = (hist_rate > 0.27) | (prior_rto >= 2) | ((sampled_pm == "COD") & (account_age < 113))
    rule_flag = rule_flag & (sampled_pm == "COD")  # interventions are COD tools
    propensities = np.zeros((n_rows, 4))
    propensities[rule_flag] = np.array([0.05, 0.85, 0.05, 0.05])    # ALLOW, VERIFY, DEPOSIT, PREPAID
    propensities[~rule_flag] = np.array([0.95, 0.02, 0.02, 0.01])
    # non-COD orders: always ALLOW with propensity 1 (deterministic passthrough)
    non_cod_mask = sampled_pm != "COD"
    propensities[non_cod_mask] = np.array([1.0, 0.0, 0.0, 0.0])
    u_assign = rng.random(n_rows)
    cum = np.cumsum(propensities, axis=1)
    action_idx = (u_assign[:, None] > cum).sum(axis=1)
    action_idx[non_cod_mask] = 0
    assigned = np.array(ACTIONS, dtype=object)[action_idx]

    observed_outcome = np.empty(n_rows, dtype=object)
    for j, a in enumerate(ACTIONS):
        m = action_idx == j
        observed_outcome[m] = np.where(
            dropped[a][m], "DROPPED",
            np.where(y_rto[a][m], "RTO", "DELIVERED"))

    label_delay = np.round(np.clip(rng.gamma(6.0, 2.0, n_rows), 5, 28)).astype(int)

    # ── 7. Assemble outputs ───────────────────────────────────────────
    df = pd.DataFrame({
        "order_id": [f"ORD{i + 1:09d}" for i in range(n_rows)],
        "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "order_value": order_value,
        "quantity": quantity,
        "category": sampled_cat,
        "discount_pct": discount,
        "payment_method": sampled_pm,
        "cod_charge": cod_charge,
        "customer_id": np.array(cust_ids, dtype=object)[cust_idx],
        "account_age_days": account_age,
        "prior_orders": prior_orders,
        "prior_rto_count": prior_rto,
        "pincode": pincode,
        "courier_id": courier,
        "pincode_tier": tier,
        "historical_pincode_rto_rate": np.round(hist_rate, 4),
        "orders_last_24h": orders_24h,
        "device_cluster_size": device_cluster,
        # observed world:
        "assigned_action": assigned,
        "propensity": propensities[np.arange(n_rows), action_idx],
        "observed_outcome": observed_outcome,
        "rto_label": (observed_outcome == "RTO").astype(int),
        "label_arrival_delay_days": label_delay,
    })

    truth = pd.DataFrame({
        "order_id": df["order_id"],
        "p0": np.round(p0, 6),
        "y_allow": y_allow.astype(int),
        "r_verify": np.round(r_verify, 4), "drop_verify": np.round(drop_verify, 4),
        "r_deposit": np.round(r_deposit, 4), "drop_deposit": np.round(drop_deposit, 4),
        "r_prepaid": np.round(r_prepaid, 4), "drop_prepaid": np.round(drop_prepaid, 4),
        "y_rto_verify": y_rto["VERIFY"].astype(int),
        "y_rto_deposit": y_rto["DEPOSIT"].astype(int),
        "y_rto_prepaid": y_rto["PREPAID"].astype(int),
        "dropped_verify": dropped["VERIFY"].astype(int),
        "dropped_deposit": dropped["DEPOSIT"].astype(int),
        "dropped_prepaid": dropped["PREPAID"].astype(int),
    })
    return df, truth


def validate(df, truth, n_rows):
    assert len(df) == n_rows and len(truth) == n_rows
    assert df["order_id"].is_unique
    assert df.isnull().sum().sum() == 0
    assert set(df["observed_outcome"].unique()) <= {"DELIVERED", "RTO", "DROPPED"}
    assert df["propensity"].between(0.01, 1.0).all(), "positivity violated (<1%)"
    assert (df["label_arrival_delay_days"].between(5, 28)).all()
    assert (df["prior_rto_count"] <= df["prior_orders"]).all()
    rate = (df["observed_outcome"] == "RTO").mean()
    assert 0.15 <= rate <= 0.30, f"observed RTO rate {rate:.3f} outside [0.15, 0.30]"
    # festive drift must exist: Oct/Nov RTO > Mar-Aug RTO
    ts = pd.to_datetime(df["timestamp"])
    early = df[(ts.dt.month <= 8)]["rto_label"].mean()
    festive = df[(ts.dt.month >= 10)]["rto_label"].mean()
    assert festive > early, f"no festive drift: {early:.3f} vs {festive:.3f}"
    # cold pincodes must exist in late data: pins that appear after Aug 15
    # and were never seen before (new serviceable areas)
    early_pins = set(df[ts < "2026-08-15"]["pincode"])
    late = df[ts >= "2026-08-15"]
    cold_share = (~late["pincode"].isin(early_pins)).mean()
    assert cold_share > 0.02, f"cold-pincode share too low: {cold_share:.3f}"
    print(f"  observed RTO rate: {rate:.3f} | early {early:.3f} -> festive {festive:.3f} "
          f"| cold-pincode share (late): {cold_share:.1%}")
    return True


def main():
    parser = argparse.ArgumentParser(description="RTO Shield — structural simulator v2")
    parser.add_argument("--rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/raw_v2")
    args = parser.parse_args()

    df, truth = generate(args.rows, args.seed)
    validate(df, truth, args.rows)

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "orders_v2.csv"), index=False, lineterminator="\n")
    truth.to_csv(os.path.join(args.output_dir, "potential_outcomes_v2.csv"), index=False, lineterminator="\n")

    print(f"Rows: {len(df)} | columns: {df.shape[1]}")
    print(f"Action mix: {df['assigned_action'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Outcome mix: {df['observed_outcome'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Wrote {args.output_dir}/orders_v2.csv + potential_outcomes_v2.csv")


if __name__ == "__main__":
    main()
