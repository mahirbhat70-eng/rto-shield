"""
src/data/generator.py
WHY: Generates the v1 synthetic order dataset used for training and evaluation
     of the frozen baseline pipeline. This is a controlled simulation — no
     causal interventions. The RTO label comes from one logistic model;
     intervention effects are *imputed* from the frozen constants in
     cost_config.yaml, not estimated.

     Key design decisions:
     - Seed 42 throughout — two runs must produce identical CSVs.
     - ~50k rows balances training signal with runtime.
     - Payment method distribution (48% COD) matches realistic Indian e-commerce.
     - pincode × tier assignment is stable within a run (same seed → same map).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Make the package importable from any CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ── Constants ──────────────────────────────────────────────────────────────────

SEED = 42
N_ROWS = 50_000

CATEGORIES = ["Electronics", "Apparel", "Footwear", "Beauty", "Home", "Jewelry"]
CAT_PROBS   = [0.20, 0.30, 0.15, 0.15, 0.12, 0.08]

PAYMENT_METHODS = ["COD", "UPI", "Credit Card", "Debit Card", "Net Banking"]
PM_PROBS        = [0.48, 0.28, 0.12, 0.08, 0.04]

COURIERS      = ["Courier_A", "Courier_B", "Courier_C", "Courier_D", "Courier_E"]
COURIER_PROBS = [0.30, 0.25, 0.20, 0.15, 0.10]

COD_CHARGES = [20, 29, 39, 49, 59, 79, 99]

N_PINCODES = 1000
N_CUSTOMERS = 10_000

# Category effect on RTO logit
CAT_RISK = {
    "Electronics": 0.20,
    "Apparel": 0.30,
    "Footwear": 0.15,
    "Beauty": 0.05,
    "Home": -0.10,
    "Jewelry": 0.08,
}

# Courier effect
COURIER_RISK = {
    "Courier_A": 0.00,
    "Courier_B": 0.05,
    "Courier_C": -0.05,
    "Courier_D": 0.10,
    "Courier_E": 0.15,
}

# Tier effect
TIER_RISK = {1: -0.15, 2: 0.10, 3: 0.30}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = N_ROWS, seed: int = SEED, output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic v1 order dataset.

    Args:
        n_rows:      Number of rows to generate.
        seed:        Random seed for full reproducibility.
        output_path: If given, writes CSV to this path.

    Returns:
        DataFrame with all v1 columns.
    """
    rng = np.random.default_rng(seed)

    # ── Pincode assignments ────────────────────────────────────────────────────
    pincodes = [f"{100000 + i:06d}" for i in range(N_PINCODES)]
    pincode_tier = {p: int(rng.integers(1, 4)) for p in pincodes}
    pincode_base_rate = {
        p: float(np.clip(rng.beta(2, 7) + TIER_RISK[pincode_tier[p]] * 0.05, 0.04, 0.55))
        for p in pincodes
    }

    # ── Customer pool ──────────────────────────────────────────────────────────
    customer_ids = [f"CUST{i:06d}" for i in range(N_CUSTOMERS)]

    # ── Static column draws ────────────────────────────────────────────────────
    n = n_rows

    # Timestamps: uniform over 2026-03-01 → 2026-11-30
    start_ts = pd.Timestamp("2026-03-01")
    end_ts   = pd.Timestamp("2026-11-30")
    span_days = (end_ts - start_ts).days
    timestamps = start_ts + pd.to_timedelta(
        rng.integers(0, span_days * 24 * 3600, size=n), unit="s"
    )
    timestamps = pd.DatetimeIndex(timestamps).sort_values()

    order_ids = [f"ORD{i:09d}" for i in range(n)]

    # Numeric features
    order_values = np.clip(
        np.exp(rng.normal(6.4, 0.8, size=n)), 50, 15_000
    ).round(2)

    quantities = np.clip(rng.poisson(1.2, size=n) + 1, 1, 10).astype(int)

    discount_zero = rng.random(size=n) < 0.5
    discount_vals = rng.beta(2, 5, size=n) * 70
    discount_pct  = np.where(discount_zero, 0.0, discount_vals).round(2)

    categories   = rng.choice(CATEGORIES, size=n, p=CAT_PROBS)
    payment_meth = rng.choice(PAYMENT_METHODS, size=n, p=PM_PROBS)
    couriers     = rng.choice(COURIERS, size=n, p=COURIER_PROBS)
    cust_ids     = rng.choice(customer_ids, size=n)
    order_pincodes = rng.choice(pincodes, size=n)

    # COD charge: only for COD orders
    cod_idx   = rng.integers(0, len(COD_CHARGES), size=n)
    cod_charge = np.where(
        payment_meth == "COD",
        np.array(COD_CHARGES)[cod_idx].astype(float),
        0.0
    )

    # Customer history features (simplified for v1 — no point-in-time)
    account_age_days  = rng.integers(0, 1461, size=n)  # up to 4 years
    prior_orders      = rng.integers(0, 50, size=n)
    prior_rto_count   = np.minimum(
        rng.integers(0, 10, size=n), prior_orders
    )
    orders_last_24h   = rng.integers(0, 6, size=n)
    device_cluster_sz = np.clip(rng.integers(1, 6, size=n), 1, 10).astype(int)

    # Pincode-derived features
    hist_rto_rate = np.array([pincode_base_rate[p] for p in order_pincodes])
    tier_arr      = np.array([pincode_tier[p] for p in order_pincodes])

    # ── RTO label (logistic model, one process, no interventions) ─────────────
    beta_0 = -2.30
    cat_risk_arr     = np.array([CAT_RISK[c] for c in categories])
    courier_risk_arr = np.array([COURIER_RISK[c] for c in couriers])
    tier_risk_arr    = np.array([TIER_RISK[t] for t in tier_arr])

    cod_flag = (payment_meth == "COD").astype(float)

    logit = (
        beta_0
        + 1.10 * cod_flag
        + 0.80 * np.log1p(prior_rto_count)
        + 2.50 * hist_rto_rate
        + 0.25 * np.log1p(orders_last_24h)
        + 0.20 * np.log1p(device_cluster_sz)
        - 0.12 * np.log1p(account_age_days + 1)
        - 0.15 * np.log1p(prior_orders)
        + cat_risk_arr
        + courier_risk_arr
        + tier_risk_arr
        + 0.005 * discount_pct
        + rng.normal(0, 0.8, size=n)
    )

    p_rto = _sigmoid(logit)
    rto_label = rng.random(size=n) < p_rto

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        "order_id":                  order_ids,
        "order_date":                timestamps,
        "order_value":               order_values,
        "quantity":                  quantities,
        "category":                  categories,
        "discount_pct":              discount_pct,
        "payment_method":            payment_meth,
        "cod_charge":                cod_charge,
        "customer_id":               cust_ids,
        "account_age_days":          account_age_days,
        "prior_orders":              prior_orders,
        "prior_rto_count":           prior_rto_count,
        "pincode":                   order_pincodes,
        "courier_id":                couriers,
        "pincode_tier":              tier_arr,
        "historical_pincode_rto_rate": hist_rto_rate.round(4),
        "orders_last_24h":           orders_last_24h,
        "device_cluster_size":       device_cluster_sz,
        "rto_label":                 rto_label.astype(int),
    })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df):,} rows -> {output_path}")
        print(f"  RTO rate: {df['rto_label'].mean():.3f}")
        print(f"  COD share: {(df['payment_method'] == 'COD').mean():.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate v1 synthetic order data.")
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="data/raw/synthetic_orders.csv")
    args = parser.parse_args()
    generate(n_rows=args.rows, seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
