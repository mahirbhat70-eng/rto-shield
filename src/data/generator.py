"""
RTO Shield — Synthetic Data Generator (Stage 1)

Generates 100,000 synthetic order records using a probabilistic latent-risk
logistic-Bernoulli process. No deterministic if/else rules for rto_label.

Target variable: rto_label (1 = RTO, 0 = Delivered)
Output file: data/raw/synthetic_orders.csv (Exactly 19 columns)

All randomness flows from a single np.random.default_rng(seed).
Intermediate latent variables (z, p_rto, theta_pincode) are NOT saved to CSV.
"""

import os
import argparse
import numpy as np
import pandas as pd


# ─── Pincode pool generation (deterministic from a seed) ───────────────
def _build_pincode_pool(rng, n_pincodes=1000):
    """
    Build a fixed pool of ~1000 synthetic 6-digit Indian-style pincodes.
    Each pincode gets a deterministic tier (1/2/3) and a latent base risk
    theta_pincode ~ Beta(2, 8) (mean ≈ 0.20).
    """
    # Generate 6-digit pincode strings: 100000..999999
    raw = rng.choice(np.arange(100000, 999999), size=n_pincodes, replace=False)
    pincodes = [str(p) for p in sorted(raw)]

    # Assign tiers: ~30% Tier 1, ~40% Tier 2, ~30% Tier 3
    tiers = rng.choice([1, 2, 3], size=n_pincodes, p=[0.30, 0.40, 0.30])

    # Latent base risk per pincode: Beta(2, 8), mean ≈ 0.20
    theta = rng.beta(a=2, b=8, size=n_pincodes)

    pool_df = pd.DataFrame({
        'pincode': pincodes,
        'pincode_tier': tiers.astype(int),
        'theta_pincode': theta,
    })
    return pool_df


# ─── Validation ────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    'order_id', 'timestamp', 'order_value', 'quantity', 'category',
    'discount_pct', 'payment_method', 'cod_charge',
    'customer_id', 'account_age_days', 'prior_orders', 'prior_rto_count',
    'pincode', 'courier_id', 'pincode_tier', 'historical_pincode_rto_rate',
    'orders_last_24h', 'device_cluster_size',
    'rto_label'
]

ALLOWED_PAYMENT = {"COD", "UPI", "Credit Card", "Debit Card", "Net Banking"}
ALLOWED_CATEGORIES = {"Electronics", "Apparel", "Footwear", "Beauty", "Home", "Jewelry"}


def validate(df, expected_rows):
    """
    Validate generated dataframe against all Stage 1 invariants.
    Raises AssertionError on any violation — never silently repairs.
    """
    # Shape
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert df.shape[1] == 19, f"Expected 19 columns, got {df.shape[1]}"
    assert list(df.columns) == EXPECTED_COLUMNS, f"Column names/order mismatch: {list(df.columns)}"

    # No nulls
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} null values"

    # order_id unique
    assert df['order_id'].is_unique, "order_id is not unique"

    # Timestamps parseable, sorted, span >= 5 months
    ts = pd.to_datetime(df['timestamp'])
    assert ts.is_monotonic_increasing, "timestamp is not sorted ascending"
    span_days = (ts.max() - ts.min()).days
    assert span_days >= 150, f"Timestamp span {span_days} days < 5 months"

    # order_value > 0
    assert (df['order_value'] > 0).all(), "order_value must be > 0"

    # quantity >= 1 integer
    assert (df['quantity'] >= 1).all(), "quantity must be >= 1"

    # discount_pct in [0, 70]
    assert (df['discount_pct'] >= 0).all() and (df['discount_pct'] <= 70).all(), \
        "discount_pct out of [0, 70]"

    # payment_method
    pm_set = set(df['payment_method'].unique())
    assert pm_set.issubset(ALLOWED_PAYMENT), f"Invalid payment methods: {pm_set - ALLOWED_PAYMENT}"
    cod_share = (df['payment_method'] == 'COD').mean()
    assert 0.30 <= cod_share <= 0.65, f"COD share {cod_share:.3f} outside [0.30, 0.65]"

    # cod_charge: 0 for non-COD, [20, 100] for COD
    non_cod = df[df['payment_method'] != 'COD']
    assert (non_cod['cod_charge'] == 0).all(), "Non-COD orders have non-zero cod_charge"
    cod_rows = df[df['payment_method'] == 'COD']
    assert (cod_rows['cod_charge'] >= 20).all() and (cod_rows['cod_charge'] <= 100).all(), \
        "COD orders cod_charge outside [20, 100]"

    # account_age_days
    assert (df['account_age_days'] >= 0).all(), "account_age_days < 0"
    assert (df['account_age_days'] <= 3650).all(), "account_age_days > 3650"
    # If age == 0 then prior_orders == 0
    new_cust = df[df['account_age_days'] == 0]
    if len(new_cust) > 0:
        assert (new_cust['prior_orders'] == 0).all(), \
            "account_age_days == 0 but prior_orders != 0"

    # prior_orders >= 0; prior_rto_count >= 0 and <= prior_orders
    assert (df['prior_orders'] >= 0).all(), "prior_orders < 0"
    assert (df['prior_rto_count'] >= 0).all(), "prior_rto_count < 0"
    assert (df['prior_rto_count'] <= df['prior_orders']).all(), \
        "prior_rto_count > prior_orders"

    # pincode: string, 6-char
    assert df['pincode'].dtype == object, "pincode must be string dtype"
    assert (df['pincode'].str.len() == 6).all(), "pincode must be 6 digits"
    n_pincodes = df['pincode'].nunique()
    if expected_rows >= 10000:
        assert 800 <= n_pincodes <= 1500, f"Distinct pincodes {n_pincodes} outside [800, 1500]"
    else:
        assert n_pincodes > 1, f"Only {n_pincodes} distinct pincodes in small dataset"

    # courier_id
    assert df['courier_id'].dtype == object, "courier_id must be object dtype"
    n_couriers = df['courier_id'].nunique()
    assert 4 <= n_couriers <= 6, f"Distinct couriers {n_couriers} outside [4, 6]"

    # pincode_tier
    assert set(df['pincode_tier'].unique()).issubset({1, 2, 3}), "pincode_tier not in {1,2,3}"

    # historical_pincode_rto_rate
    assert (df['historical_pincode_rto_rate'] >= 0).all() and \
           (df['historical_pincode_rto_rate'] <= 1).all(), \
        "historical_pincode_rto_rate outside [0, 1]"
    rate_mean = df['historical_pincode_rto_rate'].mean()
    assert 0.10 <= rate_mean <= 0.35, f"Mean historical_pincode_rto_rate {rate_mean:.4f} outside [0.10, 0.35]"

    # orders_last_24h >= 0; device_cluster_size >= 1
    assert (df['orders_last_24h'] >= 0).all(), "orders_last_24h < 0"
    assert (df['device_cluster_size'] >= 1).all(), "device_cluster_size < 1"

    # rto_label binary, prevalence
    assert set(df['rto_label'].unique()).issubset({0, 1}), "rto_label not binary"
    rto_rate = df['rto_label'].mean()
    assert 0.15 <= rto_rate <= 0.30, f"RTO rate {rto_rate:.4f} outside [0.15, 0.30]"

    # Leakage sanity: |r| between per-pincode mean(rto_label) and
    # per-pincode mean(historical_pincode_rto_rate) should be < 0.95
    pin_agg = df.groupby('pincode').agg(
        rto_mean=('rto_label', 'mean'),
        hist_mean=('historical_pincode_rto_rate', 'mean')
    )
    if len(pin_agg) > 2:
        corr = pin_agg['rto_mean'].corr(pin_agg['hist_mean'])
        assert abs(corr) < 0.95, \
            f"Leakage concern: pincode-level |r| = {abs(corr):.4f} >= 0.95"

    return True


# ─── Generator ─────────────────────────────────────────────────────────
def generate(n_rows=100000, seed=42):
    """
    Generate a synthetic e-commerce order dataset using a probabilistic
    latent-risk logistic-Bernoulli process. Returns a pd.DataFrame with
    exactly 19 columns. No file I/O.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Pincode pool ──────────────────────────────────────────────
    pincode_pool = _build_pincode_pool(rng, n_pincodes=1000)

    # ── 2. Order IDs ─────────────────────────────────────────────────
    order_ids = [f"ORD{i+1:09d}" for i in range(n_rows)]

    # ── 3. Timestamps: ~6 months ending near current date, sorted ───
    start_time = pd.Timestamp("2026-03-01 00:00:00")
    end_time = pd.Timestamp("2026-09-03 23:59:59")
    delta_seconds = int((end_time - start_time).total_seconds())
    random_seconds = np.sort(rng.integers(0, delta_seconds, size=n_rows))
    timestamps = pd.to_datetime(start_time) + pd.to_timedelta(random_seconds, unit='s')

    # ── 4. Customer pool (~30,000 distinct) ──────────────────────────
    num_customers = 30000
    customer_pool = [f"CUST{i+1:06d}" for i in range(num_customers)]

    # Latent customer attributes (stable per customer)
    cust_account_age = rng.lognormal(mean=5.5, sigma=1.0, size=num_customers).astype(int)
    cust_account_age = np.clip(cust_account_age, 0, 3650)
    # Ensure some new customers (age 0): ~3%
    new_mask = rng.random(num_customers) < 0.03
    cust_account_age[new_mask] = 0

    cust_prior_orders_base = rng.poisson(lam=2.5, size=num_customers)
    # New customers must have 0 prior orders
    cust_prior_orders_base[cust_account_age == 0] = 0

    cust_rto_propensity = rng.beta(a=1.5, b=6.0, size=num_customers)

    cust_df = pd.DataFrame({
        'customer_id': customer_pool,
        '_account_age_days': cust_account_age,
        '_prior_orders_base': cust_prior_orders_base,
        '_rto_propensity': cust_rto_propensity,
    })

    # ── 5. Map customers to orders ───────────────────────────────────
    customer_ids = rng.choice(customer_pool, size=n_rows)
    temp = pd.DataFrame({'customer_id': customer_ids, '_row': np.arange(n_rows)})
    merged = temp.merge(cust_df, on='customer_id', how='left').sort_values('_row')

    account_age_days = merged['_account_age_days'].values
    prior_orders_base = merged['_prior_orders_base'].values
    rto_prop = merged['_rto_propensity'].values

    # Per-row noise on prior_orders (latent per-row attribute, §6)
    prior_orders = prior_orders_base + rng.integers(0, 3, size=n_rows)
    prior_orders[account_age_days == 0] = 0
    prior_orders = np.clip(prior_orders, 0, 200)

    # prior_rto_count: binomial bounded by prior_orders
    prior_rto_count = rng.binomial(n=prior_orders, p=np.clip(rto_prop * 0.3, 0, 0.8))

    # ── 6. Pincodes, tiers, historical rate ──────────────────────────
    pin_indices = rng.integers(0, len(pincode_pool), size=n_rows)
    pincodes_sampled = pincode_pool['pincode'].values[pin_indices]
    pincode_tiers = pincode_pool['pincode_tier'].values[pin_indices]
    theta_pincode = pincode_pool['theta_pincode'].values[pin_indices]

    # Per-order feature = clip(theta + noise, 0, 1) — NOT from rto_label
    hist_pincode_rto_rate = np.clip(
        theta_pincode + rng.normal(0, 0.03, size=n_rows), 0.0, 1.0
    )
    hist_pincode_rto_rate = np.round(hist_pincode_rto_rate, 4)

    # ── 7. Couriers ──────────────────────────────────────────────────
    couriers = ["Courier_A", "Courier_B", "Courier_C", "Courier_D", "Courier_E"]
    courier_ids = rng.choice(couriers, size=n_rows, p=[0.30, 0.25, 0.20, 0.15, 0.10])

    # ── 8. Category ──────────────────────────────────────────────────
    categories = ["Electronics", "Apparel", "Footwear", "Beauty", "Home", "Jewelry"]
    cat_probs = [0.20, 0.30, 0.15, 0.15, 0.12, 0.08]
    sampled_categories = rng.choice(categories, size=n_rows, p=cat_probs)

    # ── 9. Order value: right-skewed, median ≈ ₹600, tail to ~₹15k ──
    order_values = rng.lognormal(mean=6.4, sigma=0.8, size=n_rows)
    order_values = np.clip(order_values, 50.0, 15000.0)
    order_values = np.round(order_values, 2)

    # ── 10. Quantity: Poisson(1.2) + 1, capped ~10 ──────────────────
    quantities = rng.poisson(lam=1.2, size=n_rows) + 1
    quantities = np.clip(quantities, 1, 10).astype(int)

    # ── 11. Discount: skewed toward small (50% zero, rest Beta*70) ──
    discount_pcts = np.zeros(n_rows)
    has_discount = rng.random(n_rows) > 0.50
    n_discounted = has_discount.sum()
    discount_pcts[has_discount] = rng.beta(a=2, b=5, size=n_discounted) * 70.0
    discount_pcts = np.round(discount_pcts, 2)

    # ── 12. Payment method: COD 40-55% ──────────────────────────────
    payment_methods = ["COD", "UPI", "Credit Card", "Debit Card", "Net Banking"]
    pm_probs = [0.48, 0.28, 0.12, 0.08, 0.04]
    sampled_pm = rng.choice(payment_methods, size=n_rows, p=pm_probs)

    # ── 13. COD charge: 0 for non-COD, [20, 100] for COD ────────────
    cod_charges = np.zeros(n_rows, dtype=float)
    cod_mask = (sampled_pm == "COD")
    cod_charge_options = [20.0, 29.0, 39.0, 49.0, 59.0, 79.0, 99.0]
    cod_charge_probs = [0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]
    cod_charges[cod_mask] = rng.choice(
        cod_charge_options, size=cod_mask.sum(), p=cod_charge_probs
    )

    # ── 14. Behavioral ───────────────────────────────────────────────
    orders_last_24h = rng.poisson(lam=1.5, size=n_rows)
    orders_last_24h = np.clip(orders_last_24h, 0, 25).astype(int)

    # device_cluster_size: 1 at 70%, 2-5 at 25%, 6+ at 5%
    dcs = np.ones(n_rows, dtype=int)
    r = rng.random(n_rows)
    mid_mask = (r >= 0.70) & (r < 0.95)
    high_mask = (r >= 0.95)
    dcs[mid_mask] = rng.integers(2, 6, size=mid_mask.sum())
    dcs[high_mask] = rng.integers(6, 21, size=high_mask.sum())

    # ── 15. Latent risk score z (logistic-Bernoulli) ─────────────────
    is_cod = (sampled_pm == "COD").astype(float)

    # Documented starting intercept. Tuned to land RTO rate in [0.15, 0.30].
    # Starting point: -2.5 per spec. Adjusted to -2.7 after testing.
    beta_0 = -2.7

    # Category risk
    cat_risk_map = {
        "Apparel": 0.30, "Electronics": 0.20, "Footwear": 0.15,
        "Beauty": 0.05, "Home": -0.10, "Jewelry": 0.08
    }
    cat_risk = np.array([cat_risk_map[c] for c in sampled_categories])

    # Courier risk (modest)
    courier_risk_map = {
        "Courier_A": 0.0, "Courier_B": 0.05, "Courier_C": -0.05,
        "Courier_D": 0.10, "Courier_E": 0.15
    }
    courier_risk = np.array([courier_risk_map[c] for c in courier_ids])

    # Pincode tier risk (modest)
    tier_risk = np.where(pincode_tiers == 3, 0.30,
                np.where(pincode_tiers == 2, 0.10, -0.15))

    z = (
        beta_0
        + 1.10 * is_cod                                     # COD increases risk
        + 0.80 * np.log1p(prior_rto_count)                   # prior RTO history
        + 2.50 * hist_pincode_rto_rate                       # pincode historical risk
        + 0.25 * np.log1p(orders_last_24h)                   # velocity
        + 0.20 * np.log1p(dcs)                               # device cluster
        - 0.12 * np.log1p(account_age_days)                  # mature accounts lower risk
        - 0.15 * np.log1p(prior_orders)                      # experienced buyers
        + cat_risk                                           # category
        + courier_risk                                       # courier
        + tier_risk                                          # tier
        + 0.005 * discount_pcts                              # discount
        + rng.normal(0, 0.80, size=n_rows)                   # stochastic noise (std ≈ 0.8)
    )

    p_rto = 1.0 / (1.0 + np.exp(-z))
    rto_label = rng.binomial(n=1, p=p_rto)

    # ── 16. Assemble DataFrame (exactly 19 columns, spec order) ──────
    df = pd.DataFrame({
        'order_id': order_ids,
        'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S'),
        'order_value': order_values,
        'quantity': quantities,
        'category': sampled_categories,
        'discount_pct': discount_pcts,
        'payment_method': sampled_pm,
        'cod_charge': cod_charges,
        'customer_id': customer_ids,
        'account_age_days': account_age_days.astype(int),
        'prior_orders': prior_orders.astype(int),
        'prior_rto_count': prior_rto_count.astype(int),
        'pincode': pincodes_sampled,
        'courier_id': courier_ids,
        'pincode_tier': pincode_tiers.astype(int),
        'historical_pincode_rto_rate': hist_pincode_rto_rate,
        'orders_last_24h': orders_last_24h,
        'device_cluster_size': dcs,
        'rto_label': rto_label.astype(int),
    })

    # Ensure pincode is string dtype
    df['pincode'] = df['pincode'].astype(str)

    return df


# ─── Main (CLI + file I/O) ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RTO Shield — Generate synthetic order dataset"
    )
    parser.add_argument("--rows", type=int, default=100000,
                        help="Number of rows (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str,
                        default="data/raw/synthetic_orders.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # Generate
    df = generate(n_rows=args.rows, seed=args.seed)

    # Validate (fail loudly)
    validate(df, expected_rows=args.rows)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    # Print summary (spec format)
    rto_count = int((df['rto_label'] == 1).sum())
    del_count = int((df['rto_label'] == 0).sum())
    total = len(df)

    print("RTO Shield Synthetic Dataset")
    print("----------------------------")
    print(f"Rows: {total}")
    print(f"Columns: {df.shape[1]}")
    print(f"RTO: {rto_count} ({rto_count/total*100:.1f}%)")
    print(f"Delivered: {del_count} ({del_count/total*100:.1f}%)")
    print(f"Output: {args.output}")
    print(f"Validation: PASSED")
    print()
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
