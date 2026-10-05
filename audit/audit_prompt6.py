import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from src.data.generator import generate

def get_latent_and_data(n_rows=100000, seed=42):
    rng = np.random.default_rng(seed)
    # We replicate generator step-by-step to get exact z and p_rto
    from src.data.generator import _build_pincode_pool
    pincode_pool = _build_pincode_pool(rng, n_pincodes=1000)
    order_ids = [f"ORD{i+1:09d}" for i in range(n_rows)]
    start_time = pd.Timestamp("2026-03-01 00:00:00")
    end_time = pd.Timestamp("2026-09-03 23:59:59")
    delta_seconds = int((end_time - start_time).total_seconds())
    random_seconds = np.sort(rng.integers(0, delta_seconds, size=n_rows))
    timestamps = pd.to_datetime(start_time) + pd.to_timedelta(random_seconds, unit='s')
    num_customers = 30000
    customer_pool = [f"CUST{i+1:06d}" for i in range(num_customers)]
    cust_account_age = rng.lognormal(mean=5.5, sigma=1.0, size=num_customers).astype(int)
    cust_account_age = np.clip(cust_account_age, 0, 3650)
    new_mask = rng.random(num_customers) < 0.03
    cust_account_age[new_mask] = 0
    cust_prior_orders_base = rng.poisson(lam=2.5, size=num_customers)
    cust_prior_orders_base[cust_account_age == 0] = 0
    cust_rto_propensity = rng.beta(a=1.5, b=6.0, size=num_customers)
    cust_df = pd.DataFrame({
        'customer_id': customer_pool,
        '_account_age_days': cust_account_age,
        '_prior_orders_base': cust_prior_orders_base,
        '_rto_propensity': cust_rto_propensity,
    })
    customer_ids = rng.choice(customer_pool, size=n_rows)
    temp = pd.DataFrame({'customer_id': customer_ids, '_row': np.arange(n_rows)})
    merged = temp.merge(cust_df, on='customer_id', how='left').sort_values('_row')
    account_age_days = merged['_account_age_days'].values
    prior_orders_base = merged['_prior_orders_base'].values
    rto_prop = merged['_rto_propensity'].values
    prior_orders = prior_orders_base + rng.integers(0, 3, size=n_rows)
    prior_orders[account_age_days == 0] = 0
    prior_orders = np.clip(prior_orders, 0, 200)
    prior_rto_count = rng.binomial(n=prior_orders, p=np.clip(rto_prop * 0.3, 0, 0.8))
    pin_indices = rng.integers(0, len(pincode_pool), size=n_rows)
    pincodes_sampled = pincode_pool['pincode'].values[pin_indices]
    pincode_tiers = pincode_pool['pincode_tier'].values[pin_indices]
    theta_pincode = pincode_pool['theta_pincode'].values[pin_indices]
    hist_pincode_rto_rate = np.clip(theta_pincode + rng.normal(0, 0.03, size=n_rows), 0.0, 1.0)
    hist_pincode_rto_rate = np.round(hist_pincode_rto_rate, 4)
    couriers = ["Courier_A", "Courier_B", "Courier_C", "Courier_D", "Courier_E"]
    courier_ids = rng.choice(couriers, size=n_rows, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    categories = ["Electronics", "Apparel", "Footwear", "Beauty", "Home", "Jewelry"]
    cat_probs = [0.20, 0.30, 0.15, 0.15, 0.12, 0.08]
    sampled_categories = rng.choice(categories, size=n_rows, p=cat_probs)
    order_values = rng.lognormal(mean=6.4, sigma=0.8, size=n_rows)
    order_values = np.clip(order_values, 50.0, 15000.0)
    order_values = np.round(order_values, 2)
    quantities = rng.poisson(lam=1.2, size=n_rows) + 1
    quantities = np.clip(quantities, 1, 10).astype(int)
    discount_pcts = np.zeros(n_rows)
    has_discount = rng.random(n_rows) > 0.50
    n_discounted = has_discount.sum()
    discount_pcts[has_discount] = rng.beta(a=2, b=5, size=n_discounted) * 70.0
    discount_pcts = np.round(discount_pcts, 2)
    payment_methods = ["COD", "UPI", "Credit Card", "Debit Card", "Net Banking"]
    pm_probs = [0.48, 0.28, 0.12, 0.08, 0.04]
    sampled_pm = rng.choice(payment_methods, size=n_rows, p=pm_probs)
    cod_charges = np.zeros(n_rows, dtype=float)
    cod_mask = (sampled_pm == "COD")
    cod_charge_options = [20.0, 29.0, 39.0, 49.0, 59.0, 79.0, 99.0]
    cod_charge_probs = [0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]
    cod_charges[cod_mask] = rng.choice(cod_charge_options, size=cod_mask.sum(), p=cod_charge_probs)
    orders_last_24h = rng.poisson(lam=1.5, size=n_rows)
    orders_last_24h = np.clip(orders_last_24h, 0, 25).astype(int)
    dcs = np.ones(n_rows, dtype=int)
    r = rng.random(n_rows)
    mid_mask = (r >= 0.70) & (r < 0.95)
    high_mask = (r >= 0.95)
    dcs[mid_mask] = rng.integers(2, 6, size=mid_mask.sum())
    dcs[high_mask] = rng.integers(6, 21, size=high_mask.sum())
    is_cod = (sampled_pm == "COD").astype(float)
    beta_0 = -2.7
    cat_risk_map = {"Apparel": 0.30, "Electronics": 0.20, "Footwear": 0.15, "Beauty": 0.05, "Home": -0.10, "Jewelry": 0.08}
    cat_risk = np.array([cat_risk_map[c] for c in sampled_categories])
    courier_risk_map = {"Courier_A": 0.0, "Courier_B": 0.05, "Courier_C": -0.05, "Courier_D": 0.10, "Courier_E": 0.15}
    courier_risk = np.array([courier_risk_map[c] for c in courier_ids])
    tier_risk = np.where(pincode_tiers == 3, 0.30, np.where(pincode_tiers == 2, 0.10, -0.15))
    noise = rng.normal(0, 0.80, size=n_rows)
    z = (beta_0 + 1.10 * is_cod + 0.80 * np.log1p(prior_rto_count) + 2.50 * hist_pincode_rto_rate
         + 0.25 * np.log1p(orders_last_24h) + 0.20 * np.log1p(dcs) - 0.12 * np.log1p(account_age_days)
         - 0.15 * np.log1p(prior_orders) + cat_risk + courier_risk + tier_risk + 0.005 * discount_pcts + noise)
    p_rto = 1.0 / (1.0 + np.exp(-z))
    rto_label = rng.binomial(n=1, p=p_rto)
    
    df = pd.DataFrame({
        'order_id': order_ids, 'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S'),
        'order_value': order_values, 'quantity': quantities, 'category': sampled_categories,
        'discount_pct': discount_pcts, 'payment_method': sampled_pm, 'cod_charge': cod_charges,
        'customer_id': customer_ids, 'account_age_days': account_age_days.astype(int),
        'prior_orders': prior_orders.astype(int), 'prior_rto_count': prior_rto_count.astype(int),
        'pincode': pincodes_sampled, 'courier_id': courier_ids, 'pincode_tier': pincode_tiers.astype(int),
        'historical_pincode_rto_rate': hist_pincode_rto_rate, 'orders_last_24h': orders_last_24h,
        'device_cluster_size': dcs, 'rto_label': rto_label.astype(int)
    })
    return df, z, p_rto

def audit_seed(seed=42):
    df, z, p_rto = get_latent_and_data(100000, seed)
    # Chronological split: test is last 15,000
    test_df = df.iloc[85000:].copy()
    test_p_rto = p_rto[85000:]
    y_test = test_df['rto_label'].values
    
    # Calculate Bayes Ceiling PR-AUC on test set
    prec, rec, _ = precision_recall_curve(y_test, test_p_rto)
    bayes_pr_auc = auc(rec, prec)
    bayes_roc_auc = roc_auc_score(y_test, test_p_rto)
    
    base_rate = df['rto_label'].mean()
    cod_rto = df[df['payment_method'] == 'COD']['rto_label'].mean()
    prepaid_rto = df[df['payment_method'] != 'COD']['rto_label'].mean()
    
    return {
        'seed': seed,
        'base_rate': base_rate,
        'cod_rto': cod_rto,
        'prepaid_rto': prepaid_rto,
        'test_base_rate': y_test.mean(),
        'bayes_pr_auc': bayes_pr_auc,
        'bayes_roc_auc': bayes_roc_auc
    }

print("=== SEED SENSITIVITY (5 SEEDS) ===")
seeds = [42, 101, 2024, 777, 999]
results = []
for s in seeds:
    res = audit_seed(s)
    results.append(res)
    print(f"Seed {s}: Base RTO={res['base_rate']:.4f}, COD RTO={res['cod_rto']:.4f}, Prepaid RTO={res['prepaid_rto']:.4f}, Bayes PR-AUC={res['bayes_pr_auc']:.4f}, Bayes ROC-AUC={res['bayes_roc_auc']:.4f}")

res_df = pd.DataFrame(results)
print("\nSummary Stats:")
print(res_df.describe().to_string())

# Feature correlations on Seed 42
df42, z42, p42 = get_latent_and_data(100000, 42)
print("\n=== FEATURE CORRELATIONS WITH LABEL AND LATENT PROB (SEED 42) ===")
num_cols = ['order_value', 'quantity', 'discount_pct', 'account_age_days', 
            'prior_orders', 'prior_rto_count', 'pincode_tier', 
            'historical_pincode_rto_rate', 'orders_last_24h', 'device_cluster_size']
corrs = []
for c in num_cols:
    r_label = np.corrcoef(df42[c], df42['rto_label'])[0, 1]
    r_latent = np.corrcoef(df42[c], p42)[0, 1]
    corrs.append({'feature': c, 'corr_with_label': round(r_label, 4), 'corr_with_latent_p': round(r_latent, 4)})
print(pd.DataFrame(corrs).to_string())
