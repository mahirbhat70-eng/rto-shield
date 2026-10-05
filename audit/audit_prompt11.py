import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, ".")
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action

test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
model = joblib.load("models/tree_model_calibrated.pkl")
engine = CostEngine()

lookup_df = pd.read_csv("data/processed/pincode_rate_lookup.csv", dtype={'pincode': str})
pincode_map = lookup_df.set_index('pincode').to_dict(orient='index')
global_mean_rate = float(lookup_df['historical_pincode_rto_rate'].mean())
global_modal_tier = int(lookup_df['pincode_tier'].mode().iloc[0])

print(f"Fallback prior: rate={global_mean_rate:.4f}, tier={global_modal_tier}")

def eval_test_df(df):
    X = df.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
    proba = model.predict_proba(X)[:, 1]
    y = df['rto_label'].values
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    
    # COD policy savings
    df_cod, proba_cod = get_cod_subset(df, proba)
    y_cod = df_cod['rto_label'].values
    actions, policy_losses = eval_multi_action(df_cod, proba_cod, engine)
    
    baseline_losses = 150.0 * proba_cod
    expected_savings = np.sum(baseline_losses) - np.sum(policy_losses)
    
    # Realized savings
    realized_base = np.sum(y_cod * 150.0)
    realized_pol = []
    for i, r in df_cod.reset_index(drop=True).iterrows():
        act = actions[i]
        yc = y_cod[i]
        V = r['order_value']
        if act == "ALLOW_COD":
            realized_pol.append(yc * 150.0)
        elif act == "VERIFY_ADDRESS":
            loss = (0.85 * 0.80 * 150.0 + 1.50) if yc == 1 else (0.15 * 0.25 * V + 1.50)
            realized_pol.append(loss)
        elif act == "REQUIRE_DEPOSIT":
            loss = (0.60 * 0.20 * 150.0 + 0.50) if yc == 1 else (0.40 * 0.25 * V + 0.50)
            realized_pol.append(loss)
        elif act == "PREPAID_ONLY":
            realized_pol.append(0.85 * 0.25 * V)
    realized_savings = realized_base - np.sum(realized_pol)
    return pr_auc, brier, expected_savings, realized_savings

# 1. Cold start stress test (0%, 12%, 30%, 60% unseen pincodes)
print("\n=== 1. COLD-START UNSEEN PINCODE STRESS TEST ===")
for unseen_pct in [0.0, 0.12, 0.30, 0.60]:
    df_mod = test.copy()
    if unseen_pct > 0.0:
        rng = np.random.default_rng(42)
        unique_pins = df_mod['pincode'].unique()
        unseen_pins = set(rng.choice(unique_pins, size=int(len(unique_pins) * unseen_pct), replace=False))
        mask = df_mod['pincode'].isin(unseen_pins)
        df_mod.loc[mask, 'historical_pincode_rto_rate'] = global_mean_rate
        df_mod.loc[mask, 'pincode_tier'] = global_modal_tier
    pr, br, es, rs = eval_test_df(df_mod)
    print(f"Unseen Pincodes {unseen_pct*100:4.1f}%: PR-AUC={pr:.4f}, Brier={br:.4f}, Realized Savings=Rs {rs:8.2f}")

# 2. Prior shift: Overall vs COD-subset base rates
print("\n=== 2. PRIOR SHIFT (RESAMPLING TEST SET RTO RATES) ===")
print("Note: In test set, Overall Platform RTO is 19.78% while COD-subset RTO is 28.27%.")
print("We evaluate both: (A) Shifting overall population, (B) Shifting COD subset alone.\n")

pos_idx = np.where(test['rto_label'] == 1)[0]
neg_idx = np.where(test['rto_label'] == 0)[0]
rng = np.random.default_rng(42)

print("--- Sweep 2A: Resampling Overall Platform Orders ---")
for target_overall_rate in [0.15, 0.20, 0.28, 0.36, 0.45]:
    n_neg = len(neg_idx)
    n_pos = int(n_neg * target_overall_rate / (1.0 - target_overall_rate))
    sampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
    resampled_idx = np.concatenate([neg_idx, sampled_pos])
    df_resampled = test.iloc[resampled_idx].reset_index(drop=True)
    actual_ov = df_resampled['rto_label'].mean()
    actual_cod = df_resampled.loc[df_resampled['payment_method'] == 'COD', 'rto_label'].mean()
    pr, br, es, rs = eval_test_df(df_resampled)
    beats_base = rs > 0
    print(f"Overall Target {target_overall_rate*100:4.1f}% (Actual Overall: {actual_ov*100:4.1f}%, Actual COD: {actual_cod*100:4.1f}%): "
          f"PR-AUC={pr:.4f}, Brier={br:.4f}, Realized Savings=Rs {rs:9.2f}, Beats Baseline: {beats_base}")

print("\n--- Sweep 2B: Resampling COD-Subset Orders Directly ---")
cod_pos_idx = np.where((test['payment_method'] == 'COD') & (test['rto_label'] == 1))[0]
cod_neg_idx = np.where((test['payment_method'] == 'COD') & (test['rto_label'] == 0))[0]
for target_cod_rate in [0.10, 0.15, 0.168, 0.20, 0.2827, 0.35]:
    n_neg = len(cod_neg_idx)
    n_pos = int(n_neg * target_cod_rate / (1.0 - target_cod_rate))
    sampled_pos = rng.choice(cod_pos_idx, size=n_pos, replace=True)
    resampled_idx = np.concatenate([cod_neg_idx, sampled_pos])
    df_resampled = test.iloc[resampled_idx].reset_index(drop=True)
    actual_cod = df_resampled['rto_label'].mean()
    pr, br, es, rs = eval_test_df(df_resampled)
    beats_base = rs > 0
    print(f"COD Target {target_cod_rate*100:4.1f}% (Actual COD: {actual_cod*100:4.1f}%): "
          f"PR-AUC={pr:.4f}, Brier={br:.4f}, Realized Savings=Rs {rs:9.2f}, Beats Baseline: {beats_base}")

# 3. Covariate shift: Order value +/- 30%
print("\n=== 3. COVARIATE SHIFT (ORDER VALUE +/- 30%) ===")
for shift_name, factor in [("-30% Order Value", 0.70), ("Baseline Order Value", 1.00), ("+30% Order Value", 1.30)]:
    df_shift = test.copy()
    df_shift['order_value'] = df_shift['order_value'] * factor
    pr, br, es, rs = eval_test_df(df_shift)
    print(f"{shift_name:22s}: PR-AUC={pr:.4f}, Realized Savings=Rs {rs:8.2f}")

# 4. Sudden complete loss of pincode lookup table
print("\n=== 4. COMPLETE LOSS OF PINCODE LOOKUP (100% FALLBACK) ===")
df_no_lookup = test.copy()
df_no_lookup['historical_pincode_rto_rate'] = global_mean_rate
df_no_lookup['pincode_tier'] = global_modal_tier
pr, br, es, rs = eval_test_df(df_no_lookup)
print(f"100% Fallback Prior: PR-AUC={pr:.4f}, Brier={br:.4f}, Realized Savings=Rs {rs:8.2f} (vs Baseline Rs 69,786)")
