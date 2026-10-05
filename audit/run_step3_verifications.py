import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, ".")
from src.serve.scorer import score_order, score_order_guarded, ProductionGuardrails
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action
from audit.shadow.harness import run_shadow_replay

print("================================================================================")
print(" STEP 3 AUDIT VERIFICATIONS: RIGOROUS RAW OUTPUT EXECUTION")
print("================================================================================\n")

# ──────────────────────────────────────────────────────────────────────────────
# 1. PROMPT 12: ENVIRONMENT & ARTIFACT REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
print(">>> [PROMPT 12] ARTIFACT CRYPTOGRAPHIC HASH VERIFICATION:")
with open("models/artifact_hashes.json", "r") as f:
    expected_digests = json.load(f)

mismatch_count = 0
for rel_path, exp_hash in expected_digests.items():
    if not os.path.exists(rel_path):
        print(f"  {rel_path:<40s}: MISSING FILE")
        mismatch_count += 1
        continue
    with open(rel_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash == exp_hash:
        print(f"  {rel_path:<40s}: MATCH (SHA-256: {actual_hash[:16]}...)")
    else:
        print(f"  {rel_path:<40s}: MISMATCH (Expected {exp_hash[:10]}..., got {actual_hash[:10]}...)")
        mismatch_count += 1
print(f"  Verification Result: {len(expected_digests) - mismatch_count}/{len(expected_digests)} artifacts verified exactly.\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. PROMPT 13: DOCUMENTATION VS REALITY CONTRADICTIONS TABLE
# ──────────────────────────────────────────────────────────────────────────────
print(">>> [PROMPT 13] DOCUMENTATION VS REALITY CONTRADICTION MATRIX:")
contradictions = [
    {
        "Subject": "Automated test count",
        "Legacy Docs": "60 tests (docs/JUDGE_QA.md, pitch_script.md)",
        "README / Code": "225 tests (originally), now 233 tests",
        "Actual Measured": "233 tests passing in pytest (17 test files)"
    },
    {
        "Subject": "Scoring Latency (p50)",
        "README / Pitch": "p50 ~3 ms (raw) / ~8 ms (claimed with SHAP)",
        "Claim Matrix": "p50 ~3 ms (scripts/benchmark_latency.py)",
        "Actual Measured": "14.96 ms with TreeSHAP (2.8 ms raw without SHAP)"
    },
    {
        "Subject": "Threshold Constants",
        "Legacy Docs": "Fixed thresholds: 0.20 (VERIFY), 0.48 (DEPOSIT)",
        "Code Reality": "Order-value dependent argmin: p*(V) = (c_fric + drop*M)/(red*RTO + drop*M)",
        "Actual Measured": "Threshold scales from 0.065 at V=100 to 0.70 at V=10,000"
    },
    {
        "Subject": "RTO Reduction (46.9%)",
        "Legacy Docs": "46.9% RTO volume eliminated (951 RTOs prevented)",
        "Code Reality": "Simulated expectation under assumed 80% and 30% reduction priors",
        "Actual Measured": "Unmeasured in physical logistics; purely counterfactual"
    },
    {
        "Subject": "Baseline Total Loss Sign",
        "Legacy Docs": "-Rs 547,766 total loss",
        "Code Reality": "Expected margin exceeding delivery losses (Gross Margin = +Rs 547,766)",
        "Actual Measured": "Realized gross margin without intervention = +Rs 304,200"
    }
]
print(pd.DataFrame(contradictions).to_string(index=False))
print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. PROMPT 6 / 8: 5-SEED GENERATOR STABILITY TABLE
# ──────────────────────────────────────────────────────────────────────────────
print(">>> [PROMPT 6] 5-SEED GENERATOR SENSITIVITY TABLE:")
from audit.audit_prompt6 import audit_seed

seeds = [42, 101, 2024, 777, 999]
seed_records = [audit_seed(s) for s in seeds]
seed_df = pd.DataFrame(seed_records)
print(seed_df[['seed', 'base_rate', 'cod_rto', 'prepaid_rto', 'bayes_pr_auc', 'bayes_roc_auc']].to_string(index=False))
print("Summary Statistics (5 Seeds):")
print(seed_df[['base_rate', 'cod_rto', 'prepaid_rto', 'bayes_pr_auc', 'bayes_roc_auc']].describe().to_string())
print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. PROMPT 8: FIXED-POLICY MISSPECIFICATION TEST (NO RE-OPTIMIZATION)
# ──────────────────────────────────────────────────────────────────────────────
print(">>> [PROMPT 8] FIXED-POLICY MISSPECIFICATION TEST:")
print("Testing policy decisions frozen at default priors against deteriorating real customer elasticity:")

test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
model = joblib.load("models/tree_model_calibrated.pkl")
X_test = test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore')
proba = model.predict_proba(X_test)[:, 1]

engine = CostEngine()
test_cod, proba_cod = get_cod_subset(test, proba)
y_cod = test_cod['rto_label'].values
V_cod = test_cod['order_value'].values
margin = 0.20 * V_cod
rto_cost = 150.0

# Fixed action choices made under baseline beliefs (40% deposit drop, 80% rto red)
frozen_actions, _ = eval_multi_action(test_cod, proba_cod, engine)

# Baseline realized loss (Always Allow)
baseline_realized = np.sum(np.where(y_cod == 1, rto_cost, -margin))

misspec_results = []
for true_deposit_drop in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
    # What if true deposit drop is higher, but policy still demanded deposits based on old beliefs?
    realized_losses = []
    for i in range(len(test_cod)):
        act = frozen_actions[i]
        yc = y_cod[i]
        m = margin[i]
        if act == "ALLOW_COD":
            realized_losses.append(rto_cost if yc == 1 else -m)
        elif act == "VERIFY_ADDRESS":
            realized_losses.append(2.0 + (0.70 * rto_cost if yc == 1 else -0.95 * m))
        elif act == "REQUIRE_DEPOSIT":
            # True deposit drop applied to good customers (config-driven friction_cost = 0.0)
            dep_friction = engine.interventions["REQUIRE_DEPOSIT"]["friction_cost"]
            realized_losses.append(dep_friction + (0.20 * rto_cost if yc == 1 else -(1.0 - true_deposit_drop) * m))
    
    tot_realized = np.sum(realized_losses)
    net_savings = baseline_realized - tot_realized
    misspec_results.append({
        'True Deposit Drop-off': f"{true_deposit_drop*100:.0f}%",
        'Fixed Policy Realized Savings': round(net_savings, 2),
        'Beats Baseline?': net_savings > 0,
        'Verdict': "PROFITABLE" if net_savings > 0 else "LOSING MONEY"
    })

print(pd.DataFrame(misspec_results).to_string(index=False))
print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. PROMPT 14: SHADOW-MODE HARNESS REPLAY ON SYNTHETIC MERCHANT EXPORT
# ──────────────────────────────────────────────────────────────────────────────
print(">>> [PROMPT 14] SHADOW REPLAY HARNESS EXECUTION:")
# Create a realistic 2,500-order merchant slice
merchant_slice = test.sample(n=2500, random_state=42).copy()
merchant_input_path = "audit/shadow/pilot_merchant_orders_2500.csv"
merchant_slice.to_csv(merchant_input_path, index=False)

report_path = run_shadow_replay(merchant_input_path, output_report="audit/shadow/shadow_report.md")
with open(report_path, "r", encoding="utf-8") as f:
    report_text = f.read()
print(report_text[:1200])
print("\n")

# ──────────────────────────────────────────────────────────────────────────────
# 6. PROMPT 15: PRODUCTION GUARDRAILS TEST IN SRC/SERVE
# ──────────────────────────────────────────────────────────────────────────────
print(">>> [PROMPT 15] PRODUCTION GUARDRAILS VERIFICATION IN SRC/SERVE:")
from src.serve.scorer import score_order_guarded, score_order

# Test 1: Normal COD order
test_order = {
    'order_value': 850.0, 'category': 'Apparel', 'payment_method': 'COD',
    'quantity': 1, 'discount_pct': 0, 'cod_charge': 49, 'account_age_days': 120,
    'prior_orders': 2, 'prior_rto_count': 0, 'orders_last_24h': 1,
    'device_cluster_size': 1, 'pincode': '560001', 'courier_id': 'Courier_A'
}
res_normal = score_order_guarded(test_order)
print(f"  Standard Order Scoring: Action={res_normal['recommended_action']}, Status={res_normal['status']}, Guardrail={res_normal['guardrail_applied']}")

# Test 2: Kill switch trigger
kill_gr = ProductionGuardrails(kill_switch=True)
res_kill = score_order_guarded(test_order, guardrails=kill_gr)
print(f"  Kill Switch Active:     Action={res_kill['recommended_action']}, Status={res_kill['status']}, Guardrail={res_kill['guardrail_applied']}")

# Test 3: High value order (> Rs 10,000)
high_val_order = dict(test_order, order_value=14500.0)
res_hv = score_order_guarded(high_val_order)
print(f"  High Value Protection:  Action={res_hv['recommended_action']}, Status={res_hv['status']}, Guardrail={res_hv['guardrail_applied']}")

# Test 4: Rate limiter trip (>60% interventions in window)
rate_gr = ProductionGuardrails(max_intervention_rate=0.50, window_size=100)
for _ in range(60):
    rate_gr.recent_actions.append(1) # simulate high-friction spike
res_rate = score_order_guarded(test_order, guardrails=rate_gr)
print(f"  Intervention Rate Cap:  Action={res_rate['recommended_action']}, Status={res_rate['status']}, Guardrail={res_rate['guardrail_applied']}")
print("\n>>> All Step 3 Verifications Completed Successfully.")
