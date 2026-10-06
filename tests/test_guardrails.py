"""
tests/test_guardrails.py — Tests for production guardrails module.
"""

import os
import pytest
from src.serve.guardrails import ProductionGuardrails

def dummy_model(features):
    return {'recommended_action': 'REQUIRE_DEPOSIT', 'probability': 0.65}

def dummy_error_model(features):
    raise RuntimeError("Database connection timed out")

def test_kill_switch():
    gr = ProductionGuardrails(kill_switch=True)
    res = gr.evaluate({'order_value': 500.0, 'pincode': '110001'}, dummy_model)
    assert res['recommended_action'] == 'ALLOW_COD'
    assert res['guardrail_applied'] == 'KILL_SWITCH_ACTIVE'

def test_high_value_override():
    gr = ProductionGuardrails(kill_switch=False, high_value_threshold=10000.0)
    res = gr.evaluate({'order_value': 15000.0, 'pincode': '110001'}, dummy_model)
    assert res['recommended_action'] == 'MANUAL_REVIEW'
    assert res['guardrail_applied'] == 'HIGH_VALUE_MANUAL_REVIEW'

def test_fail_safe_fallback():
    gr = ProductionGuardrails(kill_switch=False)
    res = gr.evaluate({'order_value': 500.0, 'pincode': '110001'}, dummy_error_model)
    assert res['recommended_action'] == 'ALLOW_COD'
    assert 'FAIL_SAFE_FALLBACK' in res['guardrail_applied']

def test_rate_cap():
    gr = ProductionGuardrails(kill_switch=False, max_intervention_rate=0.50, window_size=100)
    # Seed 60 interventions
    for _ in range(60):
        gr.recent_actions.append(1)
    res = gr.evaluate({'order_value': 500.0, 'pincode': '110001'}, dummy_model)
    assert res['recommended_action'] == 'ALLOW_COD'
    assert 'INTERVENTION_RATE_CAP_TRIPPED' in res['guardrail_applied']

def test_merchant_breakeven_gate():
    from src.serve.guardrails import calculate_merchant_breakeven
    
    # 1. Default synthetic parameters test (RTO=150, margin=20%, basket=826.89, red=0.80, drop=0.40)
    # p* = (0 + 0.40 * 165.378) / (0.80 * 150 + 0.40 * 165.378) = 66.15 / 186.15 ~ 0.355 on basket,
    # and at margin=0.10: p* = (0.40 * 82.69) / (120 + 33.08) ~ 0.216
    default_be = calculate_merchant_breakeven()
    assert 0.15 <= default_be <= 0.40

    # 2. High-margin merchant (margin 40%, basket 2000, RTO cost 100):
    # Turning away good buyers is very costly, so break-even RTO is higher
    high_margin_be = calculate_merchant_breakeven(rto_cost=100.0, margin_pct=0.40, typical_order_value=2000.0)
    assert high_margin_be > default_be, "High margin merchant must have higher break-even RTO requirement"

    # 3. Low-margin merchant (margin 5%, basket 500, RTO cost 300):
    # RTO hurts badly and margin lost on dropped buyers is minimal, so break-even RTO is much lower
    low_margin_be = calculate_merchant_breakeven(rto_cost=300.0, margin_pct=0.05, typical_order_value=500.0)
    assert low_margin_be < default_be, "Low margin merchant must have lower break-even RTO requirement"

    # 4. Gating based on merchant dynamic parameters in features
    gr = ProductionGuardrails(kill_switch=False)
    # Merchant with 12% RTO rate and high margin (break-even ~0.50) -> blocked
    payload_high_margin = {
        'order_value': 2000.0,
        'merchant_cod_rto_rate': 0.12,
        'merchant_rto_cost': 100.0,
        'merchant_margin_pct': 0.40,
        'merchant_typical_order_value': 2000.0
    }
    res = gr.evaluate(payload_high_margin, dummy_model)
    assert res['recommended_action'] == 'ALLOW_COD'
    assert 'MERCHANT_BASE_RTO_BELOW_BREAKEVEN' in res['guardrail_applied']

    # Merchant with 12% RTO rate and low margin (break-even ~0.04) -> model decision preserved
    payload_low_margin = {
        'order_value': 500.0,
        'merchant_cod_rto_rate': 0.12,
        'merchant_rto_cost': 300.0,
        'merchant_margin_pct': 0.05,
        'merchant_typical_order_value': 500.0
    }
    res_low = gr.evaluate(payload_low_margin, dummy_model)
    assert res_low['recommended_action'] == 'REQUIRE_DEPOSIT'
    assert res_low['guardrail_applied'] == 'NONE'


def test_merchant_break_even_simulation_gate():
    """
    Test guardrails.merchant_break_even (Task 5b):
    1. Low/near-crossover merchant (17% COD RTO): lower 95% CI <= safety margin -> BLOCKED.
    2. High-RTO merchant (30% COD RTO): lower 95% CI > safety margin -> ALLOWED.
    3. Fewer than 1,000 orders: Insufficient data -> fall back to ALLOW with logged reason.
    """
    from src.serve.guardrails import merchant_break_even

    costs = {
        'rto_cost': 150.0,
        'margin_pct': 0.20,
        'safety_margin_per_order': 0.25
    }
    # Standard basket distribution: 1,500 orders around Rs 800
    basket_1500 = [800.0] * 1500

    # 1. Merchant at 17% COD RTO: Must be BLOCKED
    res_17 = merchant_break_even(costs, basket_1500, historical_cod_rto_rate=0.17)
    assert not res_17['allowed']
    assert res_17['status'] == 'BLOCKED'
    assert res_17['action'] == 'ALLOW_COD'
    assert res_17['lower_95_ci_savings'] <= 0.25 * 1500

    # 2. High-RTO merchant (30% COD RTO): Must be ALLOWED
    res_30 = merchant_break_even(costs, basket_1500, historical_cod_rto_rate=0.30)
    assert res_30['allowed']
    assert res_30['status'] == 'ALLOWED'
    assert res_30['action'] == 'POLICY_ENABLED'
    assert res_30['expected_net_savings'] > 0
    assert res_30['lower_95_ci_savings'] > 0

    # 3. Fewer than 1,000 orders (e.g. 500 orders): Insufficient data -> fall back to ALLOW with logged reason
    basket_500 = [800.0] * 500
    res_small = merchant_break_even(costs, basket_500, historical_cod_rto_rate=0.30)
    assert not res_small['allowed']
    assert res_small['status'] == 'INSUFFICIENT_DATA'
    assert res_small['action'] == 'ALLOW_COD'
    assert "Insufficient order data" in res_small['reason']
    assert "Falling back to ALLOW_COD" in res_small['reason']


def test_ranked_cap_drops_lowest_benefit_orders():
    """Verify ranked cap drops lowest-benefit orders and preserves highest-benefit orders (Task 2 & Mutation 20 & 22)."""
    gr = ProductionGuardrails(kill_switch=False, max_intervention_rate=0.60, window_size=100, cap_mode='ranked')

    # Seed 60 orders with varying benefits
    for b in range(1, 61):
        gr.recent_benefits.append(float(b))
        gr.recent_actions.append(1)

    # Low benefit order (benefit = 5 < cutoff ~24.6): must be dropped to ALLOW_COD
    low_order = {
        'order_value': 800.0,
        'pincode': '110001'
    }
    low_model = lambda f: {
        'recommended_action': 'VERIFY_ADDRESS',
        'probability': 0.25,
        'el_table': {'ALLOW_COD': 50.0, 'VERIFY_ADDRESS': 45.0}  # benefit = 5.0
    }
    res_low = gr.evaluate(low_order, low_model)
    assert res_low['recommended_action'] == 'ALLOW_COD'
    assert 'INTERVENTION_CAP_RANKED' in res_low['guardrail_applied']

    # Mid benefit order (benefit = 30.0): above 40th percentile (~24.6), so kept under 60% cap
    # (Catches Mutation 22 which calculates quantile tail inverted at 60th percentile ~36.4)
    mid_order = {
        'order_value': 800.0,
        'pincode': '110001'
    }
    mid_model = lambda f: {
        'recommended_action': 'REQUIRE_DEPOSIT',
        'probability': 0.60,
        'el_table': {'ALLOW_COD': 100.0, 'REQUIRE_DEPOSIT': 70.0}  # benefit = 30.0
    }
    res_mid = gr.evaluate(mid_order, mid_model)
    assert res_mid['recommended_action'] == 'REQUIRE_DEPOSIT'
    assert res_mid['guardrail_applied'] == 'NONE'

    # High benefit order (benefit = 50 > cutoff): must be kept
    high_order = {
        'order_value': 800.0,
        'pincode': '110001'
    }
    high_model = lambda f: {
        'recommended_action': 'REQUIRE_DEPOSIT',
        'probability': 0.70,
        'el_table': {'ALLOW_COD': 120.0, 'REQUIRE_DEPOSIT': 70.0}  # benefit = 50.0
    }
    res_high = gr.evaluate(high_order, high_model)
    assert res_high['recommended_action'] == 'REQUIRE_DEPOSIT'
    assert res_high['guardrail_applied'] == 'NONE'


def test_merchant_break_even_requires_lower_bound_not_point_estimate():
    """Verify that merchant break-even gate rejects positive point estimates whose lower 95% CI is below margin (Mutation 23)."""
    from src.serve.guardrails import merchant_break_even
    costs = {
        'rto_cost': 150.0,
        'margin_pct': 0.20,
        'safety_margin_per_order': 0.25
    }
    basket_1500 = [800.0] * 1500
    # At 20.8% COD RTO, point estimate is well positive (+Rs 0.46/order > Rs 0.25 margin), but lower 95% CI is only +Rs 0.12/order (<= Rs 0.25)
    res = merchant_break_even(costs, basket_1500, historical_cod_rto_rate=0.208)
    assert res['expected_net_savings'] > 0.25 * 1500, "Point estimate must be strictly above safety margin"
    assert res['lower_95_ci_savings'] <= 0.25 * 1500, "Lower bound must fail safety margin"
    assert not res['allowed'], "Guardrail must block friction when lower 95% CI fails safety threshold"
    assert res['status'] == 'BLOCKED'
    assert res['action'] == 'ALLOW_COD'




