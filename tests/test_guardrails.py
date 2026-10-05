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

