"""
audit/test_guardrails.py — Tests for production guardrails module.
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
    gr = ProductionGuardrails(kill_switch=False, min_base_rto_rate=0.168)
    # Test brand with healthy 14% COD RTO rate (below 16.8% break-even)
    res = gr.evaluate({'order_value': 1200.0, 'merchant_cod_rto_rate': 0.14}, dummy_model)
    assert res['recommended_action'] == 'ALLOW_COD'
    assert 'MERCHANT_BASE_RTO_BELOW_BREAKEVEN' in res['guardrail_applied']
    
    # Test brand with risky 28% COD RTO rate (above break-even) -> model decision preserved
    res_risky = gr.evaluate({'order_value': 1200.0, 'merchant_cod_rto_rate': 0.28}, dummy_model)
    assert res_risky['recommended_action'] == 'REQUIRE_DEPOSIT'
    assert res_risky['guardrail_applied'] == 'NONE'

if __name__ == "__main__":
    test_kill_switch()
    test_high_value_override()
    test_fail_safe_fallback()
    test_rate_cap()
    test_merchant_breakeven_gate()
    print("All guardrail tests PASSED successfully!")
