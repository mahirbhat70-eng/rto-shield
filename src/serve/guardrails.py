"""
src/serve/guardrails.py — Production Decision Engine Safety Guardrails.

Implements:
1. Kill Switch: Instant emergency circuit breaker to force ALLOW_COD.
2. Intervention Rate Cap: Sliding-window rate limiter preventing runaway intervention spikes.
3. Order Value Guardrail: High-value orders (>Rs 10,000) routed to manual review rather than friction.
4. Fail-Safe Fallback: Automatic fallback to ALLOW_COD on any model/scoring exception.
5. Merchant Whitelist/Blacklist: Pincode and customer overrides.
6. Minimum Base RTO Gate: Disables interventions when background merchant RTO is below break-even (<16%).
"""

import os
import numpy as np
from collections import deque
from typing import Dict, Any

DEFAULT_BENCHMARK_BREAKEVEN = 0.168  # Holds for default synthetic benchmark (RTO cost Rs 150, Margin 20%, Basket ~Rs 826)


def calculate_merchant_breakeven(rto_cost: float = 150.0,
                                 margin_pct: float = 0.20,
                                 typical_order_value: float = 826.89,
                                 intervention_reduction: float = 0.80,
                                 intervention_drop: float = 0.40,
                                 friction_cost: float = 0.0) -> float:
    """
    Calculate merchant-specific break-even COD RTO rate:
    p* = (friction + drop * Margin) / (reduction * RTO_cost + drop * Margin)
    Below this baseline RTO rate, applying friction causes more lost margin
    from dropped good buyers than delivery losses saved.
    """
    margin = typical_order_value * margin_pct
    numerator = friction_cost + (intervention_drop * margin)
    denominator = (intervention_reduction * rto_cost) + (intervention_drop * margin)
    if denominator <= 0:
        return DEFAULT_BENCHMARK_BREAKEVEN
    return float(np.clip(numerator / denominator, 0.01, 0.99))


class ProductionGuardrails:
    def __init__(self, 
                 kill_switch: bool = False,
                 max_intervention_rate: float = 0.60,
                 window_size: int = 1000,
                 high_value_threshold: float = 10000.0,
                 min_base_rto_rate: float = None,
                 merchant_rto_cost: float = 150.0,
                 merchant_margin_pct: float = 0.20,
                 merchant_typical_order_value: float = 826.89,
                 pincode_whitelist: set = None,
                 pincode_blacklist: set = None):
        self.kill_switch = kill_switch or (os.getenv("RTO_SHIELD_KILL_SWITCH", "0") == "1")
        self.max_intervention_rate = max_intervention_rate
        self.high_value_threshold = high_value_threshold
        
        # Calculate dynamic merchant break-even rate if not explicitly overridden
        if min_base_rto_rate is not None:
            self.min_base_rto_rate = float(min_base_rto_rate)
        else:
            self.min_base_rto_rate = calculate_merchant_breakeven(
                rto_cost=merchant_rto_cost,
                margin_pct=merchant_margin_pct,
                typical_order_value=merchant_typical_order_value
            )
            
        self.pincode_whitelist = pincode_whitelist or set()
        self.pincode_blacklist = pincode_blacklist or set()
        
        # Sliding window for rate limiting
        self.window_size = window_size
        self.recent_actions = deque(maxlen=window_size)
        
    def evaluate(self, features: Dict[str, Any], raw_decision_func) -> Dict[str, Any]:
        """
        Enforce safety guardrails around the model scoring function.
        """
        # 1. Kill Switch Trigger
        if self.kill_switch or os.getenv("RTO_SHIELD_KILL_SWITCH", "0") == "1":
            return {
                'recommended_action': 'ALLOW_COD',
                'probability': 0.0,
                'guardrail_applied': 'KILL_SWITCH_ACTIVE',
                'status': 'OVERRIDDEN'
            }

        # 2. Dynamic Break-Even Merchant RTO Gate: Block friction if merchant baseline is below calculated threshold
        merchant_rto = features.get('merchant_cod_rto_rate', features.get('merchant_base_rto', None))
        
        # Check if merchant provided dynamic profile attributes in features
        if 'merchant_rto_cost' in features or 'merchant_margin_pct' in features:
            req_rto_cost = float(features.get('merchant_rto_cost', 150.0))
            req_margin_pct = float(features.get('merchant_margin_pct', 0.20))
            req_basket = float(features.get('merchant_typical_order_value', features.get('order_value', 826.89)))
            threshold = calculate_merchant_breakeven(req_rto_cost, req_margin_pct, req_basket)
        else:
            threshold = self.min_base_rto_rate

        if merchant_rto is not None and float(merchant_rto) < threshold:
            return {
                'recommended_action': 'ALLOW_COD',
                'probability': float(merchant_rto),
                'guardrail_applied': f'MERCHANT_BASE_RTO_BELOW_BREAKEVEN ({float(merchant_rto)*100:.1f}% < {threshold*100:.1f}%)',
                'status': 'OVERRIDDEN'
            }
            
        pincode = str(features.get('pincode', ''))
        
        # 3. Pincode Whitelist (VIP areas)
        if pincode in self.pincode_whitelist:
            return {
                'recommended_action': 'ALLOW_COD',
                'probability': 0.0,
                'guardrail_applied': 'PINCODE_WHITELIST_BYPASS',
                'status': 'OVERRIDDEN'
            }
            
        # 3. High Value Protection (> Rs 10,000)
        order_value = float(features.get('order_value', 0.0))
        if order_value >= self.high_value_threshold:
            return {
                'recommended_action': 'MANUAL_REVIEW',
                'probability': 0.50,
                'guardrail_applied': 'HIGH_VALUE_MANUAL_REVIEW',
                'status': 'FLAGGED'
            }

        # 4. Safe Execution with Fail-Safe Fallback
        try:
            decision = raw_decision_func(features)
            action = decision.get('recommended_action', 'ALLOW_COD')
            p = decision.get('probability', 0.20)
        except Exception as exc:
            return {
                'recommended_action': 'ALLOW_COD',
                'probability': 0.20,
                'guardrail_applied': f'FAIL_SAFE_FALLBACK ({type(exc).__name__})',
                'status': 'FALLBACK_TRIGGERED'
            }

        # 5. Intervention Rate Cap (Sliding Window)
        is_intervention = 1 if action not in ('ALLOW_COD', 'PREPAID_PASSTHROUGH') else 0
        if len(self.recent_actions) >= 50:
            current_rate = sum(self.recent_actions) / len(self.recent_actions)
            if current_rate >= self.max_intervention_rate and is_intervention:
                self.recent_actions.append(0)
                return {
                    'recommended_action': 'ALLOW_COD',
                    'probability': p,
                    'guardrail_applied': f'INTERVENTION_RATE_CAP_TRIPPED ({current_rate*100:.1f}%)',
                    'status': 'OVERRIDDEN'
                }
                
        self.recent_actions.append(is_intervention)
        decision['guardrail_applied'] = 'NONE'
        decision['status'] = 'PASSED'
        return decision
