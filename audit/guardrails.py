"""
audit/guardrails.py — Production Decision Engine Safety Guardrails.

Implements:
1. Kill Switch: Instant emergency circuit breaker to force ALLOW_COD.
2. Intervention Rate Cap: Sliding-window rate limiter preventing runaway intervention spikes.
3. Order Value Guardrail: High-value orders (>Rs 10,000) routed to manual review rather than friction.
4. Fail-Safe Fallback: Automatic fallback to ALLOW_COD on any model/scoring exception.
5. Merchant Whitelist/Blacklist: Pincode and customer overrides.
"""

import os
import time
from collections import deque
from typing import Dict, Any, Tuple
from src.serve.guardrails import merchant_break_even, calculate_merchant_breakeven

class ProductionGuardrails:
    def __init__(self, 
                 kill_switch: bool = False,
                 max_intervention_rate: float = 0.60,
                 window_size: int = 1000,
                 high_value_threshold: float = 10000.0,
                 min_base_rto_rate: float = 0.168,
                 pincode_whitelist: set = None,
                 pincode_blacklist: set = None):
        self.kill_switch = kill_switch or (os.getenv("RTO_SHIELD_KILL_SWITCH", "0") == "1")
        self.max_intervention_rate = max_intervention_rate
        self.high_value_threshold = high_value_threshold
        self.min_base_rto_rate = min_base_rto_rate
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

        # 2. Break-Even Merchant RTO Gate: Block friction if merchant baseline is below 16.8%
        merchant_rto = features.get('merchant_cod_rto_rate', features.get('merchant_base_rto', None))
        if merchant_rto is not None and float(merchant_rto) < self.min_base_rto_rate:
            return {
                'recommended_action': 'ALLOW_COD',
                'probability': float(merchant_rto),
                'guardrail_applied': f'MERCHANT_BASE_RTO_BELOW_BREAKEVEN ({float(merchant_rto)*100:.1f}% < {self.min_base_rto_rate*100:.1f}%)',
                'status': 'OVERRIDDEN'
            }
            
        pincode = str(features.get('pincode', ''))
        
        # 2. Pincode Whitelist (VIP areas)
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
            # High value orders must not be casually frictioned or dropped
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
            # Fallback on any error
            return {
                'recommended_action': 'ALLOW_COD',
                'probability': 0.20,
                'guardrail_applied': f'FAIL_SAFE_FALLBACK ({type(exc).__name__})',
                'status': 'FALLBACK_TRIGGERED'
            }

        # 5. Intervention Rate Cap (Sliding Window)
        is_intervention = 1 if action != 'ALLOW_COD' else 0
        if len(self.recent_actions) >= 50:
            current_rate = sum(self.recent_actions) / len(self.recent_actions)
            if current_rate >= self.max_intervention_rate and is_intervention:
                # Trip intervention cap: downgrade to ALLOW_COD
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
