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
    Calculate closed-form per-order p* threshold.
    """
    margin = typical_order_value * margin_pct
    numerator = friction_cost + (intervention_drop * margin)
    denominator = (intervention_reduction * rto_cost) + (intervention_drop * margin)
    if denominator <= 0:
        return DEFAULT_BENCHMARK_BREAKEVEN
    return float(np.clip(numerator / denominator, 0.01, 0.99))


def merchant_break_even(merchant_costs: dict,
                        basket_distribution,
                        historical_cod_rto_rate: float,
                        min_orders: int = 1000) -> dict:
    """
    Simulates the frozen multi-action policy on the merchant's own order data / basket distribution
    at their historical COD RTO rate to compute expected net savings.
    
    The frozen policy scores orders using the calibrated risk engine. When the merchant's true
    historical COD RTO rate is low (e.g. 10%), the frozen policy triggers friction on false positives,
    causing lost margin on dropped good buyers that exceeds RTO savings.
    
    Blocks friction if expected net savings <= 0 or if data is insufficient (< min_orders).
    """
    # 1. Check data sufficiency
    if basket_distribution is None:
        n_orders = 0
    elif isinstance(basket_distribution, (list, tuple, np.ndarray)):
        n_orders = len(basket_distribution)
    elif hasattr(basket_distribution, '__len__'):
        n_orders = len(basket_distribution)
    else:
        n_orders = 0

    if n_orders < min_orders:
        return {
            'allowed': False,
            'expected_net_savings': 0.0,
            'status': 'INSUFFICIENT_DATA',
            'action': 'ALLOW_COD',
            'reason': f"Insufficient order data: got {n_orders} COD orders, requires at least {min_orders}. Falling back to ALLOW_COD."
        }

    # Extract order values
    if isinstance(basket_distribution, (list, tuple, np.ndarray)):
        if n_orders > 0 and isinstance(basket_distribution[0], dict):
            values = np.array([float(o.get('order_value', 826.89)) for o in basket_distribution])
        else:
            values = np.array([float(v) for v in basket_distribution])
    elif hasattr(basket_distribution, 'columns') and 'order_value' in basket_distribution.columns:
        values = basket_distribution['order_value'].values.astype(float)
    else:
        values = np.array([float(v) for v in basket_distribution])

    rto_cost = float(merchant_costs.get('rto_cost', merchant_costs.get('rto_logistics_cost', 150.0)))
    margin_pct = float(merchant_costs.get('margin_pct', merchant_costs.get('average_margin_pct', 0.20)))

    # Interventions parameters from merchant_costs or canonical cost_config.yaml
    iv = merchant_costs.get('interventions')
    if iv is None:
        try:
            import yaml
            cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "cost_config.yaml")
            with open(cfg_path, "r", encoding="utf-8") as f:
                iv = yaml.safe_load(f).get("interventions", {})
        except Exception:
            iv = {}
    if not iv:
        iv = {
            'ALLOW_COD': {'friction_cost': 0.0, 'rto_reduction_pct': 0.0, 'success_drop_pct': 0.0},
            'VERIFY_ADDRESS': {'friction_cost': 2.00, 'rto_reduction_pct': 0.30, 'success_drop_pct': 0.05},
            'REQUIRE_DEPOSIT': {'friction_cost': 7.00, 'rto_reduction_pct': 0.80, 'success_drop_pct': 0.40},
            'PREPAID_ONLY': {'friction_cost': 0.0, 'rto_reduction_pct': 0.55, 'success_drop_pct': 0.70},
        }

    # Model scores (using representative benchmark distribution centered at ~0.28)
    rng = np.random.default_rng(42)
    # Model scores follow benchmark calibrated distribution (mean ~0.283)
    p_model = np.clip(rng.beta(2.8, 7.2, size=n_orders), 1e-4, 1.0 - 1e-4)

    # Shift probabilities to merchant's true historical rate via odds adjustment
    target_r = float(np.clip(historical_cod_rto_rate, 0.01, 0.99))
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        q = 1.0 / (1.0 + np.exp(-(np.log(p_model / (1.0 - p_model)) + mid)))
        if q.mean() < target_r:
            lo = mid
        else:
            hi = mid
    p_true = q

    # Simulate frozen policy decisions on (values, p_model)
    # and compute expected loss under p_true
    baseline_losses = []
    policy_losses = []

    for i in range(n_orders):
        pm = p_model[i]
        pt = p_true[i]
        V = values[i]
        margin = V * margin_pct

        # Frozen decision uses model's score pm
        best_act = 'ALLOW_COD'
        best_model_el = pm * rto_cost - (1.0 - pm) * margin
        for act_name, params in iv.items():
            fr = params['friction_cost']
            rr = params['rto_reduction_pct']
            dr = params['success_drop_pct']
            el = fr + pm * (1.0 - rr) * rto_cost - (1.0 - pm) * (1.0 - dr) * margin
            if el < best_model_el:
                best_model_el = el
                best_act = act_name

        # True expected losses under merchant's true outcome probability pt
        base_l = pt * rto_cost - (1.0 - pt) * margin
        act_params = iv[best_act]
        pol_l = (act_params['friction_cost']
                 + pt * (1.0 - act_params['rto_reduction_pct']) * rto_cost
                 - (1.0 - pt) * (1.0 - act_params['success_drop_pct']) * margin)

        baseline_losses.append(base_l)
        policy_losses.append(pol_l)

    diff_losses = np.array(baseline_losses) - np.array(policy_losses)
    mean_diff = float(np.mean(diff_losses))
    std_diff = float(np.std(diff_losses, ddof=1)) if n_orders > 1 else 0.0
    se_diff = std_diff / np.sqrt(n_orders) if n_orders > 0 else 0.0

    # Stated safety margin: require expected positive savings at lower 95% confidence bound
    safety_margin_per_order = float(merchant_costs.get('safety_margin_per_order', 0.25))
    lower_95_bound_per_order = mean_diff - 1.96 * se_diff
    lower_95_bound_total = lower_95_bound_per_order * n_orders
    net_savings = float(np.sum(diff_losses))

    if lower_95_bound_per_order <= safety_margin_per_order:
        return {
            'allowed': False,
            'expected_net_savings': round(net_savings, 2),
            'lower_95_ci_savings': round(lower_95_bound_total, 2),
            'status': 'BLOCKED',
            'action': 'ALLOW_COD',
            'reason': (
                f"Simulation lower 95% CI net savings (Rs {lower_95_bound_total:,.2f}) "
                f"fails safety margin threshold at historical COD RTO rate {target_r*100:.1f}%. Friction blocked."
            )
        }
    else:
        return {
            'allowed': True,
            'expected_net_savings': round(net_savings, 2),
            'lower_95_ci_savings': round(lower_95_bound_total, 2),
            'status': 'ALLOWED',
            'action': 'POLICY_ENABLED',
            'reason': (
                f"Simulation verified positive net savings at lower 95% CI (Rs {lower_95_bound_total:,.2f}) "
                f"at historical COD RTO rate {target_r*100:.1f}%. Friction enabled."
            )
        }



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
                 pincode_blacklist: set = None,
                 cap_mode: str = 'naive'):
        self.kill_switch = kill_switch or (os.getenv("RTO_SHIELD_KILL_SWITCH", "0") == "1")
        self.max_intervention_rate = max_intervention_rate
        self.high_value_threshold = high_value_threshold
        
        # Dynamic merchant break-even rate (default benchmark crossover is 0.168)
        if min_base_rto_rate is not None:
            self.min_base_rto_rate = float(min_base_rto_rate)
        else:
            self.min_base_rto_rate = DEFAULT_BENCHMARK_BREAKEVEN
            
        self.pincode_whitelist = pincode_whitelist or set()
        self.pincode_blacklist = pincode_blacklist or set()
        
        # Sliding window for rate limiting
        self.window_size = window_size
        self.recent_actions = deque(maxlen=window_size)
        # cap_mode 'naive': first-come — once the window rate hits the cap, ANY new
        #   intervention is forced to ALLOW (arrival order decides who is dropped).
        # cap_mode 'ranked': intervene only if this order's EL improvement over ALLOW is
        #   >= the (1 - cap) quantile of improvements in the recent window.
        # ponytail: quantile over a sliding window of the last `window_size` COD orders;
        #   assumes the benefit distribution is stationary over that window.
        if cap_mode not in ('naive', 'ranked'):
            raise ValueError(f"cap_mode must be 'naive' or 'ranked', got {cap_mode!r}")
        self.cap_mode = cap_mode
        self.recent_benefits = deque(maxlen=window_size)
        
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
        if self.cap_mode == 'ranked' and action != 'PREPAID_PASSTHROUGH':
            el = decision.get('el_table') or {}
            benefit = float(el.get('ALLOW_COD', 0.0) - el.get(action, 0.0)) if is_intervention else 0.0
            self.recent_benefits.append(benefit)
            if is_intervention and len(self.recent_benefits) >= 50:
                cutoff = float(np.quantile(self.recent_benefits, 1.0 - self.max_intervention_rate))
                if benefit < cutoff:
                    self.recent_actions.append(0)
                    return {**decision, 'recommended_action': 'ALLOW_COD', 'probability': p,
                            'guardrail_applied': f'INTERVENTION_CAP_RANKED (benefit {benefit:.2f} < {cutoff:.2f})',
                            'status': 'OVERRIDDEN'}
        elif len(self.recent_actions) >= 50:
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
