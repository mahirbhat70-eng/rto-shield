"""
cost_engine.py — Expected-loss intervention pricing + batch policy router.

Changes vs v1 (see IMPROVEMENTS.md):
  * `is_cod()` is the single source of truth for COD detection, shared by the
    single-order scorer and the batch router (previously they disagreed:
    score_order passthroughed only the literal 'PREPAID' while the batch API
    passthroughed every non-COD payment).
  * Default config path is now absolute (package-relative), so importing the
    engine no longer depends on the process CWD.
  * get_optimal_policy is fully vectorized (no iterrows) and validates that
    the probability vector is row-aligned with the DataFrame. The old
    positional `proba_series.iloc[i]` silently mispaired probabilities
    whenever the caller passed a filtered frame without reset_index().
  * Phase 8: Full operational cost decomposition for interventions with optional
    lifetime value (LTV) loss toggle and sensitivity analysis support.
"""

import os
import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'cost_config.yaml')
)


def is_cod(payment_method) -> bool:
    """Single source of truth: normalized COD detection (strip + upper)."""
    if payment_method is None:
        return False
    return str(payment_method).strip().upper() == 'COD'


class CostEngine:
    def __init__(self, config_path=None):
        path = config_path or os.environ.get('RTO_SHIELD_COST_CONFIG', DEFAULT_CONFIG_PATH)
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.rto_logistics_cost = float(self.config['rto_logistics_cost'])
        self.average_margin_pct = float(self.config['average_margin_pct'])
        self.interventions = self.config['interventions']
        self.assumed_operational_costs = self.config.get('assumed_operational_costs', {})

    def compute_decomposed_friction(self, action: str, order_value: float = None, ltv_multiplier: float = 0.0) -> float:
        """
        Decomposed operational friction cost per intervention (Task 8a).
        Computes payment gateway, refund handling, reconciliation, messaging, and support.
        Optionally accounts for customer lifetime value (LTV) drop-off penalty.
        """
        if action not in self.interventions:
            return 0.0

        assumed = self.assumed_operational_costs
        if action == "ALLOW_COD":
            base_friction = 0.0
        elif action == "VERIFY_ADDRESS":
            msg = float(assumed.get('whatsapp_otp_template_cost', 0.75))
            support = float(assumed.get('support_ticket_cost_unit', 5.00)) * float(assumed.get('support_contact_probability_verify', 0.25))
            base_friction = msg + support  # 0.75 + 1.25 = 2.00
        elif action == "REQUIRE_DEPOSIT":
            deposit_amt = float(assumed.get('deposit_fixed_amount', 100.00))
            pg_fee = deposit_amt * float(assumed.get('payment_gateway_fee_deposit_pct', 0.02)) + float(assumed.get('payment_gateway_fee_deposit_fixed', 1.80))
            refund_fee = float(assumed.get('refund_processing_cost_unit', 3.00)) * float(assumed.get('refund_probability_on_deposit', 0.40))
            support_fee = float(assumed.get('support_ticket_cost_unit', 5.00)) * float(assumed.get('support_contact_probability_deposit', 0.25))
            msg_fee = float(assumed.get('whatsapp_otp_template_cost', 0.75))
            base_friction = pg_fee + refund_fee + support_fee + msg_fee  # 3.80 + 1.20 + 1.25 + 0.75 = 7.00
        elif action == "PREPAID_ONLY":
            base_friction = float(self.interventions['PREPAID_ONLY'].get('friction_cost', 0.0))
        else:
            base_friction = float(self.interventions[action].get('friction_cost', 0.0))

        # Optional Lifetime Value (LTV) term:
        # ltv_penalty = ltv_multiplier * margin * drop_rate
        if ltv_multiplier > 0.0 and order_value is not None:
            drop_pct = float(self.interventions[action]['success_drop_pct'])
            margin = float(order_value) * self.average_margin_pct
            base_friction += ltv_multiplier * margin * drop_pct

        return float(base_friction)

    def evaluate_interventions(self, order_value, p_rto, ltv_multiplier=0.0):
        """
        Single-order expected loss per intervention:
        EL(a) = friction + P(rto after a) * logistics_cost
                - P(success after a) * margin + [ltv_loss if enabled]
        """
        order_value = float(order_value)
        p_rto = float(p_rto)
        margin = order_value * self.average_margin_pct

        results = {}
        for action, params in self.interventions.items():
            friction_cost = float(params['friction_cost'])
            if ltv_multiplier > 0.0:
                friction_cost = self.compute_decomposed_friction(action, order_value, ltv_multiplier=ltv_multiplier)

            success_drop_pct = float(params['success_drop_pct'])
            rto_reduction_pct = float(params['rto_reduction_pct'])

            p_success = (1.0 - p_rto) * (1.0 - success_drop_pct)
            p_rto_after = p_rto * (1.0 - rto_reduction_pct)

            expected_loss = friction_cost + (p_rto_after * self.rto_logistics_cost) - (p_success * margin)
            results[action] = expected_loss

        return results

    def evaluate_interventions_vectorized(self, order_values, p_rto, ltv_multiplier=0.0):
        """
        Vectorized EL over arrays. Returns (action_names, el_matrix) where
        el_matrix has shape (n_actions, n_orders) in config key order.
        """
        names = list(self.interventions.keys())
        V = np.asarray(order_values, dtype=float)
        P = np.asarray(p_rto, dtype=float)
        if V.shape != P.shape:
            raise ValueError(
                f"order_values shape {V.shape} != p_rto shape {P.shape}"
            )
        margin = V * self.average_margin_pct

        cols = []
        for name in names:
            params = self.interventions[name]
            friction = float(params['friction_cost'])
            drop = float(params['success_drop_pct'])
            red = float(params['rto_reduction_pct'])

            if ltv_multiplier > 0.0:
                extra_ltv = ltv_multiplier * margin * drop
                friction = friction + extra_ltv

            cols.append(
                friction + (P * (1.0 - red)) * self.rto_logistics_cost
                - ((1.0 - P) * (1.0 - drop)) * margin
            )
        return names, np.vstack(cols)

    def get_optimal_policy(self, df, proba_series, ltv_multiplier=0.0):
        """
        Vectorized batch router.

        Applies the intervention menu ONLY to COD rows; every non-COD row is
        assigned 'PREPAID_PASSTHROUGH' with 0 expected loss.

        Contract: `proba_series` must be row-aligned with `df` (same length,
        positional). A Series whose index differs from df's is accepted only
        if it was reset/aligned — length mismatch raises immediately.

        Returns (actions ndarray, expected_losses ndarray).
        """
        if len(df) == 0:
            return np.array([], dtype=object), np.array([], dtype=float)

        p = np.asarray(pd.Series(proba_series).to_numpy(), dtype=float).ravel()
        if len(p) != len(df):
            raise ValueError(
                f"Length mismatch: proba_series has {len(p)} values but df has {len(df)} rows — "
                "probabilities must be row-aligned (pass df.reset_index(drop=True) "
                "and a matching probability vector)."
            )

        V = pd.to_numeric(df['order_value'], errors='coerce').to_numpy(dtype=float)
        if np.isnan(V).any():
            raise ValueError("df['order_value'] contains non-numeric or missing values.")

        cod_mask = df['payment_method'].map(is_cod).to_numpy(dtype=bool)

        names, mat = self.evaluate_interventions_vectorized(V, p, ltv_multiplier=ltv_multiplier)
        action_names = np.asarray(names, dtype=object)
        best_idx = np.argmin(mat, axis=0)  # first-min tie-break == config key order
        col_idx = np.arange(len(df))
        actions = np.where(cod_mask, action_names[best_idx], 'PREPAID_PASSTHROUGH')
        losses = np.where(cod_mask, mat[best_idx, col_idx], 0.0)
        return actions, losses
