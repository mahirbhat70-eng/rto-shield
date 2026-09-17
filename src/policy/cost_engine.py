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
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rto_logistics_cost = self.config['rto_logistics_cost']
        self.average_margin_pct = self.config['average_margin_pct']
        self.interventions = self.config['interventions']

    def evaluate_interventions(self, order_value, p_rto):
        """
        Single-order expected loss per intervention:
        EL(a) = friction + P(rto after a) * logistics_cost
                - P(success after a) * margin
        """
        order_value = float(order_value)
        p_rto = float(p_rto)
        margin = order_value * self.average_margin_pct

        results = {}
        for action, params in self.interventions.items():
            friction_cost = params['friction_cost']
            success_drop_pct = params['success_drop_pct']
            rto_reduction_pct = params['rto_reduction_pct']

            p_success = (1.0 - p_rto) * (1.0 - success_drop_pct)
            p_rto_after = p_rto * (1.0 - rto_reduction_pct)

            expected_loss = friction_cost + (p_rto_after * self.rto_logistics_cost) - (p_success * margin)
            results[action] = expected_loss

        return results

    def evaluate_interventions_vectorized(self, order_values, p_rto):
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
            friction = params['friction_cost']
            drop = params['success_drop_pct']
            red = params['rto_reduction_pct']
            cols.append(
                friction + (P * (1.0 - red)) * self.rto_logistics_cost
                - ((1.0 - P) * (1.0 - drop)) * margin
            )
        return names, np.vstack(cols)

    def get_optimal_policy(self, df, proba_series):
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

        names, mat = self.evaluate_interventions_vectorized(V, p)
        action_names = np.asarray(names, dtype=object)
        best_idx = np.argmin(mat, axis=0)  # first-min tie-break == config key order
        col_idx = np.arange(len(df))
        actions = np.where(cod_mask, action_names[best_idx], 'PREPAID_PASSTHROUGH')
        losses = np.where(cod_mask, mat[best_idx, col_idx], 0.0)
        return actions, losses
