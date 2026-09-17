"""
src/policy/cost_engine.py
WHY: The core decision layer. Converts a calibrated P(RTO) probability into
     an expected-loss table across all four intervention actions, then routes
     each order to the argmin action.

     Design choices:
     - is_cod() is the single source of truth for payment classification.
       One centralized function prevents the v1 bug where 'UPI', 'Credit Card',
       etc. were priced through the COD EL table.
     - CostEngine is initialized from a config file (absolute path, env override)
       so it works from any CWD.
     - get_optimal_policy is fully vectorized (no iterrows). The length-match
       validation raises immediately on mismatch — silent index misalignment
       was a real bug in the original implementation.
"""

import os
import yaml
import numpy as np
import pandas as pd


def is_cod(payment_method) -> bool:
    """
    Single source of truth: normalized COD detection.
    Handles None, whitespace, and case variants.
    Returns True ONLY for the string 'COD' (case-insensitive, stripped).
    """
    if payment_method is None:
        return False
    return str(payment_method).strip().upper() == "COD"


def _default_config_path() -> str:
    """Absolute package-relative path to the cost config yaml."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "configs", "cost_config.yaml")


class CostEngine:
    """
    EL(a) = friction(a) + p * (1 - r_a) * C_logistics - (1 - p) * (1 - d_a) * margin * V
    where C_logistics and margin come from the config file.
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to cost_config.yaml. Defaults to the package-relative
                         configs/ directory. Can be overridden with the env var
                         RTO_SHIELD_COST_CONFIG.
        """
        if config_path is None:
            config_path = os.environ.get(
                "RTO_SHIELD_COST_CONFIG", _default_config_path()
            )
        config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.rto_logistics_cost: float = float(cfg["rto_logistics_cost"])
        self.average_margin_pct: float = float(cfg["average_margin_pct"])
        self.interventions: dict = cfg["interventions"]
        # Cache ordered action list for vectorized ops
        self._action_names: list = list(self.interventions.keys())

    # ------------------------------------------------------------------
    # Single-order API
    # ------------------------------------------------------------------

    def evaluate_interventions(self, order_value: float, p_rto: float) -> dict:
        """
        Compute EL for every action for one order.

        Returns:
            dict mapping action name -> expected loss (float).
            Lower EL = preferred action.
        """
        order_value = float(order_value)
        p_rto = float(p_rto)
        margin = order_value * self.average_margin_pct

        results = {}
        for action, params in self.interventions.items():
            friction = float(params["friction_cost"])
            d = float(params["success_drop_pct"])
            r = float(params["rto_reduction_pct"])

            p_rto_after = p_rto * (1.0 - r)
            p_success = (1.0 - p_rto) * (1.0 - d)
            el = friction + p_rto_after * self.rto_logistics_cost - p_success * margin
            results[action] = float(el)

        return results

    # ------------------------------------------------------------------
    # Vectorized API
    # ------------------------------------------------------------------

    def evaluate_interventions_vectorized(
        self, order_values: np.ndarray, p_rto: np.ndarray
    ):
        """
        Vectorized EL computation for all actions.

        Args:
            order_values: array shape (n,)
            p_rto:        array shape (n,)

        Returns:
            (action_names: list[str], el_matrix: np.ndarray shape (n_actions, n))
        """
        order_values = np.asarray(order_values, dtype=float)
        p_rto = np.asarray(p_rto, dtype=float)
        n = len(order_values)
        n_actions = len(self._action_names)
        el_matrix = np.empty((n_actions, n), dtype=float)

        margins = order_values * self.average_margin_pct

        for i, action in enumerate(self._action_names):
            params = self.interventions[action]
            friction = float(params["friction_cost"])
            d = float(params["success_drop_pct"])
            r = float(params["rto_reduction_pct"])

            p_rto_after = p_rto * (1.0 - r)
            p_success = (1.0 - p_rto) * (1.0 - d)
            el_matrix[i] = friction + p_rto_after * self.rto_logistics_cost - p_success * margins

        return self._action_names, el_matrix

    # ------------------------------------------------------------------
    # Batch router (main production path)
    # ------------------------------------------------------------------

    def get_optimal_policy(
        self, df: pd.DataFrame, proba_series: pd.Series
    ):
        """
        Route each order to the argmin EL action.

        Contract:
            - len(proba_series) == len(df): enforced, raises on mismatch.
            - Non-COD rows → ('PREPAID_PASSTHROUGH', 0.0) immediately.
            - COD rows → vectorized EL evaluation → argmin action.
            - No iterrows, no positional iloc[i] pairing.

        Args:
            df:           DataFrame with at least columns ['payment_method', 'order_value'].
            proba_series: Aligned probability series (same index as df).

        Returns:
            (actions: np.ndarray[str], losses: np.ndarray[float])
        """
        if len(proba_series) != len(df):
            raise ValueError(
                f"Length mismatch: df has {len(df)} rows but proba_series has "
                f"{len(proba_series)} elements. Alignment contract violated."
            )

        actions = np.full(len(df), "PREPAID_PASSTHROUGH", dtype=object)
        losses = np.zeros(len(df), dtype=float)

        # Build aligned numpy arrays (reset-index safe)
        pm_array = df["payment_method"].to_numpy()
        ov_array = df["order_value"].to_numpy(dtype=float)
        prob_array = np.asarray(proba_series, dtype=float)

        cod_mask = np.array([is_cod(pm) for pm in pm_array])

        if cod_mask.any():
            cod_ov = ov_array[cod_mask]
            cod_prob = prob_array[cod_mask]
            _, el_matrix = self.evaluate_interventions_vectorized(cod_ov, cod_prob)
            best_indices = np.argmin(el_matrix, axis=0)
            best_actions = np.array(self._action_names)[best_indices]
            best_losses = el_matrix[best_indices, np.arange(len(cod_ov))]

            actions[cod_mask] = best_actions
            losses[cod_mask] = best_losses

        return actions, losses
