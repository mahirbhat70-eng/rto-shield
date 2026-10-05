"""
audit/independent_engine.py
Independent reimplementation of the Expected-Loss Decision Engine.
Does NOT import from src/.
Uses only numpy, pandas, and yaml.
"""

import os
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "cost_config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

RTO_COST = float(CONFIG['rto_logistics_cost'])       # 150.0
MARGIN_PCT = float(CONFIG['average_margin_pct'])    # 0.20
INTERVENTIONS = CONFIG['interventions']

class IndependentCostEngine:
    def __init__(self, rto_cost=RTO_COST, margin_pct=MARGIN_PCT, interventions=INTERVENTIONS):
        self.rto_cost = rto_cost
        self.margin_pct = margin_pct
        self.interventions = interventions

    def evaluate_order(self, order_value: float, p_rto: float):
        """
        Calculates EL(a) for all 4 interventions:
        EL(a) = friction + P_RTO(a) * rto_cost - P_success(a) * margin
        where:
          margin = order_value * margin_pct
          P_success(a) = (1 - p_rto) * (1 - success_drop_pct)
          P_RTO(a) = p_rto * (1 - rto_reduction_pct)
        """
        V = float(order_value)
        p = float(p_rto)
        margin = V * self.margin_pct

        losses = {}
        for action, params in self.interventions.items():
            f = float(params['friction_cost'])
            d = float(params['success_drop_pct'])
            r = float(params['rto_reduction_pct'])

            p_success = (1.0 - p) * (1.0 - d)
            p_rto_after = p * (1.0 - r)

            el = f + (p_rto_after * self.rto_cost) - (p_success * margin)
            losses[action] = el

        best_action = min(losses, key=losses.get)
        return losses, best_action

    def evaluate_batch(self, order_values, p_rtos):
        V = np.asarray(order_values, dtype=float)
        p = np.asarray(p_rtos, dtype=float)
        margin = V * self.margin_pct

        el_list = []
        action_names = list(self.interventions.keys())
        for a in action_names:
            params = self.interventions[a]
            f = float(params['friction_cost'])
            d = float(params['success_drop_pct'])
            r = float(params['rto_reduction_pct'])

            p_succ = (1.0 - p) * (1.0 - d)
            p_rto_aft = p * (1.0 - r)
            el = f + (p_rto_aft * self.rto_cost) - (p_succ * margin)
            el_list.append(el)

        el_matrix = np.vstack(el_list) # (4, N)
        best_indices = np.argmin(el_matrix, axis=0)
        best_actions = [action_names[idx] for idx in best_indices]
        min_losses = el_matrix[best_indices, np.arange(len(V))]
        return best_actions, min_losses, el_matrix

if __name__ == "__main__":
    engine = IndependentCostEngine()
    print("Independent Cost Engine initialized.")
