"""
tests/test_cost_engine.py
Phase 3: comprehensive tests for CostEngine, is_cod(), and get_optimal_policy.
WHY: These are the most financially critical tests. A bug in cost_engine.py
     causes the ENTIRE recommendation layer to produce wrong actions — it's
     the highest-leverage place to have coverage.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd

from src.policy.cost_engine import CostEngine, is_cod


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    return CostEngine()


# ── is_cod() tests ─────────────────────────────────────────────────────────────

class TestIsCod:
    def test_cod_exact(self):
        assert is_cod("COD") is True

    def test_cod_lower(self):
        assert is_cod("cod") is True

    def test_cod_mixed(self):
        assert is_cod("Cod") is True

    def test_cod_whitespace(self):
        assert is_cod("  COD  ") is True

    def test_upi(self):
        assert is_cod("UPI") is False

    def test_credit_card(self):
        assert is_cod("Credit Card") is False

    def test_debit_card(self):
        assert is_cod("Debit Card") is False

    def test_net_banking(self):
        assert is_cod("Net Banking") is False

    def test_none(self):
        assert is_cod(None) is False

    def test_empty_string(self):
        assert is_cod("") is False

    def test_prepaid_not_cod(self):
        assert is_cod("PREPAID") is False

    def test_prepaid_only_not_cod(self):
        assert is_cod("PREPAID_ONLY") is False


# ── CostEngine config loading ──────────────────────────────────────────────────

class TestCostEngineConfig:
    def test_loads_without_error(self, engine):
        assert engine is not None

    def test_rto_logistics_cost(self, engine):
        assert engine.rto_logistics_cost == 150.0

    def test_average_margin_pct(self, engine):
        assert engine.average_margin_pct == 0.20

    def test_four_actions(self, engine):
        assert len(engine.interventions) == 4

    def test_required_actions_present(self, engine):
        assert "ALLOW_COD" in engine.interventions
        assert "VERIFY_ADDRESS" in engine.interventions
        assert "REQUIRE_DEPOSIT" in engine.interventions
        assert "PREPAID_ONLY" in engine.interventions


# ── evaluate_interventions (single order) ─────────────────────────────────────

class TestEvaluateInterventions:
    def test_returns_dict_with_all_actions(self, engine):
        result = engine.evaluate_interventions(1000.0, 0.5)
        assert set(result.keys()) == set(engine.interventions.keys())

    def test_high_risk_order_allow_is_expensive(self, engine):
        """At p=0.90, ALLOW_COD should have higher EL than VERIFY_ADDRESS."""
        result = engine.evaluate_interventions(2000.0, 0.90)
        assert result["ALLOW_COD"] > result["VERIFY_ADDRESS"]

    def test_zero_risk_allow_is_cheapest(self, engine):
        """At p=0.0, ALLOW_COD has EL=0 (no RTO cost), others drop conversion."""
        result = engine.evaluate_interventions(1000.0, 0.0)
        assert result["ALLOW_COD"] <= result["VERIFY_ADDRESS"]

    def test_el_is_float(self, engine):
        result = engine.evaluate_interventions(500.0, 0.3)
        for v in result.values():
            assert isinstance(v, float)

    def test_string_order_value_accepted(self, engine):
        """validate that string numeric inputs are coerced (real-world API payloads)."""
        result = engine.evaluate_interventions("1000.0", 0.5)
        assert isinstance(result["ALLOW_COD"], float)

    def test_allow_cod_formula_p0(self, engine):
        """ALLOW_COD: friction=0, r=0, d=0 → EL = p*C - (1-p)*margin*V."""
        ov, p = 1000.0, 0.3
        margin = ov * 0.20
        expected = 0 + p * 150 - (1 - p) * margin
        result = engine.evaluate_interventions(ov, p)
        assert abs(result["ALLOW_COD"] - expected) < 1e-6

    def test_verify_address_formula(self, engine):
        """VERIFY_ADDRESS: friction=2, r=0.30, d=0.05."""
        ov, p = 2000.0, 0.5
        margin = ov * 0.20
        r, d, friction = 0.30, 0.05, 2.0
        expected = friction + p * (1 - r) * 150 - (1 - p) * (1 - d) * margin
        result = engine.evaluate_interventions(ov, p)
        assert abs(result["VERIFY_ADDRESS"] - expected) < 1e-6


# ── vectorized ────────────────────────────────────────────────────────────────

class TestVectorized:
    def test_shapes(self, engine):
        ov   = np.array([500.0, 1000.0, 2000.0])
        prob = np.array([0.1, 0.4, 0.8])
        names, el = engine.evaluate_interventions_vectorized(ov, prob)
        assert el.shape == (4, 3)
        assert len(names) == 4

    def test_matches_scalar(self, engine):
        ov, p = 1500.0, 0.45
        scalar_result = engine.evaluate_interventions(ov, p)
        names, el_mat = engine.evaluate_interventions_vectorized(
            np.array([ov]), np.array([p])
        )
        for i, name in enumerate(names):
            assert abs(el_mat[i, 0] - scalar_result[name]) < 1e-6


# ── get_optimal_policy ─────────────────────────────────────────────────────────

class TestGetOptimalPolicy:
    def _make_df(self, n=5, pm="COD", ov=1000.0):
        return pd.DataFrame({
            "payment_method": [pm] * n,
            "order_value":    [ov] * n,
        })

    def test_length_mismatch_raises(self, engine):
        df = self._make_df(5)
        prob = pd.Series([0.3, 0.4])
        with pytest.raises(ValueError, match="Length mismatch"):
            engine.get_optimal_policy(df, prob)

    def test_cod_returns_action(self, engine):
        df = self._make_df(3, pm="COD", ov=2000.0)
        prob = pd.Series([0.8, 0.8, 0.8])
        actions, losses = engine.get_optimal_policy(df, prob)
        assert len(actions) == 3
        assert all(a in engine.interventions for a in actions)

    def test_non_cod_passthrough(self, engine):
        df = self._make_df(3, pm="UPI", ov=2000.0)
        prob = pd.Series([0.8, 0.8, 0.8])
        actions, losses = engine.get_optimal_policy(df, prob)
        assert all(a == "PREPAID_PASSTHROUGH" for a in actions)
        assert all(l == 0.0 for l in losses)

    def test_mixed_payment_methods(self, engine):
        df = pd.DataFrame({
            "payment_method": ["COD", "UPI", "COD", "Credit Card", "COD"],
            "order_value":    [1000, 2000, 500, 1500, 3000],
        })
        prob = pd.Series([0.8, 0.5, 0.1, 0.4, 0.9])
        actions, losses = engine.get_optimal_policy(df, prob)
        assert actions[1] == "PREPAID_PASSTHROUGH"
        assert actions[3] == "PREPAID_PASSTHROUGH"
        assert actions[0] in engine.interventions  # COD → real action
        assert actions[2] in engine.interventions
        assert actions[4] in engine.interventions

    def test_argmin_selects_lowest_el(self, engine):
        """At very high p, REQUIRE_DEPOSIT or PREPAID_ONLY should be selected."""
        df = self._make_df(1, pm="COD", ov=5000.0)
        prob = pd.Series([0.95])
        actions, losses = engine.get_optimal_policy(df, prob)
        el = engine.evaluate_interventions(5000.0, 0.95)
        best_manual = min(el, key=el.get)
        assert actions[0] == best_manual
