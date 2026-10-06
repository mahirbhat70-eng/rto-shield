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

    def test_interventions_match_decomposed_totals(self, engine):
        """Verify that interventions section friction costs match decomposed operational totals."""
        assert abs(engine.interventions["REQUIRE_DEPOSIT"]["friction_cost"] - engine.compute_decomposed_friction("REQUIRE_DEPOSIT")) < 1e-4
        assert abs(engine.interventions["VERIFY_ADDRESS"]["friction_cost"] - engine.compute_decomposed_friction("VERIFY_ADDRESS")) < 1e-4


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


def test_deposit_operational_costs_nonzero(engine):
    """Ensure cost config defines non-zero operational costs for deposits (Task 10 & Mutation 19)."""
    cfg = engine.config
    assumed = cfg.get("assumed_operational_costs", {})
    deposit_friction = assumed.get("total_assumed_deposit_friction_cost", 0.0)
    assert deposit_friction > 0, "Deposit operational friction costs must be greater than zero"


def test_all_deposit_cost_components_nonzero(engine):
    """Ensure every individual decomposed operational cost component is strictly positive (Task 8 & Mutation 21)."""
    cfg = engine.config
    assumed = cfg.get("assumed_operational_costs", {})
    assert assumed.get("payment_gateway_fee_deposit_pct", 0.0) > 0.0, "PG fee percentage must be > 0"
    assert assumed.get("payment_gateway_fee_deposit_fixed", 0.0) > 0.0, "Fixed PG fee must be > 0"
    assert assumed.get("refund_processing_cost_unit", 0.0) > 0.0, "Refund fee must be > 0"
    assert assumed.get("support_ticket_cost_unit", 0.0) > 0.0, "Support ticket fee must be > 0"
    assert assumed.get("whatsapp_otp_template_cost", 0.0) > 0.0, "Messaging fee must be > 0"


def test_deposit_components_flow_into_expected_loss(engine):
    """Verify that deposit friction cost flows directly through to the evaluated expected loss."""
    ov, p = 1500.0, 0.60
    el_standard = engine.evaluate_interventions(ov, p)["REQUIRE_DEPOSIT"]
    # Decomposed calculation
    friction = engine.interventions["REQUIRE_DEPOSIT"]["friction_cost"]
    assert abs(friction - 7.00) < 1e-4, f"Engine deposit friction must be 7.00, got {friction}"
    p_success = (1.0 - p) * (1.0 - engine.interventions["REQUIRE_DEPOSIT"]["success_drop_pct"])
    p_rto_after = p * (1.0 - engine.interventions["REQUIRE_DEPOSIT"]["rto_reduction_pct"])
    margin = ov * engine.average_margin_pct
    expected_manual = friction + (p_rto_after * engine.rto_logistics_cost) - (p_success * margin)
    assert abs(el_standard - expected_manual) < 1e-6


def test_deposit_component_zero_mutation_changes_engine_loss():
    """Mutation check: zeroing any individual cost component MUST change the engine's expected loss."""
    import copy
    base_eng = CostEngine()
    ov, p = 1000.0, 0.50
    base_loss = base_eng.evaluate_interventions(ov, p)["REQUIRE_DEPOSIT"]

    components_to_zero = [
        ("payment_gateway_fee_deposit_fixed", 1.80),
        ("payment_gateway_fee_deposit_pct", 0.02 * 100.0),
        ("refund_processing_cost_unit", 3.00 * 0.40),
        ("support_ticket_cost_unit", 5.00 * 0.25),
        ("whatsapp_otp_template_cost", 0.75),
    ]

    for key, expected_delta in components_to_zero:
        cfg_mut = copy.deepcopy(base_eng.config)
        cfg_mut["assumed_operational_costs"][key] = 0.0
        mut_eng = CostEngine()
        mut_eng.config = cfg_mut
        mut_eng.assumed_operational_costs = cfg_mut["assumed_operational_costs"]
        # Trigger dynamic re-wiring
        mut_eng.interventions["REQUIRE_DEPOSIT"]["friction_cost"] = mut_eng.compute_decomposed_friction("REQUIRE_DEPOSIT")
        mut_loss = mut_eng.evaluate_interventions(ov, p)["REQUIRE_DEPOSIT"]
        diff = base_loss - mut_loss
        assert abs(diff - expected_delta) < 1e-4, (
            f"Zeroing {key} did not flow through to expected loss: expected delta {expected_delta}, got {diff}"
        )

