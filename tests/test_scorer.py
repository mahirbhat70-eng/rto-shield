"""
tests/test_scorer.py
Phase 3: tests for the hardened scorer — validation, cold-start, non-COD fast path,
         audit trail, and output contract.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.serve.scorer import validate_and_clean, score_order, MODEL_VERSION
from src.serve.audit import build_audit_record, verify_audit_record, _payload_hash


# ── Base valid payload ─────────────────────────────────────────────────────────

def _cod_payload(**overrides):
    base = {
        "order_value": 1500.0, "quantity": 2, "category": "Apparel",
        "discount_pct": 10.0, "payment_method": "COD", "cod_charge": 39.0,
        "account_age_days": 200, "prior_orders": 5, "prior_rto_count": 1,
        "pincode": "560001", "courier_id": "Courier_A",
        "orders_last_24h": 1, "device_cluster_size": 2,
    }
    base.update(overrides)
    return base


def _upi_payload(**overrides):
    base = _cod_payload(payment_method="UPI", cod_charge=0.0)
    base.update(overrides)
    return base


# ── validate_and_clean tests ───────────────────────────────────────────────────

class TestValidateAndClean:
    def test_valid_cod_passes(self):
        clean, warn = validate_and_clean(_cod_payload())
        assert "order_value" in clean
        assert isinstance(warn, list)

    def test_missing_field_raises(self):
        payload = _cod_payload()
        del payload["order_value"]
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_and_clean(payload)

    def test_multiple_errors_single_raise(self):
        """All errors must be collected into ONE ValueError, not the first."""
        payload = _cod_payload(order_value=-100, quantity=-5)
        with pytest.raises(ValueError) as exc_info:
            validate_and_clean(payload)
        msg = str(exc_info.value)
        assert "order_value" in msg
        assert "quantity" in msg

    def test_oracle_fields_blocked(self):
        payload = _cod_payload()
        payload["historical_pincode_rto_rate"] = 0.3
        with pytest.raises(ValueError, match="Oracle features"):
            validate_and_clean(payload)

    def test_non_cod_charge_nonzero_raises(self):
        payload = _upi_payload(cod_charge=30.0)
        with pytest.raises(ValueError, match="cod_charge must be 0"):
            validate_and_clean(payload)

    def test_non_cod_charge_zero_passes(self):
        payload = _upi_payload(cod_charge=0.0)
        clean, warn = validate_and_clean(payload)
        assert clean["cod_charge"] == 0.0

    def test_prior_rto_exceeds_prior_orders_raises(self):
        payload = _cod_payload(prior_orders=2, prior_rto_count=5)
        with pytest.raises(ValueError, match="prior_rto_count"):
            validate_and_clean(payload)

    def test_order_value_out_of_range_raises(self):
        with pytest.raises(ValueError, match="order_value"):
            validate_and_clean(_cod_payload(order_value=50000.0))

    def test_order_value_below_min_raises(self):
        with pytest.raises(ValueError, match="order_value"):
            validate_and_clean(_cod_payload(order_value=0.5))

    def test_boolean_order_value_rejected(self):
        with pytest.raises(ValueError, match="order_value"):
            validate_and_clean(_cod_payload(order_value=True))

    def test_invalid_pincode_raises(self):
        with pytest.raises(ValueError, match="pincode"):
            validate_and_clean(_cod_payload(pincode="ABCDEF"))

    def test_5digit_pincode_raises(self):
        with pytest.raises(ValueError, match="pincode"):
            validate_and_clean(_cod_payload(pincode="12345"))

    def test_unknown_category_warns(self):
        _, warn = validate_and_clean(_cod_payload(category="Unicorn"))
        assert any("unknown" in w.lower() or "OOD" in w for w in warn)

    def test_unknown_courier_warns(self):
        _, warn = validate_and_clean(_cod_payload(courier_id="SpeedEx"))
        assert any("unknown" in w.lower() or "OOD" in w for w in warn)

    def test_prepaid_payment_warns(self):
        _, warn = validate_and_clean(_cod_payload(payment_method="PREPAID", cod_charge=0.0))
        assert any("PREPAID" in w or "OOD" in w for w in warn)

    def test_pincode_integer_accepted(self):
        clean, warn = validate_and_clean(_cod_payload(pincode=560001))
        assert clean["pincode"] == "560001"

    def test_nan_order_value_raises(self):
        import math
        with pytest.raises(ValueError):
            validate_and_clean(_cod_payload(order_value=float("nan")))

    def test_inf_order_value_raises(self):
        with pytest.raises(ValueError):
            validate_and_clean(_cod_payload(order_value=float("inf")))


# ── score_order output contract ────────────────────────────────────────────────

class TestScoreOrder:
    def test_cod_returns_probability(self):
        res = score_order(_cod_payload())
        assert res["probability"] is not None
        assert 0.0 <= res["probability"] <= 1.0

    def test_cod_returns_el_table(self):
        res = score_order(_cod_payload())
        assert "el_table" in res
        assert len(res["el_table"]) == 4

    def test_cod_returns_action(self):
        res = score_order(_cod_payload())
        assert res["recommended_action"] in ["ALLOW_COD", "VERIFY_ADDRESS",
                                              "REQUIRE_DEPOSIT", "PREPAID_ONLY"]

    def test_cod_returns_shap(self):
        res = score_order(_cod_payload())
        assert isinstance(res["shap_top_factors"], list)
        assert len(res["shap_top_factors"]) <= 3

    def test_cod_model_version(self):
        res = score_order(_cod_payload())
        assert len(res["model_version"]) == 12  # 12-hex-char SHA-256 prefix

    def test_non_cod_action_is_passthrough(self):
        res = score_order(_upi_payload())
        assert res["recommended_action"] == "PREPAID_PASSTHROUGH"

    def test_non_cod_probability_is_none(self):
        res = score_order(_upi_payload())
        assert res["probability"] is None

    def test_non_cod_shap_is_empty(self):
        res = score_order(_upi_payload())
        assert res["shap_top_factors"] == []

    def test_cold_start_pincode_warns(self):
        res = score_order(_cod_payload(pincode="999999"))
        assert any("global prior" in w.lower() for w in res["warnings"])

    def test_deterministic_same_input_same_output(self):
        payload = _cod_payload()
        r1 = score_order(payload)
        r2 = score_order(payload)
        assert r1["probability"] == r2["probability"]
        assert r1["recommended_action"] == r2["recommended_action"]
        assert r1["model_version"] == r2["model_version"]

    def test_validation_error_propagates(self):
        with pytest.raises(ValueError):
            score_order(_cod_payload(order_value=-1))

    def test_credit_card_is_passthrough(self):
        res = score_order(_cod_payload(payment_method="Credit Card", cod_charge=0.0))
        assert res["recommended_action"] == "PREPAID_PASSTHROUGH"

    def test_net_banking_is_passthrough(self):
        res = score_order(_cod_payload(payment_method="Net Banking", cod_charge=0.0))
        assert res["recommended_action"] == "PREPAID_PASSTHROUGH"

    def test_debit_card_is_passthrough(self):
        res = score_order(_cod_payload(payment_method="Debit Card", cod_charge=0.0))
        assert res["recommended_action"] == "PREPAID_PASSTHROUGH"

    def test_action_is_argmin_el(self):
        """The recommended action must match the argmin of the EL table."""
        res = score_order(_cod_payload())
        best = min(res["el_table"], key=res["el_table"].get)
        assert res["recommended_action"] == best

    def test_model_version_is_consistent(self):
        res1 = score_order(_cod_payload())
        res2 = score_order(_cod_payload(order_value=500.0))
        # Model version should not change between calls on same loaded model
        assert res1["model_version"] == res2["model_version"]


# ── Audit trail tests ──────────────────────────────────────────────────────────

class TestAuditTrail:
    def test_decision_id_present(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res)
        assert "decision_id" in record
        assert len(record["decision_id"]) == 64  # SHA-256 hex

    def test_decision_id_deterministic(self):
        payload = _cod_payload()
        res = score_order(payload)
        r1 = build_audit_record(payload, res)
        r2 = build_audit_record(payload, res)
        assert r1["decision_id"] == r2["decision_id"]

    def test_verify_intact_record(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res)
        assert verify_audit_record(record) is True

    def test_verify_tampered_record(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res)
        # Tamper with the recommended action
        record["recommended_action"] = "ALLOW_COD"
        assert verify_audit_record(record) is False

    def test_latency_ms_stored(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res, latency_ms=12.3)
        assert record["latency_ms"] == pytest.approx(12.3)

    def test_record_has_required_keys(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res)
        for key in ["decision_id", "model_version", "payload", "probability",
                    "el_table", "recommended_action", "shap_top_factors", "warnings"]:
            assert key in record
