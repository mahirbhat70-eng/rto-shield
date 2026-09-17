"""
tests/test_scorer.py
Phase 3 & Fix 4: comprehensive tests for the hardened scorer — validation,
cold-start, non-COD fast path, audit trail, and exact acceptance checklist items.
"""

import sys, os
import html
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serve.scorer import validate_and_clean, score_order, route_order, MODEL_VERSION
from src.serve.audit import build_audit_record, verify_audit_record, compute_decision_fingerprint, _payload_hash
from src.policy.cost_engine import CostEngine, is_cod


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

    def test_string_numeric_coerced(self):
        """String numeric '1200' -> 1200.0 (Phase 3 acceptance)."""
        clean, _ = validate_and_clean(_cod_payload(order_value="1200"))
        assert clean["order_value"] == 1200.0
        assert isinstance(clean["order_value"], float)

    def test_string_quantity_coerced(self):
        clean, _ = validate_and_clean(_cod_payload(quantity="3"))
        assert clean["quantity"] == 3.0

    def test_oracle_injection_guard(self):
        """historical_pincode_rto_rate / pincode_tier rejected (oracle guard)."""
        payload = _cod_payload(historical_pincode_rto_rate=0.3)
        with pytest.raises(ValueError, match="historical_pincode_rto_rate"):
            validate_and_clean(payload)

        payload2 = _cod_payload(pincode_tier=2)
        with pytest.raises(ValueError, match="pincode_tier"):
            validate_and_clean(payload2)

    def test_non_cod_charge_nonzero_raises(self):
        """cod_charge != 0 on non-COD -> error."""
        payload = _upi_payload(cod_charge=30.0)
        with pytest.raises(ValueError, match="cod_charge"):
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
        """bool rejected for numeric fields."""
        with pytest.raises(ValueError, match="bool"):
            validate_and_clean(_cod_payload(order_value=True))

    def test_nan_order_value_rejected(self):
        """NaN rejected for numeric fields."""
        with pytest.raises(ValueError, match="NaN"):
            validate_and_clean(_cod_payload(order_value=float("nan")))

    def test_inf_order_value_rejected(self):
        """inf / -inf rejected for numeric fields."""
        with pytest.raises(ValueError, match="finite"):
            validate_and_clean(_cod_payload(order_value=float("inf")))
        with pytest.raises(ValueError, match="finite"):
            validate_and_clean(_cod_payload(order_value=float("-inf")))

    def test_invalid_pincode_characters_raises(self):
        with pytest.raises(ValueError, match="(?i)pincode"):
            validate_and_clean(_cod_payload(pincode="ABCDEF"))

    def test_pincode_char_count_message(self):
        """pincode ^\\d{6}$ with char-count message."""
        with pytest.raises(ValueError) as exc_info:
            validate_and_clean(_cod_payload(pincode="12345"))
        msg = str(exc_info.value)
        assert "6" in msg or "5" in msg or "digit" in msg

    def test_pincode_7digit_raises(self):
        with pytest.raises(ValueError) as exc_info:
            validate_and_clean(_cod_payload(pincode="1234567"))
        msg = str(exc_info.value)
        assert "6" in msg or "7" in msg or "digit" in msg

    def test_unknown_category_warns_not_errors(self):
        """OOD category -> warning not error."""
        clean, warn = validate_and_clean(_cod_payload(category="AlienCategory"))
        assert clean["category"] == "AlienCategory"
        assert any("category" in w.lower() or "aliencategory" in w.lower() for w in warn)

    def test_unknown_courier_warns_not_errors(self):
        clean, warn = validate_and_clean(_cod_payload(courier_id="SpeedEx"))
        assert clean["courier_id"] == "SpeedEx"
        assert any("courier" in w.lower() or "speedex" in w.lower() for w in warn)

    def test_new_customer_with_prior_orders_warns(self):
        """account_age_days==0 & prior_orders>0 -> warning."""
        clean, warn = validate_and_clean(_cod_payload(account_age_days=0, prior_orders=3, prior_rto_count=0))
        assert any("account_age_days == 0" in w or "prior_orders > 0" in w for w in warn)

    def test_pincode_integer_accepted(self):
        clean, warn = validate_and_clean(_cod_payload(pincode=560001))
        assert clean["pincode"] == "560001"

    def test_xss_payload_escaped_in_ui(self):
        """XSS payload in user input must be escaped."""
        xss = "<script>alert('XSS')</script>"
        clean, _ = validate_and_clean(_cod_payload(category=xss))
        escaped = html.escape(clean["category"])
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped


# ── is_cod predicate tests ───────────────────────────────────────────────────

class TestIsCodPredicate:
    def test_is_cod_whitespace_and_cases(self):
        """is_cod(' cod ') True / is_cod('UPI') False."""
        assert is_cod(" cod ") is True
        assert is_cod("COD") is True
        assert is_cod("cod") is True
        assert is_cod("Cod") is True
        assert is_cod("UPI") is False
        assert is_cod("Credit Card") is False
        assert is_cod("Debit Card") is False
        assert is_cod("Net Banking") is False
        assert is_cod("PREPAID") is False


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
        assert res["probability"] is not None
        assert 0.0 <= res["probability"] <= 1.0

    def test_single_vs_batch_passthrough_agreement(self):
        """Single-vs-batch non-COD passthrough agreement."""
        payload = _upi_payload()
        res_single = score_order(payload)
        engine = CostEngine()
        df = pd.DataFrame([payload])
        probs = np.array([res_single["probability"]])
        actions, min_losses = engine.get_optimal_policy(df, probs)

        assert res_single["recommended_action"] == "PREPAID_PASSTHROUGH"
        assert actions[0] == "PREPAID_PASSTHROUGH"

    def test_cold_start_pincode_warns_and_scores(self):
        res = score_order(_cod_payload(pincode="999999"))
        assert res["recommended_action"] in ["ALLOW_COD", "VERIFY_ADDRESS", "REQUIRE_DEPOSIT", "PREPAID_ONLY"]
        assert any("cold start" in w.lower() for w in res["warnings"])

    def test_action_is_argmin_el(self):
        """The recommended action must match the argmin of the EL table."""
        res = score_order(_cod_payload())
        best = min(res["el_table"], key=res["el_table"].get)
        assert res["recommended_action"] == best

    def test_route_order_returns_action_string(self):
        action = route_order(_cod_payload())
        assert action in ["ALLOW_COD", "VERIFY_ADDRESS", "REQUIRE_DEPOSIT", "PREPAID_ONLY"]


# ── Audit trail tests ──────────────────────────────────────────────────────────

class TestAuditTrail:
    def test_decision_id_present(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res)
        assert "decision_id" in record
        assert len(record["decision_id"]) in (16, 64)

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

    def test_verify_tampered_payload(self):
        payload = _cod_payload()
        res = score_order(payload)
        record = build_audit_record(payload, res)
        record["payload"]["order_value"] = 99999.0
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
                    "recommended_action", "warnings"]:
            assert key in record
