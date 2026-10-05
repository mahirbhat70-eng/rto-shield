"""
test_audit.py — Unit tests for the decision audit trail and replay verification.
"""

import json
import pytest
from src.serve.audit import build_audit_record, to_jsonl, verify_audit_record, compute_decision_fingerprint


@pytest.fixture
def sample_payload():
    return {
        "order_value": 616.0,
        "category": "Apparel",
        "payment_method": "COD",
        "quantity": 2,
        "discount_pct": 2.0,
        "cod_charge": 49.0,
        "account_age_days": 236,
        "prior_orders": 3,
        "prior_rto_count": 0,
        "orders_last_24h": 1,
        "device_cluster_size": 1,
        "pincode": "597542",
        "courier_id": "Courier_A"
    }


@pytest.fixture
def sample_res():
    return {
        "probability": 0.2131,
        "recommended_action": "VERIFY_ADDRESS",
        "el_table": {
            "ALLOW_COD": -50.12,
            "REQUIRE_DEPOSIT": -60.45,
            "VERIFY_ADDRESS": -67.72,
            "PREPAID_ONLY": 0.0
        },
        "shap_top_factors": [
            ("num_cod_charge", 0.4712),
            ("historical_pincode_rto_rate", -0.3211)
        ],
        "model_version": "2e6b0c5198dd",
        "warnings": []
    }


def test_build_audit_record_structure(sample_payload, sample_res):
    record = build_audit_record(sample_payload, sample_res, latency_ms=12.45)
    
    assert "decision_id" in record
    assert len(record["decision_id"]) == 16
    assert record["latency_ms"] == 12.45
    assert record["probability"] == 0.2131
    assert record["recommended_action"] == "VERIFY_ADDRESS"
    assert record["expected_losses"]["VERIFY_ADDRESS"] == -67.72
    assert len(record["top_factors"]) == 2
    assert verify_audit_record(record) is True

def test_pii_masking_in_audit_record(sample_res):
    payload_with_pii = {
        "order_value": 500.0,
        "customer_id": "CUST012345",
        "phone": "+91-9876543210",
        "email": "customer@example.com",
        "address": "Flat 402, Sunshine Apartments, MG Road, Bengaluru",
        "pincode": "560001",
        "payment_method": "COD"
    }
    record = build_audit_record(payload_with_pii, sample_res)
    assert "phone" not in record["payload"]
    assert "email" not in record["payload"]
    assert "address" not in record["payload"]
    assert record["payload"]["customer_id"].startswith("HASH_")
    assert verify_audit_record(record) is True



def test_audit_tamper_detection(sample_payload, sample_res):
    record = build_audit_record(sample_payload, sample_res, latency_ms=8.3)
    assert verify_audit_record(record) is True

    # Tampering with payload fails verification
    tampered_record = dict(record)
    tampered_record["payload"] = dict(record["payload"])
    tampered_record["payload"]["order_value"] = 9999.0
    assert verify_audit_record(tampered_record) is False


def test_audit_record_carries_model_version(sample_payload, sample_res):
    record = build_audit_record(sample_payload, sample_res, latency_ms=9.9)
    assert record["model_version"] == "2e6b0c5198dd"


def test_audit_decision_id_is_replay_stable(sample_payload, sample_res):
    # Same payload + same model => same decision_id, regardless of timestamp.
    import datetime
    r1 = build_audit_record(sample_payload, sample_res, latency_ms=10.0,
                            timestamp=datetime.datetime(2026, 1, 1))
    r2 = build_audit_record(sample_payload, sample_res, latency_ms=14.2,
                            timestamp=datetime.datetime(2026, 9, 6))
    assert r1["decision_id"] == r2["decision_id"]


def test_to_jsonl_serialization(sample_payload, sample_res):
    rec1 = build_audit_record(sample_payload, sample_res, latency_ms=10.1)
    rec2 = build_audit_record(sample_payload, sample_res, latency_ms=14.2)

    jsonl_output = to_jsonl([rec1, rec2])
    lines = jsonl_output.strip().split("\n")
    assert len(lines) == 2

    # Check parseable
    parsed1 = json.loads(lines[0])
    parsed2 = json.loads(lines[1])
    assert parsed1["decision_id"] == rec1["decision_id"]
    assert parsed2["decision_id"] == rec2["decision_id"]


def test_allowlist_drops_arbitrary_free_text(sample_res):
    payload_with_free_text = {
        "order_value": 750.0,
        "payment_method": "COD",
        "customer_notes": "Please deliver to John behind the hospital",
        "delivery_instructions": "Leave package with guard Rajesh 9876543210",
        "gift_message": "Happy Birthday to my brother!"
    }
    record = build_audit_record(payload_with_free_text, sample_res)
    assert "customer_notes" not in record["payload"]
    assert "delivery_instructions" not in record["payload"]
    assert "gift_message" not in record["payload"]
    assert "order_value" in record["payload"]

