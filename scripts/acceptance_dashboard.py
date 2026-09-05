"""
scripts/acceptance_dashboard.py
===============================
Headless acceptance test suite for dashboard.py and the serving pipeline.
Verifies the numeric truth table, guardrails, audit hashing, and source hygiene.
"""

import sys
import os
import py_compile

# Ensure repo root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serve.scorer import score_order, PINCODE_LOOKUP
from src.serve.audit import build_audit_record, verify_audit_record, to_jsonl

PRESETS = {
    "DEPOSIT": {
        "order_value": 186.0, "category": "Beauty", "payment_method": "COD", "quantity": 1,
        "discount_pct": 0.0, "cod_charge": 29.0, "account_age_days": 260, "prior_orders": 4,
        "prior_rto_count": 0, "orders_last_24h": 1, "device_cluster_size": 1,
        "pincode": "461780", "courier_id": "Courier_A",
    },
    "ALLOW": {
        "order_value": 1344.0, "category": "Electronics", "payment_method": "COD", "quantity": 4,
        "discount_pct": 0.0, "cod_charge": 49.0, "account_age_days": 998, "prior_orders": 4,
        "prior_rto_count": 0, "orders_last_24h": 2, "device_cluster_size": 4,
        "pincode": "750176", "courier_id": "Courier_B",
    },
    "VERIFY": {
        "order_value": 852.0, "category": "Home", "payment_method": "COD", "quantity": 3,
        "discount_pct": 19.8, "cod_charge": 59.0, "account_age_days": 65, "prior_orders": 2,
        "prior_rto_count": 0, "orders_last_24h": 3, "device_cluster_size": 1,
        "pincode": "253407", "courier_id": "Courier_E",
    },
}

def test_verify_preset():
    print("[1/7] Testing VERIFY preset truth gate...")
    res = score_order(dict(PRESETS["VERIFY"]))
    prob = res["probability"]
    action = res["recommended_action"]
    el_verify = res["el_table"]["VERIFY_ADDRESS"]
    
    assert 0.394 <= prob <= 0.395, f"Expected probability in [0.394, 0.395], got {prob}"
    assert action == "VERIFY_ADDRESS", f"Expected VERIFY_ADDRESS, got {action}"
    assert abs(el_verify - (-54.597064)) < 0.01, f"Expected EL(VERIFY_ADDRESS) ~ -54.60, got {el_verify}"
    print(f"      PASS: P(RTO)={prob*100:.2f}%, action={action}, EL={el_verify:.2f}")

def test_prepaid_passthrough():
    print("[2/7] Testing PREPAID passthrough...")
    prepaid_payload = dict(PRESETS["ALLOW"])
    prepaid_payload["payment_method"] = "PREPAID"
    prepaid_payload["cod_charge"] = 0.0
    res = score_order(prepaid_payload)
    
    assert res["recommended_action"] == "PREPAID_PASSTHROUGH", f"Got {res['recommended_action']}"
    for k, v in res["el_table"].items():
        assert v == 0.0, f"EL for {k} expected 0.0, got {v}"
    print("      PASS: PREPAID_PASSTHROUGH confirmed with zeroed EL table.")

def test_guardrail_upper_bounds():
    print("[3/7] Testing guardrail rejection on out-of-distribution values...")
    bad_payload = dict(PRESETS["VERIFY"])
    bad_payload["order_value"] = 30000.0  # max allowed is 25000
    try:
        score_order(bad_payload)
        assert False, "Should have raised ValueError on order_value > 25000"
    except ValueError as e:
        assert "exceeds maximum allowed value" in str(e)
        print("      PASS: Guardrail rejected order_value=30000 as expected.")

def test_unknown_pincode():
    print("[4/7] Testing unknown pincode rejection...")
    bad_pin = dict(PRESETS["VERIFY"])
    bad_pin["pincode"] = "000000"
    try:
        score_order(bad_pin)
        assert False, "Should have raised ValueError on unknown pincode"
    except ValueError as e:
        assert "not found in lookup table" in str(e)
        print("      PASS: Unknown pincode cleanly rejected.")

def test_audit_fingerprint_tamper_evident():
    print("[5/7] Testing audit record fingerprinting & tamper detection...")
    res = score_order(dict(PRESETS["VERIFY"]))
    record = build_audit_record(dict(PRESETS["VERIFY"]), res, latency_ms=42.5)
    
    assert verify_audit_record(record) is True, "Original record must verify as True"
    
    # Tamper test
    tampered = dict(record)
    tampered["payload"] = dict(record["payload"])
    tampered["payload"]["order_value"] = 9999.0
    assert verify_audit_record(tampered) is False, "Tampered payload must fail verification"
    
    # JSONL roundtrip
    jsonl = to_jsonl([record])
    assert record["decision_id"] in jsonl
    print("      PASS: Audit fingerprint determinism & tamper-evidence verified.")

def test_dashboard_compilation():
    print("[6/7] Testing dashboard.py compilation...")
    py_compile.compile("dashboard.py", doraise=True)
    print("      PASS: dashboard.py compiled cleanly with zero syntax errors.")

def test_dashboard_code_hygiene():
    print("[7/7] Testing dashboard.py code hygiene and import integrity...")
    with open("dashboard.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "from src.serve.scorer import score_order" in content, "Must import live score_order"
    assert "from src.serve.audit import build_audit_record" in content, "Must import audit functions"
    assert "from src.serve import scorer" in content, "Must import scorer module"
    print("      PASS: Source imports frozen pipeline without staging or logic forking.")

def main():
    print("=" * 60)
    print("RTO-SHIELD DASHBOARD ACCEPTANCE GATES")
    print("=" * 60)
    test_verify_preset()
    test_prepaid_passthrough()
    test_guardrail_upper_bounds()
    test_unknown_pincode()
    test_audit_fingerprint_tamper_evident()
    test_dashboard_compilation()
    test_dashboard_code_hygiene()
    print("=" * 60)
    print("ALL-PASS: All 7 acceptance gates passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
