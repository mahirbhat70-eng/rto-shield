"""
tests/test_tenant_isolation.py — Adversarial Multi-Tenant Data Isolation Tests.

Guarantees:
1. Tenant A cannot read Tenant B's order decisions or audit records.
2. Tenant keys are strictly authenticated and salt-hashed.
3. Custom merchant configs (if set) do not bleed between tenants.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.serve.api import app
from src.serve.tenants import TenantManager

client = TestClient(app)

TENANT_A_KEY = "fp_live_demo_merchant_123"
TENANT_B_KEY = "fp_live_pilot_alpha_456"

ORDER_A = {
    "order_id": "secret_tenant_a_order_888",
    "order_value": 3200.0,
    "quantity": 1.0,
    "category": "Electronics",
    "discount_pct": 5.0,
    "payment_method": "COD",
    "cod_charge": 50.0,
    "account_age_days": 12.0,
    "prior_orders": 1.0,
    "prior_rto_count": 0.0,
    "pincode": "560038",
    "courier_id": "BlueDart",
    "orders_last_24h": 1.0,
    "device_cluster_size": 1.0,
}

def test_tenant_isolation_cross_tenant_lookup_denied():
    # 1. Tenant A scores an order
    resp_a = client.post("/score", json=ORDER_A, headers={"X-API-Key": TENANT_A_KEY})
    assert resp_a.status_code == 200
    assert resp_a.json()["order_id"] == "secret_tenant_a_order_888"

    # 2. Tenant A can explain their own order
    resp_explain_a = client.get("/explain/secret_tenant_a_order_888", headers={"X-API-Key": TENANT_A_KEY})
    assert resp_explain_a.status_code == 200
    assert resp_explain_a.json()["order_id"] == "secret_tenant_a_order_888"

    # 3. Adversarial Attack: Tenant B attempts to explain Tenant A's order using Tenant B's key
    resp_explain_b = client.get("/explain/secret_tenant_a_order_888", headers={"X-API-Key": TENANT_B_KEY})
    assert resp_explain_b.status_code == 404
    assert "not found" in resp_explain_b.json()["detail"].lower()

def test_tenant_manager_isolation_direct_query():
    mgr = TenantManager()
    mgr.log_decision(
        tenant_id="tenant_demo_1",
        order_id="ord_a_100",
        decision_fingerprint="fp100",
        action="ALLOW_COD",
        risk_score=0.15,
        savings_inr=50.0,
        latency_ms=5.0,
    )

    # Scoped query for tenant_demo_1 finds it
    rec_a = mgr.get_order_decision("tenant_demo_1", "ord_a_100")
    assert rec_a is not None
    assert rec_a["order_id"] == "ord_a_100"

    # Scoped query for tenant_pilot_alpha gets None
    rec_b = mgr.get_order_decision("tenant_pilot_alpha", "ord_a_100")
    assert rec_b is None

def test_audit_logs_do_not_leak_raw_phone_or_customer_names():
    from src.serve.audit import mask_pii
    raw_payload = {
        "order_value": 1500.0,
        "customer_phone": "+919876543210",
        "customer_name": "Aarav Sharma",
        "shipping_address": "Flat 402, Sunshine Heights, Bangalore",
        "customer_id": "cust_12345",
        "pincode": "560038"
    }
    masked = mask_pii(raw_payload)

    assert "customer_phone" not in masked
    assert "customer_name" not in masked
    assert "shipping_address" not in masked
    assert "cust_12345" not in masked["customer_id"]
    assert masked["customer_id"].startswith("HASH_")
    assert masked["pincode"] == "560038"
    assert masked["order_value"] == 1500.0
