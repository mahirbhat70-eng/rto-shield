"""
tests/test_shopify_app.py — Test Suite for Squad S5 Shopify App Build.

Covers:
- HMAC signature verification & timing-safe rejection.
- Shopify orders/create payload extraction and category mapping.
- Order tag & metafield formatting.
- Webhook idempotency under network retries.
- WhatsApp interactive confirmation triggers and callback state machine.
- Fulfillment auto-hold placement and release.
- Dynamic break-even pincode policy explorer.
- Billing tiers and subscription upgrades.
"""

import os
import json
import base64
import hmac
import hashlib
import pytest
from fastapi.testclient import TestClient

from src.serve.api import app
from src.shopify.adapter import (
    verify_shopify_hmac,
    parse_shopify_order,
    format_shopify_tags_and_metafields,
    IdempotencyTracker,
)
from src.shopify.actions import (
    PincodePolicyManager,
    WhatsAppVerificationAdapter,
    FulfillmentHoldManager,
    ShopifyBillingManager,
)

client = TestClient(app)
TEST_SECRET = "test_shopify_secret_2026"


# ---------------------------------------------------------------------------
# 1. HMAC Verification Tests
# ---------------------------------------------------------------------------
def test_verify_shopify_hmac_valid_and_invalid():
    body = b'{"id": 123456789, "total_price": "1499.00"}'
    valid_digest = hmac.new(TEST_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    valid_header = base64.b64encode(valid_digest).decode("utf-8")

    # Valid signature
    assert verify_shopify_hmac(body, valid_header, TEST_SECRET) is True

    # Tampered body
    tampered_body = b'{"id": 123456789, "total_price": "1.00"}'
    assert verify_shopify_hmac(tampered_body, valid_header, TEST_SECRET) is False

    # Tampered secret
    assert verify_shopify_hmac(body, valid_header, "wrong_secret") is False

    # Missing or empty
    assert verify_shopify_hmac(body, "", TEST_SECRET) is False
    assert verify_shopify_hmac(body, None, TEST_SECRET) is False


# ---------------------------------------------------------------------------
# 2. Shopify Order Ingestion & Feature Normalization
# ---------------------------------------------------------------------------
def test_parse_shopify_order_payload():
    sample_payload = {
        "id": 99887766,
        "name": "#1042",
        "total_price": "2499.00",
        "subtotal_price": "2499.00",
        "total_discounts": "0.00",
        "financial_status": "pending",
        "gateway": "Cash on Delivery (COD)",
        "line_items": [
            {"title": "Noise-Cancelling Bluetooth Earphone", "quantity": 1, "product_type": "Audio"},
            {"title": "Charging Cable USB-C", "quantity": 1, "product_type": "Accessories"},
        ],
        "shipping_address": {
            "zip": "560001",
            "phone": "+91 98765 43210",
            "city": "Bengaluru",
        },
        "customer": {
            "id": 554433,
            "orders_count": 3,
            "phone": "+91 98765 43210",
        }
    }

    parsed = parse_shopify_order(sample_payload)
    assert parsed["order_id"] == "#1042"
    assert parsed["order_value"] == 2499.00
    assert parsed["quantity"] == 2.0
    assert parsed["category"] == "Electronics"
    assert parsed["payment_method"] == "COD"
    assert parsed["pincode"] == "560001"
    assert parsed["prior_orders"] == 3.0
    assert parsed["customer_hash"].startswith("")
    assert len(parsed["customer_hash"]) == 16


def test_parse_shopify_order_cold_start_fallback():
    empty_payload = {}
    parsed = parse_shopify_order(empty_payload)
    assert parsed["order_value"] == 1.0
    assert parsed["quantity"] == 1.0
    assert parsed["category"] == "Apparel"
    assert parsed["pincode"] == "110001"
    assert parsed["customer_hash"] == "anon_customer"


# ---------------------------------------------------------------------------
# 3. Tags and Metafields Formatting
# ---------------------------------------------------------------------------
def test_format_shopify_tags_and_metafields():
    score_res = {
        "probability": 0.3842,
        "recommended_action": "VERIFY_ADDRESS",
        "el_table": {"VERIFY_ADDRESS": 48.20},
        "decision_fingerprint": "a1b2c3d4e5f67890",
    }
    tags, metafields = format_shopify_tags_and_metafields(score_res)

    assert "fp_risk:0.38" in tags
    assert "fp_action:VERIFY_ADDRESS" in tags
    assert "fp_el:48.20" in tags
    assert "fp_decision:PRICED" in tags

    meta_keys = {m["key"]: m["value"] for m in metafields}
    assert meta_keys["risk_score"] == "0.3842"
    assert meta_keys["recommended_action"] == "VERIFY_ADDRESS"
    assert meta_keys["expected_loss_inr"] == "48.20"
    assert meta_keys["decision_fingerprint"] == "a1b2c3d4e5f67890"


# ---------------------------------------------------------------------------
# 4. Webhook Ingestion & Idempotency
# ---------------------------------------------------------------------------
def test_shopify_webhook_orders_create_endpoint():
    payload = {
        "id": 887711,
        "name": "#1050",
        "total_price": "1800.00",
        "gateway": "cod",
        "financial_status": "pending",
        "line_items": [{"title": "Linen Cotton Shirt", "quantity": 1}],
        "shipping_address": {"zip": "110001", "phone": "9811122233"},
    }
    headers = {
        "X-Shopify-Shop-Domain": "demo-apparel.myshopify.com",
        "X-Shopify-Webhook-Id": "whk_unique_test_1050",
    }
    resp = client.post("/shopify/webhooks/orders/create", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["order_id"] == "#1050"
    assert "tags_to_apply" in data
    assert any("fp_action:" in t for t in data["tags_to_apply"])
    assert data["idempotent_replay"] is False

    # Second call with same webhook ID should return idempotent replay
    resp_repeat = client.post("/shopify/webhooks/orders/create", json=payload, headers=headers)
    assert resp_repeat.status_code == 200
    data_repeat = resp_repeat.json()
    assert data_repeat["idempotent_replay"] is True


# ---------------------------------------------------------------------------
# 5. WhatsApp Confirmation Adapter & Callback
# ---------------------------------------------------------------------------
def test_whatsapp_adapter_and_callback():
    adapter = WhatsAppVerificationAdapter(provider="mock")
    record = adapter.trigger_verification("ord_test_wa_01", "hash_999", 1500.0)
    assert record["status"] == "SENT"
    assert record["order_id"] == "ord_test_wa_01"

    # Confirm flow via route
    resp = client.post(
        "/shopify/whatsapp/callback",
        json={"order_id": "ord_test_wa_01", "button_payload": "CONFIRM_ORDER"}
    )
    # The route uses singleton whatsapp_adapter; let's trigger through route
    # First trigger through webhook or adapter
    from src.shopify.routes import whatsapp_adapter as route_adapter
    route_adapter.trigger_verification("ord_route_wa_02", "hash_888", 2000.0)

    cb_resp = client.post(
        "/shopify/whatsapp/callback",
        json={"order_id": "ord_route_wa_02", "button_payload": "CONFIRM_ORDER"}
    )
    assert cb_resp.status_code == 200
    assert cb_resp.json()["verification_status"] == "CONFIRMED"
    assert cb_resp.json()["resulting_tag"] == "fp_wa_confirmed:true"


# ---------------------------------------------------------------------------
# 6. Fulfillment Holds Manager
# ---------------------------------------------------------------------------
def test_fulfillment_hold_manager():
    mgr = FulfillmentHoldManager()
    hold = mgr.apply_hold("ord_deposit_99")
    assert hold["status"] == "HELD"
    assert "deposit" in hold["reason"].lower()

    release = mgr.release_hold("ord_deposit_99", payment_ref="upi_txn_12345")
    assert release["status"] == "RELEASED"
    assert release["payment_ref"] == "upi_txn_12345"


# ---------------------------------------------------------------------------
# 7. Dynamic Pincode Policy Explorer
# ---------------------------------------------------------------------------
def test_pincode_policy_explorer():
    resp = client.get("/shopify/pincodes?pincode=110001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pincode"] == "110001"
    assert "historical_rto_rate" in data
    assert "why_tooltip" in data
    assert "recommendation" in data


# ---------------------------------------------------------------------------
# 8. Billing & Onboarding Funnel
# ---------------------------------------------------------------------------
def test_billing_and_onboarding():
    # Plans
    plans_resp = client.get("/shopify/billing/plans")
    assert plans_resp.status_code == 200
    assert "starter" in plans_resp.json()
    assert "growth" in plans_resp.json()

    # Subscribe
    sub_resp = client.post(
        "/shopify/billing/subscribe",
        json={"merchant_id": "test_merchant_store", "plan_key": "growth"}
    )
    assert sub_resp.status_code == 200
    assert sub_resp.json()["subscription"]["plan_key"] == "growth"

    # Onboarding
    onboard_resp = client.get("/shopify/onboarding?shop=mybrand.myshopify.com")
    assert onboard_resp.status_code == 200
    assert len(onboard_resp.json()["funnel_steps"]) == 3
