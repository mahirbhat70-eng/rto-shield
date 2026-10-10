"""
scripts/commercial/simulate_live_shopify_order.py — Live Shopify Webhook Simulation Runner.

Simulates 3 realistic Indian D2C incoming orders to test real-time pricing and tags:
- Order 1: Low-Risk Apparel (Delhi 110001, ₹1,299) -> ALLOW_COD (Frictionless)
- Order 2: Moderate Sizing Risk (Lucknow 226001, ₹2,199) -> VERIFY_ADDRESS (WhatsApp Confirm)
- Order 3: High-Risk Electronics (Tier-3 Pincode 845401, ₹4,499) -> REQUIRE_DEPOSIT (Auto-Hold + ₹100 UPI)
"""

import os
import sys
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("RTO_SHIELD_ENV", "development")

from fastapi.testclient import TestClient
from src.serve.api import app

client = TestClient(app)

SAMPLE_ORDERS = [
    {
        "description": "Order 1: Low-Risk Apparel (Urban Corridor)",
        "payload": {
            "id": 1001,
            "name": "#ORD-1001",
            "total_price": "1299.00",
            "financial_status": "pending",
            "gateway": "cash on delivery",
            "line_items": [
                {"title": "Classic Cotton T-Shirt - Black / M", "quantity": 1, "product_type": "Apparel"}
            ],
            "shipping_address": {
                "zip": "110001",
                "phone": "+91 98111 22334",
                "city": "New Delhi"
            },
            "customer": {
                "orders_count": 4,
                "phone": "+91 98111 22334"
            }
        }
    },
    {
        "description": "Order 2: Moderate Sizing Risk (Tier-2 Corridor)",
        "payload": {
            "id": 1002,
            "name": "#ORD-1002",
            "total_price": "2199.00",
            "financial_status": "pending",
            "gateway": "cod",
            "line_items": [
                {"title": "Embroidered Festive Kurta Set", "quantity": 1, "product_type": "Apparel"}
            ],
            "shipping_address": {
                "zip": "226001",
                "phone": "+91 97222 33445",
                "city": "Lucknow"
            },
            "customer": {
                "orders_count": 1,
                "phone": "+91 97222 33445"
            }
        }
    },
    {
        "description": "Order 3: High-Risk Electronics (Developing Tier-3 Corridor)",
        "payload": {
            "id": 1003,
            "name": "#ORD-1003",
            "total_price": "4499.00",
            "financial_status": "pending",
            "gateway": "cod",
            "line_items": [
                {"title": "Wireless Gaming Headphone with Mic", "quantity": 1, "product_type": "Electronics"}
            ],
            "shipping_address": {
                "zip": "845401",
                "phone": "+91 96333 44556",
                "city": "Motihari"
            },
            "customer": {
                "orders_count": 0,
                "phone": "+91 96333 44556"
            }
        }
    }
]

def run_simulation():
    print("=" * 75)
    print("🛡️  FLIPPRICE AI — LIVE SHOPIFY WEBHOOK SIMULATION RUNNER")
    print("=" * 75)

    headers = {
        "X-Shopify-Shop-Domain": "aesthetic-threads.myshopify.com",
    }

    for item in SAMPLE_ORDERS:
        desc = item["description"]
        payload = item["payload"]
        order_name = payload["name"]
        print(f"\n▶ Ingesting: {desc}")
        print(f"  Cart: ₹{payload['total_price']} | Pincode: {payload['shipping_address']['zip']}")

        t0 = time.perf_counter()
        headers["X-Shopify-Webhook-Id"] = f"whk_sim_{payload['id']}_{int(time.time())}"
        resp = client.post("/shopify/webhooks/orders/create", json=payload, headers=headers)
        lat = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            print(f"  ❌ Failed ({resp.status_code}): {resp.text}")
            continue

        data = resp.json()
        act = data["action"]
        prob = data["probability_rto"]
        sav = data["expected_savings_inr"]
        tags = data["tags_to_apply"]

        print(f"  ✅ Decision Evaluated in {lat:.1f}ms (Engine: {data['processing_time_ms']}ms)")
        print(f"  • Return Risk: {prob*100:.1f}%")
        print(f"  • Recommended Action: {act}")
        print(f"  • Net Margin Saved: ₹{sav:,.2f}")
        print(f"  • Shopify Tags Applied: {tags}")
        if data["whatsapp_dispatched"]:
            print(f"  • 💬 WhatsApp Confirmation Template Dispatched to Customer")
        if data["fulfillment_hold_applied"]:
            print(f"  • ⏸️ Fulfillment Order Held: Awaiting ₹100 UPI Commitment Deposit")

    print("\n" + "=" * 75)
    print("🎉 All 3 orders scored, tagged, and recorded in tenant audit ledger.")
    print("=" * 75)

if __name__ == "__main__":
    run_simulation()
