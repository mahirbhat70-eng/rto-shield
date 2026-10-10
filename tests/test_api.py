"""
tests/test_api.py — FastAPI Serving Endpoints & Batch Benchmark Test Suite.
"""

import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.serve.api import app
from src.serve.tenants import TenantManager

client = TestClient(app)

DEMO_KEY = "fp_live_demo_merchant_123"
PILOT_KEY = "fp_live_pilot_alpha_456"

SAMPLE_ORDER = {
    "order_id": "test_ord_9901",
    "order_value": 1499.0,
    "quantity": 2.0,
    "category": "Apparel",
    "discount_pct": 10.0,
    "payment_method": "COD",
    "cod_charge": 50.0,
    "account_age_days": 150.0,
    "prior_orders": 3.0,
    "prior_rto_count": 0.0,
    "pincode": "110001",
    "courier_id": "Delhivery",
    "orders_last_24h": 1.0,
    "device_cluster_size": 1.0,
}

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert data["service"] == "FlipPrice AI Decision Engine"
    assert data["primary_model"] == "logistic_regression"

def test_unauthenticated_score_fails():
    resp = client.post("/score", json=SAMPLE_ORDER)
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["detail"]

def test_invalid_key_fails():
    resp = client.post("/score", json=SAMPLE_ORDER, headers={"X-API-Key": "fp_invalid_bad_key"})
    assert resp.status_code == 401
    assert "Invalid or deactivated API key" in resp.json()["detail"]

def test_single_order_score_success():
    resp = client.post("/score", json=SAMPLE_ORDER, headers={"X-API-Key": DEMO_KEY})
    assert resp.status_code == 200
    data = resp.json()

    assert data["order_id"] == "test_ord_9901"
    assert 0.0 <= data["probability_rto"] <= 1.0
    assert data["recommended_action"] in ["ALLOW_COD", "VERIFY_ADDRESS", "REQUIRE_DEPOSIT", "FORCE_PREPAID"]
    assert "interventions_evaluated" in data
    assert data["expected_savings_inr"] >= 0.0
    assert len(data["decision_fingerprint"]) == 16
    assert data["latency_ms"] > 0.0
    assert len(data["top_risk_drivers"]) > 0

def test_batch_scoring_benchmark_1000_orders_under_1_5s():
    """Verify batch scoring singleton satisfies < 1.5s for 1,000 orders."""
    import numpy as np
    rng = np.random.default_rng(42)
    pincodes = ["110001", "560038", "400001", "700001", "500001"]

    batch_orders = []
    for i in range(1000):
        batch_orders.append({
            "order_id": f"bench_{i}",
            "order_value": float(rng.uniform(500, 4500)),
            "quantity": float(rng.integers(1, 4)),
            "category": "Apparel",
            "discount_pct": float(rng.choice([0, 10, 20])),
            "payment_method": "COD",
            "cod_charge": 50.0,
            "account_age_days": float(rng.uniform(0, 500)),
            "prior_orders": float(rng.integers(0, 10)),
            "prior_rto_count": 0.0,
            "pincode": str(rng.choice(pincodes)),
            "courier_id": "Delhivery",
            "orders_last_24h": 1.0,
            "device_cluster_size": 1.0,
        })

    t0 = time.perf_counter()
    resp = client.post(
        "/score/batch",
        json={"orders": batch_orders, "include_shap": False},
        headers={"X-API-Key": DEMO_KEY},
    )
    elapsed = time.perf_counter() - t0

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_orders"] == 1000
    assert len(data["orders"]) == 1000

    # Strict SLA Assertion: < 1.5s for 1,000 orders
    assert elapsed < 1.5, f"Batch scoring took {elapsed:.3f}s, exceeding 1.5s SLA!"
    print(f"\n[BENCHMARK] 1,000 orders scored in {elapsed*1000:.1f}ms ({elapsed*1000/1000:.2f}ms / order)")
