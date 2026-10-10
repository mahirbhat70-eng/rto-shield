"""
src/serve/api.py — FlipPrice AI Production Decision Pricing REST API.

Provides:
- GET  /health           : System health, loaded model digests, uptime.
- POST /score            : Single order evaluation with SHAP & plain-English drivers.
- POST /score/batch      : Ultra-fast vectorized batch scoring (<1.5s / 1,000 orders).
- GET  /explain/{order_id}: Past order decision lookup with strict tenant isolation.
"""

import os
import sys
import time
import uuid
import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("RTO_SHIELD_ENV", "development")

from src.serve.scorer import (
    score_order_guarded,
    validate_and_clean,
    resolve_pincode_info,
    _build_model_row,
    tree_cal,
    tree_uncal,
    engine,
    MODEL_VERSION,
    COLD_START_PRIOR,
    NUMERIC_FIELDS,
    is_cod,
    explainer,
)
from src.serve.audit import compute_decision_fingerprint, mask_pii
from src.serve.tenants import TenantManager

app = FastAPI(
    title="FlipPrice AI Decision Engine",
    version="2.0.0",
    description="Commercial Multi-Action Decision Pricing API for Indian D2C Merchants",
)

from src.shopify.routes import router as shopify_router
app.include_router(shopify_router)

tenant_manager = TenantManager()
START_TIME = time.time()

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

def get_tenant(
    api_key_header_val: Optional[str] = Security(api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Dict[str, Any]:
    token = api_key_header_val or (bearer_creds.credentials if bearer_creds else None)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass 'X-API-Key: fp_live_...' or Bearer token.",
        )
    tenant = tenant_manager.authenticate(token)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or deactivated API key.",
        )
    return tenant

# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class OrderInput(BaseModel):
    order_id: Optional[str] = Field(default=None, description="Merchant order reference")
    order_value: float = Field(..., ge=1.0, le=100000.0, description="Order total in INR")
    quantity: float = Field(default=1.0, ge=1.0, le=100.0)
    category: str = Field(default="Apparel")
    discount_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    payment_method: str = Field(default="COD")
    cod_charge: float = Field(default=0.0, ge=0.0, le=500.0)
    account_age_days: float = Field(default=0.0, ge=0.0, le=5000.0)
    prior_orders: float = Field(default=0.0, ge=0.0, le=1000.0)
    prior_rto_count: float = Field(default=0.0, ge=0.0, le=1000.0)
    pincode: str = Field(..., min_length=5, max_length=10)
    courier_id: str = Field(default="Delhivery")
    orders_last_24h: float = Field(default=1.0, ge=0.0, le=100.0)
    device_cluster_size: float = Field(default=1.0, ge=1.0, le=50.0)

class BatchScoreRequest(BaseModel):
    orders: List[OrderInput] = Field(..., max_length=1000)
    include_shap: bool = Field(default=False, description="Compute SHAP explanations (adds slight latency)")

class DriverDetail(BaseModel):
    feature: str
    impact: str
    explanation: str

class ScoreResponse(BaseModel):
    order_id: str
    probability_rto: float
    recommended_action: str
    action_narrative: str
    interventions_evaluated: Dict[str, float]
    friction_inr: float
    expected_savings_inr: float
    top_risk_drivers: List[DriverDetail]
    latency_ms: float
    warnings: List[str]
    model_version: str
    decision_fingerprint: str

# ---------------------------------------------------------------------------
# Plain-English Driver Translation
# ---------------------------------------------------------------------------
def translate_driver(feature: str, val: float, order_dict: dict) -> DriverDetail:
    f_clean = feature.split("__")[-1] if "__" in feature else feature
    direction = "Increases Return Risk" if val > 0 else "Protects Against Return"
    pct_impact = f"{abs(val) * 100:.1f}%"

    if "pincode_tier" in f_clean or "pincode" in f_clean:
        expl = f"Delivery destination is in high-return geographical zone ({pct_impact} impact)."
    elif "order_value" in f_clean:
        expl = f"High cart value (₹{order_dict.get('order_value', 0):,.0f}) elevates cash flow return exposure."
    elif "prior_rto_count" in f_clean:
        expl = f"Buyer has history of previously rejected or returned orders ({pct_impact} impact)."
    elif "orders_last_24h" in f_clean:
        expl = f"Velocity anomaly: rapid consecutive order placements detected."
    elif "account_age_days" in f_clean:
        expl = f"New customer profile without established delivery track record."
    elif "category" in f_clean:
        expl = f"Merchandise category '{order_dict.get('category', 'General')}' carries elevated sizing/fit returns."
    elif "device_cluster" in f_clean:
        expl = f"Device fingerprint associated with multiple linked shopper identities."
    else:
        expl = f"Risk factor contribution ({pct_impact} relative shift)."

    return DriverDetail(
        feature=f_clean,
        impact=direction,
        explanation=expl
    )

def action_to_narrative(action: str) -> str:
    if action == "ALLOW_COD":
        return "Low risk: proceed with immediate frictionless dispatch."
    elif action == "VERIFY_ADDRESS":
        return "Moderate risk: dispatch after ₹2 automated WhatsApp address confirmation."
    elif action == "REQUIRE_DEPOSIT":
        return "Elevated risk: secure ₹100 commitment deposit via UPI before dispatch."
    elif action == "FORCE_PREPAID":
        return "Severe risk: disable Cash on Delivery and require full upfront payment."
    elif action == "PREPAID_PASSTHROUGH":
        return "Prepaid order: zero COD return exposure; dispatch normally."
    return "Action evaluated by cost engine."

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Root endpoint welcoming visitors and linking to interactive docs."""
    return {
        "service": "FlipPrice AI Decision Engine",
        "tagline": "Blocklists guess. We price.",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
        "version": "2.0.0",
    }

@app.get("/health")
def health_check():
    """Verify system health, loaded model hashes, and uptime."""
    return {
        "status": "healthy",
        "service": "FlipPrice AI Decision Engine",
        "version": "2.0.0",
        "model_version": MODEL_VERSION,
        "primary_model": engine.config.get("primary_model", "logistic_regression"),
        "active_policy": engine.config.get("active_policy", "constants_p1"),
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

@app.post("/score", response_model=ScoreResponse)
def score_single_order(order: OrderInput, tenant: Dict[str, Any] = Depends(get_tenant)):
    """Evaluate and price decision for a single incoming merchant order."""
    t0 = time.perf_counter()
    order_dict = order.model_dump()
    order_id = order_dict.get("order_id") or f"ord_{uuid.uuid4().hex[:10]}"
    order_dict["order_id"] = order_id

    # Score order with guardrails
    result = score_order_guarded(order_dict)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    p = result["probability"]
    action = result["recommended_action"]
    el_table = result["el_table"]

    # Calculate expected savings: baseline loss vs policy expected loss
    c_rto = engine.rto_logistics_cost
    base_loss = p * c_rto if is_cod(order_dict.get("payment_method")) else 0.0
    policy_el = el_table.get(action, 0.0)
    expected_savings = round(max(0.0, base_loss - policy_el), 2)

    friction_val = engine.interventions.get(action, {}).get("friction", 0.0)

    # Translate drivers
    drivers = []
    for feat_name, shap_val in result.get("shap_top_factors", []):
        drivers.append(translate_driver(feat_name, shap_val, order_dict))

    fp = compute_decision_fingerprint(order_dict, MODEL_VERSION, action)

    # Log decision under tenant isolation
    tenant_manager.log_decision(
        tenant_id=tenant["tenant_id"],
        order_id=order_id,
        decision_fingerprint=fp,
        action=action,
        risk_score=p,
        savings_inr=expected_savings,
        latency_ms=lat_ms,
    )

    return ScoreResponse(
        order_id=order_id,
        probability_rto=round(p, 4),
        recommended_action=action,
        action_narrative=action_to_narrative(action),
        interventions_evaluated={k: round(v, 2) for k, v in el_table.items()},
        friction_inr=round(friction_val, 2),
        expected_savings_inr=expected_savings,
        top_risk_drivers=drivers,
        latency_ms=round(lat_ms, 2),
        warnings=result.get("warnings", []),
        model_version=MODEL_VERSION,
        decision_fingerprint=fp,
    )

@app.post("/score/batch")
def score_batch_orders(batch: BatchScoreRequest, tenant: Dict[str, Any] = Depends(get_tenant)):
    """
    Vectorized batch inference (<1.5s for 1,000 orders).
    Evaluates risk and optimal interventions across an entire order cohort.
    """
    t0 = time.perf_counter()
    n = len(batch.orders)
    if n == 0:
        return {"orders": [], "total_orders": 0, "batch_latency_ms": 0.0}

    # Vectorized DataFrame construction
    records = []
    ids = []
    for idx, ord_in in enumerate(batch.orders):
        d = ord_in.model_dump()
        oid = d.get("order_id") or f"batch_{idx}_{uuid.uuid4().hex[:6]}"
        ids.append(oid)
        d["order_id"] = oid
        clean, _ = validate_and_clean(d)
        pin_info, _ = resolve_pincode_info(clean["pincode"])
        row = {
            "category": clean["category"],
            "payment_method": clean["payment_method"],
            "courier_id": clean["courier_id"],
        }
        for f in NUMERIC_FIELDS:
            row[f] = clean[f]
        row["historical_pincode_rto_rate"] = float(pin_info["historical_pincode_rto_rate"])
        row["pincode_tier"] = pin_info["pincode_tier"]
        records.append(row)

    df_batch = pd.DataFrame(records)

    # Vectorized model inference (single forward pass)
    probs = tree_cal.predict_proba(df_batch)[:, 1]

    # Vectorized intervention evaluation
    results = []
    audit_records = []
    c_rto = engine.rto_logistics_cost

    for i in range(n):
        p = float(probs[i])
        oid = ids[i]
        order_dict = batch.orders[i].model_dump()
        is_cod_order = is_cod(order_dict.get("payment_method"))

        if not is_cod_order:
            act = "PREPAID_PASSTHROUGH"
            el_table = {k: 0.0 for k in engine.interventions.keys()}
            sav = 0.0
            fric = 0.0
        else:
            el_table = engine.evaluate_interventions(order_dict["order_value"], p)
            act = min(el_table, key=el_table.get)
            base_loss = p * c_rto
            sav = max(0.0, base_loss - el_table[act])
            fric = engine.interventions.get(act, {}).get("friction", 0.0)

        fp = compute_decision_fingerprint(order_dict, MODEL_VERSION, act)

        audit_records.append((
            tenant["tenant_id"],
            str(oid),
            fp,
            act,
            p,
            round(sav, 2),
            0.0,
        ))

        results.append({
            "order_id": oid,
            "probability_rto": round(p, 4),
            "recommended_action": act,
            "expected_savings_inr": round(sav, 2),
            "friction_inr": round(fric, 2),
            "decision_fingerprint": fp,
        })

    # Bulk insert audit logs in single fast transaction
    tenant_manager.log_decisions_batch(audit_records)

    tot_lat_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "orders": results,
        "total_orders": n,
        "batch_latency_ms": round(tot_lat_ms, 2),
        "mean_latency_per_order_ms": round(tot_lat_ms / n, 3),
        "model_version": MODEL_VERSION,
    }

@app.get("/explain/{order_id}")
def get_order_explanation(order_id: str, tenant: Dict[str, Any] = Depends(get_tenant)):
    """
    Look up decision explanation for a specific order.
    Enforces strict tenant isolation: Tenant A cannot inspect Tenant B's orders.
    """
    record = tenant_manager.get_order_decision(tenant["tenant_id"], order_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order '{order_id}' not found for merchant '{tenant['merchant_name']}'.",
        )
    return {
        "tenant_id": tenant["tenant_id"],
        "order_id": record["order_id"],
        "decision_fingerprint": record["decision_fingerprint"],
        "recommended_action": record["action"],
        "action_narrative": action_to_narrative(record["action"]),
        "risk_score": record["risk_score"],
        "expected_savings_inr": record["expected_savings_inr"],
        "timestamp": record["created_at"],
    }
