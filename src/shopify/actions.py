"""
src/shopify/actions.py — Commercial Actions Layer for Shopify App.

Implements:
1. PincodePolicyManager (A-03): Expected-Loss seeded dynamic COD control with explainable tooltips.
2. WhatsAppVerificationAdapter (A-05): Gupshup/Wati/Interakt template dispatch & response listener.
3. FulfillmentHoldManager (A-06): Auto-hold on REQUIRE_DEPOSIT with guardrails release.
4. ShopifyBillingManager (A-07): Tiered subscription and usage billing management.
"""

import os
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.serve.scorer import COLD_START_PRIOR, resolve_pincode_info, engine


# ---------------------------------------------------------------------------
# 1. Pincode COD Control & "Anti-Blocklist" Policy (A-03, A-04)
# ---------------------------------------------------------------------------
class PincodePolicyManager:
    """
    Computes break-even COD policy per pincode based on expected loss rather than blunt blacklists.
    Formula: An order is loss-making under unconditional ALLOW_COD if p_rto * C_rto > (1 - p_rto) * margin.
    Instead of blocking COD, FlipPrice prices the risk with VERIFY_ADDRESS or REQUIRE_DEPOSIT.
    """
    def __init__(self, cost_engine=engine):
        self.engine = cost_engine

    def evaluate_pincode(
        self,
        pincode: str,
        sample_order_value: float = 1200.0,
        margin_pct: float = 0.20
    ) -> Dict[str, Any]:
        info, is_cold = resolve_pincode_info(pincode)
        hist_rate = info.get("historical_pincode_rto_rate", COLD_START_PRIOR["historical_pincode_rto_rate"])
        tier = info.get("pincode_tier", COLD_START_PRIOR["pincode_tier"])

        c_rto = self.engine.rto_logistics_cost
        margin_inr = sample_order_value * margin_pct
        expected_allow_loss = hist_rate * c_rto - (1.0 - hist_rate) * margin_inr

        # Multi-action pricing on this pincode baseline
        el_table = self.engine.evaluate_interventions(sample_order_value, hist_rate)
        best_action = min(el_table, key=el_table.get)
        el_savings = max(0.0, expected_allow_loss - el_table[best_action])

        # Generate "Why" explanation tooltip
        if expected_allow_loss > 0:
            tooltip = (
                f"Pincode {pincode} (Tier {tier}, {hist_rate*100:.1f}% baseline RTO): "
                f"Static blacklists would forfeit ₹{margin_inr:,.0f} margin by denying COD. "
                f"FlipPrice routes to {best_action}, saving ₹{el_savings:,.1f}/order while keeping the sale."
            )
            recommendation = "PRICE_WITH_FRICTION"
        else:
            tooltip = (
                f"Pincode {pincode} (Tier {tier}, {hist_rate*100:.1f}% baseline RTO): "
                f"Net profitable under standard delivery (+₹{-expected_allow_loss:,.1f} expected margin). "
                f"Safe for instant frictionless dispatch."
            )
            recommendation = "ALLOW_COD"

        return {
            "pincode": pincode,
            "pincode_tier": tier,
            "historical_rto_rate": round(hist_rate, 4),
            "is_cold_start": is_cold,
            "recommendation": recommendation,
            "best_action": best_action,
            "expected_allow_loss_inr": round(expected_allow_loss, 2),
            "expected_priced_savings_inr": round(el_savings, 2),
            "why_tooltip": tooltip,
        }


# ---------------------------------------------------------------------------
# 2. WhatsApp COD Confirmation Adapter (A-05)
# ---------------------------------------------------------------------------
class WhatsAppVerificationAdapter:
    """
    Multi-provider adapter for WhatsApp Business API (Gupshup / Wati / Interakt).
    Fired when an order is routed to VERIFY_ADDRESS.
    """
    SUPPORTED_PROVIDERS = ["mock", "gupshup", "wati", "interakt"]

    def __init__(self, provider: str = "mock", api_key: str = "demo_key"):
        self.provider = provider if provider in self.SUPPORTED_PROVIDERS else "mock"
        self.api_key = api_key
        # In-memory tracking of dispatched verifications
        self.dispatches: Dict[str, Dict[str, Any]] = {}

    def trigger_verification(
        self,
        order_id: str,
        phone_hash: str,
        order_value: float,
        customer_name: str = "Valued Customer"
    ) -> Dict[str, Any]:
        """Dispatches an interactive confirmation template via WhatsApp."""
        dispatch_id = f"wa_{uuid.uuid4().hex[:12]}"
        record = {
            "dispatch_id": dispatch_id,
            "order_id": order_id,
            "phone_hash": phone_hash,
            "order_value": order_value,
            "customer_name": customer_name,
            "provider": self.provider,
            "status": "SENT",
            "sent_at": time.time(),
            "template": "rto_cod_confirm_v2",
            "buttons": ["CONFIRM_ORDER", "CANCEL_ORDER"],
        }
        self.dispatches[order_id] = record
        return record

    def handle_customer_response(self, order_id: str, button_payload: str) -> Dict[str, Any]:
        """Processes inbound customer webhook response."""
        record = self.dispatches.get(order_id)
        if not record:
            return {"order_id": order_id, "status": "NOT_FOUND"}

        btn = button_payload.upper().strip()
        if "CONFIRM" in btn:
            new_status = "CONFIRMED"
            resulting_tag = "fp_wa_confirmed:true"
        elif "CANCEL" in btn:
            new_status = "CANCELLED_BY_CUSTOMER"
            resulting_tag = "fp_wa_cancelled:true"
        else:
            new_status = "UNKNOWN_REPLY"
            resulting_tag = "fp_wa_pending:true"

        record["status"] = new_status
        record["responded_at"] = time.time()
        record["resulting_tag"] = resulting_tag
        return record


# ---------------------------------------------------------------------------
# 3. Auto-Hold Fulfillment Flow (A-06)
# ---------------------------------------------------------------------------
class FulfillmentHoldManager:
    """
    Manages Shopify Fulfillment Orders holds on REQUIRE_DEPOSIT orders.
    Prevents warehouse pick & pack until the customer pays commitment deposit.
    """
    def __init__(self):
        self.holds: Dict[str, Dict[str, Any]] = {}

    def apply_hold(
        self,
        order_id: str,
        fulfillment_order_id: str = None,
        reason: str = "Awaiting commitment deposit (₹100)"
    ) -> Dict[str, Any]:
        f_id = fulfillment_order_id or f"fo_{order_id}"
        hold_record = {
            "order_id": order_id,
            "fulfillment_order_id": f_id,
            "status": "HELD",
            "reason": reason,
            "held_at": time.time(),
            "deposit_link": f"https://pay.flipprice.ai/deposit/{order_id}?amt=100",
        }
        self.holds[order_id] = hold_record
        return hold_record

    def release_hold(self, order_id: str, payment_ref: str = None) -> Dict[str, Any]:
        record = self.holds.get(order_id)
        if not record:
            return {"order_id": order_id, "status": "NOT_HELD"}

        record["status"] = "RELEASED"
        record["released_at"] = time.time()
        record["payment_ref"] = payment_ref or "manual_merchant_override"
        return record


# ---------------------------------------------------------------------------
# 4. Shopify Billing Manager (A-07)
# ---------------------------------------------------------------------------
class ShopifyBillingManager:
    """
    Manages merchant subscription tiers and usage metering.
    Tiers:
      - Free Bleed Audit: ₹0/mo, up to 100 orders/mo
      - Starter: ₹4,999/mo, up to 2,500 orders/mo, ₹2.00 overage
      - Growth: ₹14,999/mo, up to 10,000 orders/mo, ₹1.50 overage
    """
    PLANS = {
        "free_audit": {
            "name": "Free Bleed Audit",
            "price_inr": 0,
            "order_limit": 100,
            "overage_per_order_inr": 0.0,
            "features": ["CSV Historical Audit", "Weekly Bleed Report", "Basic Pincode View"],
        },
        "starter": {
            "name": "Starter Pricing",
            "price_inr": 4999,
            "order_limit": 2500,
            "overage_per_order_inr": 2.00,
            "features": ["Live Order Tagging", "Auto WhatsApp Confirmations", "Pincode Pricing", "Standard Support"],
        },
        "growth": {
            "name": "Growth Scaler",
            "price_inr": 14999,
            "order_limit": 10000,
            "overage_per_order_inr": 1.50,
            "features": ["All Starter Features", "Auto-Hold Deposits", "Custom Margin Config", "Dedicated Support", "SLA Guarantee"],
        },
    }

    def __init__(self):
        self.merchant_subscriptions: Dict[str, Dict[str, Any]] = {}

    def subscribe(self, merchant_id: str, plan_key: str) -> Dict[str, Any]:
        if plan_key not in self.PLANS:
            raise ValueError(f"Unknown plan: {plan_key}. Choose from {list(self.PLANS.keys())}")
        sub = {
            "merchant_id": merchant_id,
            "plan_key": plan_key,
            "plan": self.PLANS[plan_key],
            "activated_at": time.time(),
            "orders_scored_this_billing_cycle": 0,
        }
        self.merchant_subscriptions[merchant_id] = sub
        return sub

    def record_usage(self, merchant_id: str, count: int = 1) -> Dict[str, Any]:
        sub = self.merchant_subscriptions.get(merchant_id)
        if not sub:
            sub = self.subscribe(merchant_id, "free_audit")

        sub["orders_scored_this_billing_cycle"] += count
        total = sub["orders_scored_this_billing_cycle"]
        limit = sub["plan"]["order_limit"]
        overage_rate = sub["plan"]["overage_per_order_inr"]
        overage_count = max(0, total - limit)
        overage_charge = round(overage_count * overage_rate, 2)

        return {
            "merchant_id": merchant_id,
            "plan_name": sub["plan"]["name"],
            "orders_scored": total,
            "orders_limit": limit,
            "overage_count": overage_count,
            "overage_charge_inr": overage_charge,
        }
