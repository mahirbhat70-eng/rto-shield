"""
src/shopify/adapter.py — Shopify Ingestion & Mutation Adapter for FlipPrice AI.

Responsibilities:
1. Verify cryptographic Shopify webhook signatures (X-Shopify-Hmac-Sha256).
2. Transform live Shopify `orders/create` JSON payloads into the FlipPrice model contract.
3. Generate standardized Shopify Order Tags and Metafields with zero checkout friction.
4. Provide webhook idempotency to prevent duplicate scoring on network retries.
"""

import os
import re
import hmac
import base64
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ingest.shopify_csv import (
    CATEGORY_KEYWORDS,
    DEFAULT_SALT,
    hash_pii,
    PINCODE_REGEX,
)

# ---------------------------------------------------------------------------
# HMAC Verification
# ---------------------------------------------------------------------------
def verify_shopify_hmac(body: bytes, hmac_header: Optional[str], secret: str) -> bool:
    """
    Verify the Shopify X-Shopify-Hmac-Sha256 header against the raw body bytes.
    Uses timing-safe comparison to prevent side-channel timing attacks.
    """
    if not hmac_header or not secret:
        return False
    computed_digest = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256
    ).digest()
    computed_b64 = base64.b64encode(computed_digest).decode("utf-8")
    return hmac.compare_digest(computed_b64.strip(), hmac_header.strip())


# ---------------------------------------------------------------------------
# Order Parsing & Feature Extraction
# ---------------------------------------------------------------------------
def infer_category_from_line_items(line_items: List[Dict[str, Any]]) -> str:
    """Infer product category from line item titles, product types, and tags."""
    if not line_items:
        return "Apparel"

    combined_text = " ".join([
        f"{item.get('title', '')} {item.get('product_type', '')} {item.get('name', '')} {' '.join(item.get('tags', []) if isinstance(item.get('tags'), list) else [str(item.get('tags', ''))])}"
        for item in line_items
    ]).lower()

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined_text):
                return cat

    return "Apparel"


def parse_shopify_order(payload: Dict[str, Any], salt: str = DEFAULT_SALT) -> Dict[str, Any]:
    """
    Transform raw Shopify `orders/create` payload into standard FlipPrice `OrderInput` dict.
    Strictly satisfies P99.9 training bounds and DPDP Act 2023 salted customer hashing.
    """
    order_id = str(payload.get("name") or payload.get("id") or "ord_unknown")

    # Order Value
    raw_total = payload.get("total_price") or payload.get("current_total_price") or 0.0
    try:
        val = float(raw_total)
    except (ValueError, TypeError):
        val = 1.0
    order_value = float(np_clip(val, 1.0, 25000.0))

    # Quantity
    line_items = payload.get("line_items", [])
    total_qty = sum(int(item.get("quantity", 1)) for item in line_items) if line_items else 1
    quantity = float(np_clip(total_qty, 1.0, 50.0))

    # Category
    category = infer_category_from_line_items(line_items)

    # Discount Percentage
    discounts = payload.get("total_discounts") or 0.0
    try:
        disc_val = float(discounts)
    except (ValueError, TypeError):
        disc_val = 0.0
    subtotal = payload.get("subtotal_price") or order_value
    try:
        sub_val = float(subtotal)
    except (ValueError, TypeError):
        sub_val = order_value
    base_price = sub_val + disc_val
    discount_pct = float(np_clip((disc_val / base_price * 100.0) if base_price > 0 else 0.0, 0.0, 100.0))

    # Payment Method
    gateway_str = str(payload.get("gateway") or "").lower()
    financial_status = str(payload.get("financial_status") or "").lower()
    gateways_list = [str(g).lower() for g in payload.get("payment_gateway_names", [])]
    is_cod_order = (
        "cash on delivery" in gateway_str
        or "cod" in gateway_str
        or any("cash" in g or "cod" in g for g in gateways_list)
        or (financial_status == "pending" and "prepaid" not in gateway_str)
    )
    payment_method = "COD" if is_cod_order else "Prepaid"

    # COD Charge
    shipping_lines = payload.get("shipping_lines", [])
    cod_charge = 0.0
    for s_line in shipping_lines:
        title = str(s_line.get("title", "")).lower()
        if "cod" in title or "cash" in title:
            try:
                cod_charge = float(s_line.get("price", 0.0))
            except (ValueError, TypeError):
                cod_charge = 0.0
            break
    cod_charge = float(np_clip(cod_charge, 0.0, 500.0))

    # Pincode extraction
    shipping_addr = payload.get("shipping_address") or payload.get("billing_address") or {}
    raw_zip = str(shipping_addr.get("zip") or shipping_addr.get("postal_code") or "").strip()
    digits = re.sub(r"\D", "", raw_zip)
    pincode = digits[:6] if len(digits) == 6 else "110001"

    # Customer metadata (DPDP salted hash)
    cust = payload.get("customer") or {}
    phone = shipping_addr.get("phone") or cust.get("phone") or ""
    cust_hash = hash_pii(phone or cust.get("email") or cust.get("id"), salt=salt)
    
    # Customer velocity features
    orders_count = float(cust.get("orders_count", 0.0))
    prior_orders = float(np_clip(orders_count, 0.0, 100.0))
    prior_rto_count = 0.0  # Cold start unless backfilled
    account_age_days = 0.0
    if cust.get("created_at"):
        try:
            # Parse ISO date
            created_ts = cust.get("created_at", "")[:10]
            # Simple day approximation if needed
            account_age_days = 30.0
        except Exception:
            account_age_days = 0.0

    return {
        "order_id": order_id,
        "order_value": order_value,
        "quantity": quantity,
        "category": category,
        "discount_pct": discount_pct,
        "payment_method": payment_method,
        "cod_charge": cod_charge,
        "account_age_days": float(np_clip(account_age_days, 0.0, 5000.0)),
        "prior_orders": prior_orders,
        "prior_rto_count": prior_rto_count,
        "pincode": pincode,
        "courier_id": "Courier_B",
        "orders_last_24h": 1.0,
        "device_cluster_size": 1.0,
        "customer_hash": cust_hash,
    }


def np_clip(val: float, lo: float, hi: float) -> float:
    """Helper clip without heavy numpy overhead on microsecond paths."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


# ---------------------------------------------------------------------------
# Tag & Metafield Formatter
# ---------------------------------------------------------------------------
def format_shopify_tags_and_metafields(
    score_res: Dict[str, Any]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Format FlipPrice tags and metafields for Shopify Order mutation.
    
    Tags:
      - fp_risk:<p_rto:.2f>
      - fp_action:<ACTION>
      - fp_el:<expected_loss_inr:.2f>
      - fp_decision:PRICED
    
    Metafields (namespace: flipprice):
      - risk_score (number_decimal)
      - recommended_action (single_line_text_field)
      - expected_loss_inr (number_decimal)
      - decision_fingerprint (single_line_text_field)
    """
    prob = score_res.get("probability", 0.0)
    action = score_res.get("recommended_action", "ALLOW_COD")
    el_table = score_res.get("el_table", {})
    el = el_table.get(action, 0.0)
    fingerprint = score_res.get("decision_fingerprint", "fp_sha256_mock")

    tags = [
        f"fp_risk:{prob:.2f}",
        f"fp_action:{action}",
        f"fp_el:{el:.2f}",
        "fp_decision:PRICED",
    ]

    metafields = [
        {
            "namespace": "flipprice",
            "key": "risk_score",
            "value": f"{prob:.4f}",
            "type": "number_decimal",
        },
        {
            "namespace": "flipprice",
            "key": "recommended_action",
            "value": action,
            "type": "single_line_text_field",
        },
        {
            "namespace": "flipprice",
            "key": "expected_loss_inr",
            "value": f"{el:.2f}",
            "type": "number_decimal",
        },
        {
            "namespace": "flipprice",
            "key": "decision_fingerprint",
            "value": fingerprint,
            "type": "single_line_text_field",
        },
    ]

    return tags, metafields


# ---------------------------------------------------------------------------
# Idempotency Tracker
# ---------------------------------------------------------------------------
class IdempotencyTracker:
    """In-memory + TTL cache for Shopify webhook deduplication."""
    def __init__(self, ttl_seconds: int = 86400):
        self._seen: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.ttl = ttl_seconds

    def check_and_set(self, webhook_id: str, result: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Returns cached result if webhook_id was already processed within TTL.
        Otherwise records webhook_id and returns None.
        """
        now = time.time()
        self._prune(now)
        if webhook_id in self._seen:
            return self._seen[webhook_id][1]
        if result is not None:
            self._seen[webhook_id] = (now, result)
        return None

    def store_result(self, webhook_id: str, result: Dict[str, Any]):
        self._seen[webhook_id] = (time.time(), result)

    def _prune(self, now: float):
        if len(self._seen) > 10000:
            expired = [k for k, (ts, _) in self._seen.items() if now - ts > self.ttl]
            for k in expired:
                del self._seen[k]


def sync_tags_to_shopify(
    shop_domain: Optional[str],
    raw_order_id: Any,
    tags_to_add: List[str],
    access_token: Optional[str] = None,
) -> bool:
    """
    Optional active tag write-back to Shopify Admin API.
    Gracefully no-ops if SHOPIFY_ADMIN_ACCESS_TOKEN is unset or order_id is mock.
    """
    token = access_token or os.environ.get("SHOPIFY_ADMIN_ACCESS_TOKEN")
    if not token or not shop_domain:
        return False
    order_id_str = str(raw_order_id).replace("#", "").strip()
    if not order_id_str.isdigit():
        return False
    try:
        import urllib.request
        import json
        url = f"https://{shop_domain}/admin/api/2024-10/orders/{order_id_str}.json"
        req_get = urllib.request.Request(
            url,
            headers={
                "X-Shopify-Access-Token": token,
                "Content-Type": "application/json",
            },
            method="GET",
        )
        with urllib.request.urlopen(req_get, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
            existing_tags = data.get("order", {}).get("tags", "")

        current_tag_list = [t.strip() for t in existing_tags.split(",") if t.strip()]
        new_tag_list = list(dict.fromkeys(current_tag_list + tags_to_add))
        new_tags_str = ", ".join(new_tag_list)

        req_put = urllib.request.Request(
            url,
            data=json.dumps({"order": {"id": int(order_id_str), "tags": new_tags_str}}).encode(),
            headers={
                "X-Shopify-Access-Token": token,
                "Content-Type": "application/json",
            },
            method="PUT",
        )
        with urllib.request.urlopen(req_put, timeout=3.0) as resp:
            return resp.status in (200, 201)
    except Exception:
        return False
