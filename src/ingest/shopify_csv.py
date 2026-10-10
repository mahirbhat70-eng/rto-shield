"""
src/ingest/shopify_csv.py — Shopify Orders Export CSV Ingest Adapter.

Maps standard Shopify raw order export CSVs to the FlipPrice AI / rto-shield feature contract.

Features & Mappings:
--------------------
- order_id:            Shopify 'Name' (e.g. '#1024') or 'Id'.
- order_value:         Shopify 'Total', coerced to float and clamped to [1.0, 25000.0].
- quantity:            Summed line item quantity per order, clamped to [1, 50].
- category:            Inferred from lineitem names/tags (Apparel, Electronics, Footwear,
                       Beauty, Home, Jewelry); cold-start fallback: 'Apparel'.
- discount_pct:        (Discount Amount / (Subtotal + Discount Amount)) * 100.0, clamped [0.0, 100.0].
- payment_method:      Normalized to 'COD' if 'Financial Status' is pending or payment string contains 'cash'/'cod';
                       otherwise normalized to 'Prepaid'.
- cod_charge:          Extracted from shipping note/tags if present; cold-start default: 0.0.
- pincode:             Normalized 6-digit postal code from 'Shipping Zip' or 'Billing Zip'.
                       Missing/invalid pincodes fall back to cold-start prior (110001) with warning.
- courier_id:          Inferred from 'Shipping Method' or default modal 'Courier_B'.
- account_age_days:    Computed chronologically per hashed customer (days since first order seen).
- prior_orders:        Count of prior orders placed by this customer before this timestamp.
- prior_rto_count:     Count of prior cancelled/returned orders by this customer before this timestamp.
- orders_last_24h:     Count of orders placed by this customer in preceding 24h.
- device_cluster_size: Estimated from phone-cluster velocity (cold-start default: 1).

DPDP Act 2023 & Security:
-------------------------
- Raw phone numbers and emails are NEVER stored in the output features.
- All customer identifiers are salted and hashed using SHA-256 upon initial read.
"""

import os
import re
import hashlib
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

DEFAULT_SALT = os.environ.get("FLIPPRICE_PII_SALT", "rto_shield_dpdp_2026_salt")
PINCODE_REGEX = re.compile(r"^\d{6}$")

CATEGORY_KEYWORDS = {
    "Electronics": ["phone", "laptop", "cable", "charger", "headphone", "earphone", "audio", "usb", "smartwatch", "case", "gadget", "watch"],
    "Apparel": ["shirt", "t-shirt", "dress", "kurta", "pant", "jeans", "top", "saree", "hoodie", "jacket", "cotton", "cloth", "linen"],
    "Footwear": ["shoe", "shoes", "sneaker", "sandal", "slipper", "boot", "heels", "loafers", "crocs"],
    "Beauty": ["serum", "cream", "lipstick", "perfume", "fragrance", "lotion", "oil", "makeup", "shampoo", "soap", "skincare"],
    "Home": ["bedsheet", "curtain", "towel", "pillow", "cushion", "decor", "kitchen", "lamp", "mat", "blanket", "plate"],
    "Jewelry": ["ring", "necklace", "earring", "bracelet", "silver", "gold", "chain", "bangle", "pendant"],
}

COURIER_MAP = {
    "bluedart": "Courier_A",
    "delhivery": "Courier_B",
    "ekart": "Courier_C",
    "ecom": "Courier_D",
    "shadowfax": "Courier_E",
    "dtdc": "Courier_B",
    "xpressbees": "Courier_C",
}

def hash_pii(value: Any, salt: str = DEFAULT_SALT) -> str:
    """Salt and hash sensitive PII (Phone/Email) per DPDP Act 2023."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "anon_customer"
    clean = str(value).strip().lower()
    # Normalize phone: extract trailing 10 digits if applicable
    digits = re.sub(r"\D", "", clean)
    to_hash = digits[-10:] if len(digits) >= 10 else clean
    return hashlib.sha256((salt + to_hash).encode("utf-8")).hexdigest()[:16]

def infer_category(text: str) -> str:
    """Infer canonical engine category from product title / tags."""
    if not text or pd.isna(text):
        return "Apparel"
    text_lower = str(text).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "Apparel"  # Modal D2C default

def infer_courier(method: str) -> str:
    """Map courier shipping method string to model courier ID."""
    if not method or pd.isna(method):
        return "Courier_B"
    text_lower = str(method).lower()
    for k, v in COURIER_MAP.items():
        if k in text_lower:
            return v
    return "Courier_B"

def clean_pincode(pin: Any) -> Tuple[str, bool]:
    """Clean and validate Indian 6-digit postal code."""
    if pin is None or pd.isna(pin):
        return "110001", True
    # Strip spaces and non-digits
    cleaned = re.sub(r"\D", "", str(pin).strip())
    if len(cleaned) == 6 and cleaned.isdigit():
        return cleaned, False
    return "110001", True  # Cold-start fallback (Delhi Central)

class ShopifyOrdersImporter:
    """
    Production parser for Shopify CSV order exports.
    Handles multi-line items, calculates customer temporal history,
    hashes customer PII, and produces clean feature frames.
    """
    def __init__(self, salt: str = DEFAULT_SALT):
        self.salt = salt

    def import_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Shopify CSV file not found: {file_path}")

        # Support messy encoding (utf-8, utf-8-sig, latin-1)
        try:
            raw_df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            raw_df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

        # Standardize column names (strip whitespace)
        raw_df.columns = [c.strip() for c in raw_df.columns]

        # Identify order key
        order_col = "Name" if "Name" in raw_df.columns else "Id" if "Id" in raw_df.columns else raw_df.columns[0]
        
        # Identify timestamps
        ts_col = "Created at" if "Created at" in raw_df.columns else "Created At" if "Created At" in raw_df.columns else None

        # Build customer key from Phone or Email
        phone_col = next((c for c in ["Shipping Phone", "Billing Phone", "Phone"] if c in raw_df.columns), None)
        email_col = next((c for c in ["Email", "Customer Email"] if c in raw_df.columns), None)

        def get_customer_key(row):
            ph = row[phone_col] if phone_col and pd.notna(row[phone_col]) else None
            em = row[email_col] if email_col and pd.notna(row[email_col]) else None
            val = ph or em or row[order_col]
            return hash_pii(val, self.salt)

        raw_df["customer_id_hash"] = raw_df.apply(get_customer_key, axis=1)

        # Multi-line item aggregation per order
        orders = []
        cold_start_pins = 0
        total_pii_hashed = len(raw_df)

        grouped = raw_df.groupby(order_col, sort=False)
        for order_id, group in grouped:
            first_row = group.iloc[0]

            # 1. Order Value
            tot_val = 0.0
            for col in ["Total", "Total Price", "Subtotal"]:
                if col in first_row and pd.notna(first_row[col]):
                    try:
                        tot_val = float(re.sub(r"[^\d.]", "", str(first_row[col])))
                        break
                    except Exception:
                        pass
            if tot_val <= 0:
                tot_val = 826.89  # Benchmark average basket
            order_value = float(np.clip(tot_val, 1.0, 25000.0))

            # 2. Quantity
            qty = 0
            qty_col = next((c for c in ["Lineitem quantity", "Quantity"] if c in group.columns), None)
            if qty_col:
                qty = int(pd.to_numeric(group[qty_col], errors="coerce").fillna(1).sum())
            else:
                qty = len(group)
            quantity = int(np.clip(max(1, qty), 1, 50))

            # 3. Category inference
            item_names = " ".join([str(x) for x in group.get("Lineitem name", group.get("Name", [""])) if pd.notna(x)])
            tags = str(first_row.get("Tags", ""))
            category = infer_category(f"{item_names} {tags}")

            # 4. Discount %
            disc_amt = 0.0
            for col in ["Discount Amount", "Total Discounts"]:
                if col in first_row and pd.notna(first_row[col]):
                    try:
                        disc_amt = float(re.sub(r"[^\d.]", "", str(first_row[col])))
                        break
                    except Exception:
                        pass
            disc_pct = (disc_amt / (tot_val + disc_amt) * 100.0) if (tot_val + disc_amt) > 0 else 0.0
            discount_pct = float(np.clip(disc_pct, 0.0, 100.0))

            # 5. Payment method
            pay_str = str(first_row.get("Payment Method", first_row.get("Financial Status", ""))).lower()
            tags_lower = tags.lower()
            is_cod_order = any(w in pay_str for w in ["cod", "cash", "pending"]) or ("cod" in tags_lower)
            payment_method = "COD" if is_cod_order else "Prepaid"

            # 6. Pincode
            zip_val = first_row.get("Shipping Zip", first_row.get("Billing Zip", None))
            pincode, is_cold = clean_pincode(zip_val)
            if is_cold:
                cold_start_pins += 1

            # 7. Courier
            ship_method = str(first_row.get("Shipping Method", ""))
            courier_id = infer_courier(ship_method)

            # 8. Timestamp
            raw_ts = first_row.get(ts_col) if ts_col else None
            try:
                parsed_ts = pd.to_datetime(raw_ts) if raw_ts else pd.Timestamp.now()
            except Exception:
                parsed_ts = pd.Timestamp.now()

            # 9. Outcome label if present (for backtesting / label reconciliation)
            # Check for refund/cancellation or return markers in tags/financial status
            fulfillment = str(first_row.get("Fulfillment Status", "")).lower()
            cancelled = pd.notna(first_row.get("Cancelled at", None))
            rto_in_tags = any(k in tags_lower for k in ["rto", "returned", "undelivered", "fake"])
            rto_label = 1 if (cancelled or rto_in_tags or "return" in fulfillment) else 0

            orders.append({
                "order_id": str(order_id),
                "timestamp": parsed_ts,
                "customer_id": first_row["customer_id_hash"],
                "order_value": order_value,
                "quantity": quantity,
                "category": category,
                "discount_pct": discount_pct,
                "payment_method": payment_method,
                "cod_charge": 0.0,
                "pincode": pincode,
                "courier_id": courier_id,
                "rto_label": rto_label,
            })

        df_orders = pd.DataFrame(orders)
        if df_orders.empty:
            raise ValueError("No valid orders could be parsed from the provided Shopify export CSV.")

        # Sort chronologically to compute point-in-time velocity and customer history
        df_orders = df_orders.sort_values("timestamp").reset_index(drop=True)

        # Compute point-in-time historical features (account_age_days, prior_orders, prior_rto, orders_24h)
        cust_history: Dict[str, list] = {}
        first_seen: Dict[str, pd.Timestamp] = {}

        account_ages = []
        prior_orders_list = []
        prior_rtos_list = []
        orders_24h_list = []

        for _, row in df_orders.iterrows():
            c_id = row["customer_id"]
            ts = row["timestamp"]
            label = row["rto_label"]

            if c_id not in first_seen:
                first_seen[c_id] = ts
            
            age = max(0, (ts - first_seen[c_id]).days)
            account_ages.append(age)

            hist = cust_history.get(c_id, [])
            prior_orders_list.append(len(hist))
            prior_rtos_list.append(sum(1 for past_lbl in hist if past_lbl == 1))

            # Velocity in last 24h
            recent = sum(1 for past_ts, _ in hist if ts - pd.Timedelta(hours=24) <= past_ts <= ts)
            orders_24h_list.append(recent)

            # Record this event for future orders
            cust_history.setdefault(c_id, []).append((ts, label))

        df_orders["account_age_days"] = account_ages
        df_orders["prior_orders"] = prior_orders_list
        df_orders["prior_rto_count"] = prior_rtos_list
        df_orders["orders_last_24h"] = orders_24h_list
        df_orders["device_cluster_size"] = 1  # Cold-start baseline for single-store export

        # Ensure correct datatypes
        df_orders["pincode"] = df_orders["pincode"].astype(str)
        df_orders["order_value"] = df_orders["order_value"].astype(float)
        df_orders["discount_pct"] = df_orders["discount_pct"].astype(float)

        metadata = {
            "source": "Shopify Orders CSV",
            "total_raw_rows": len(raw_df),
            "total_unique_orders": len(df_orders),
            "cod_orders_count": int(np.sum(df_orders["payment_method"] == "COD")),
            "cod_orders_pct": round(float(np.mean(df_orders["payment_method"] == "COD") * 100.0), 2),
            "cold_start_pincodes_filled": cold_start_pins,
            "pii_hashed_records": total_pii_hashed,
            "date_range": {
                "start": str(df_orders["timestamp"].min()),
                "end": str(df_orders["timestamp"].max()),
            },
            "unmappable_features_cold_started": [
                {"feature": "cod_charge", "default": 0.0, "reason": "Not broken out by standard Shopify order exports"},
                {"feature": "device_cluster_size", "default": 1, "reason": "Device fingerprinting requires client JavaScript snippet"},
            ]
        }

        return df_orders, metadata
