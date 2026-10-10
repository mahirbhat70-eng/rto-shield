"""
src/ingest/shiprocket_csv.py — Shiprocket / NDR Export Ingest Adapter.

Parses Shiprocket shipping, NDR, and delivery reports to extract order features
and definitive final delivery vs RTO outcomes for backtesting and decisioning.

Supported Status Outcomes:
--------------------------
- Delivered:   rto_label = 0 (successful fulfillment)
- RTO / NDR:   rto_label = 1 (RTO Initiated, RTO Delivered, Customer Refusal, Address Issue)
"""

import os
import re
import hashlib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.ingest.shopify_csv import hash_pii, infer_category, infer_courier, clean_pincode, DEFAULT_SALT

RTO_STATUSES = {
    "rto initiated", "rto delivered", "rto acknowledged", "return to origin",
    "undelivered", "customer refused", "rto", "rto in transit", "cancelled"
}

DELIVERED_STATUSES = {
    "delivered", "fulfilled", "complete", "completed"
}

class ShiprocketOrdersImporter:
    """
    Adapter for Shiprocket / NDR shipments exports.
    """
    def __init__(self, salt: str = DEFAULT_SALT):
        self.salt = salt

    def import_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Shiprocket CSV file not found: {file_path}")

        try:
            raw_df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            raw_df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

        raw_df.columns = [c.strip() for c in raw_df.columns]

        # Identify Order ID column
        order_col = next((c for c in ["Channel Order Id", "Order Id", "Order ID", "AWB", "Shipment ID"] if c in raw_df.columns), raw_df.columns[0])

        # Status column
        status_col = next((c for c in ["Status", "Shipment Status", "Delivery Status", "Current Status"] if c in raw_df.columns), None)

        # Phone / Email
        phone_col = next((c for c in ["Customer Phone", "Phone", "Mobile", "Contact"] if c in raw_df.columns), None)
        email_col = next((c for c in ["Customer Email", "Email"] if c in raw_df.columns), None)

        # Value column
        val_col = next((c for c in ["Order Value", "Invoice Value", "Total Amount", "Amount", "Collectable Amount"] if c in raw_df.columns), None)

        # Pincode column
        pin_col = next((c for c in ["Delivery Pincode", "Pincode", "Destination Pincode", "Pin Code"] if c in raw_df.columns), None)

        # Courier column
        courier_col = next((c for c in ["Courier Company", "Courier", "Courier Name"] if c in raw_df.columns), None)

        # Payment method column
        pay_col = next((c for c in ["Payment Method", "Payment Type", "Order Type"] if c in raw_df.columns), None)

        # Timestamp column
        ts_col = next((c for c in ["Order Date", "Pickup Date", "Created Date", "Shipment Date"] if c in raw_df.columns), None)

        orders = []
        cold_start_pins = 0
        labeled_orders = 0

        grouped = raw_df.groupby(order_col, sort=False)
        for order_id, group in grouped:
            first_row = group.iloc[0]

            # 1. Outcome Status (RTO vs Delivered)
            status_str = str(first_row.get(status_col, "")).strip().lower() if status_col else ""
            if any(s in status_str for s in RTO_STATUSES):
                rto_label = 1
                labeled_orders += 1
            elif any(s in status_str for s in DELIVERED_STATUSES):
                rto_label = 0
                labeled_orders += 1
            else:
                rto_label = 0  # Default unconfirmed

            # 2. Order Value
            tot_val = 0.0
            if val_col and pd.notna(first_row[val_col]):
                try:
                    tot_val = float(re.sub(r"[^\d.]", "", str(first_row[val_col])))
                except Exception:
                    tot_val = 826.89
            order_value = float(np.clip(tot_val if tot_val > 0 else 826.89, 1.0, 25000.0))

            # 3. Pincode
            pincode, is_cold = clean_pincode(first_row.get(pin_col))
            if is_cold:
                cold_start_pins += 1

            # 4. Courier
            courier_id = infer_courier(str(first_row.get(courier_col, "")))

            # 5. Payment method
            pay_str = str(first_row.get(pay_col, "")).lower()
            payment_method = "COD" if ("cod" in pay_str or "cash" in pay_str) else "Prepaid"

            # 6. Customer ID hash
            cust_val = first_row.get(phone_col) or first_row.get(email_col) or order_id
            cust_hash = hash_pii(cust_val, self.salt)

            # 7. Timestamp
            raw_ts = first_row.get(ts_col) if ts_col else None
            try:
                parsed_ts = pd.to_datetime(raw_ts) if raw_ts else pd.Timestamp.now()
            except Exception:
                parsed_ts = pd.Timestamp.now()

            orders.append({
                "order_id": str(order_id),
                "timestamp": parsed_ts,
                "customer_id": cust_hash,
                "order_value": order_value,
                "quantity": 1,
                "category": "Apparel",  # Default modal D2C
                "discount_pct": 0.0,
                "payment_method": payment_method,
                "cod_charge": 0.0,
                "pincode": pincode,
                "courier_id": courier_id,
                "rto_label": rto_label,
                "raw_status": status_str,
            })

        df_orders = pd.DataFrame(orders).sort_values("timestamp").reset_index(drop=True)

        # Historical temporal features
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

            account_ages.append(max(0, (ts - first_seen[c_id]).days))
            hist = cust_history.get(c_id, [])
            prior_orders_list.append(len(hist))
            prior_rtos_list.append(sum(1 for past_lbl in hist if past_lbl == 1))
            orders_24h_list.append(sum(1 for past_ts, _ in hist if ts - pd.Timedelta(hours=24) <= past_ts <= ts))

            cust_history.setdefault(c_id, []).append((ts, label))

        df_orders["account_age_days"] = account_ages
        df_orders["prior_orders"] = prior_orders_list
        df_orders["prior_rto_count"] = prior_rtos_list
        df_orders["orders_last_24h"] = orders_24h_list
        df_orders["device_cluster_size"] = 1

        metadata = {
            "source": "Shiprocket / NDR Export CSV",
            "total_raw_rows": len(raw_df),
            "total_unique_orders": len(df_orders),
            "cod_orders_count": int(np.sum(df_orders["payment_method"] == "COD")),
            "cod_orders_pct": round(float(np.mean(df_orders["payment_method"] == "COD") * 100.0), 2),
            "labeled_orders_coverage_pct": round(float(labeled_orders / len(df_orders) * 100.0), 2),
            "observed_rto_rate_pct": round(float(df_orders["rto_label"].mean() * 100.0), 2),
            "cold_start_pincodes_filled": cold_start_pins,
            "pii_hashed_records": len(df_orders),
        }

        return df_orders, metadata
