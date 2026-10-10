"""
src/ingest/label_backfill.py — Multi-Source Label Reconciliation & Backfill Service.

Joins order details (e.g. from Shopify) with delivery outcomes (e.g. from Shiprocket / courier logs)
to produce a high-fidelity ground truth labeled cohort for merchant backtesting.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def reconcile_labels(orders_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reconciles merchant orders with logistics fulfillment outcomes.
    
    Parameters:
        orders_df: DataFrame parsed from Shopify (contains order_id, features).
        outcomes_df: DataFrame parsed from Shiprocket/courier logs (contains order_id, rto_label).
        
    Returns:
        joined_df: Orders with verified labels.
        metrics: Honest coverage and class balance metrics.
    """
    # Normalize order_id keys (strip #, leading zeroes, whitespace)
    def clean_id(s):
        return str(s).strip().replace("#", "").lower()

    orders_df = orders_df.copy()
    outcomes_df = outcomes_df.copy()

    orders_df["join_key"] = orders_df["order_id"].apply(clean_id)
    outcomes_df["join_key"] = outcomes_df["order_id"].apply(clean_id)

    # Outcome deduplication (keep latest status per order)
    outcomes_unique = outcomes_df.drop_duplicates(subset="join_key", keep="last")

    # Merge labels onto orders
    merged = pd.merge(
        orders_df,
        outcomes_unique[["join_key", "rto_label", "raw_status"]],
        on="join_key",
        how="left",
        suffixes=("_orders", "_logistics")
    )

    # If logistics has label, use it; else fallback to order tag label
    if "rto_label_logistics" in merged.columns:
        final_label = merged["rto_label_logistics"].combine_first(merged.get("rto_label_orders", pd.Series(0, index=merged.index)))
        has_logistics_match = merged["rto_label_logistics"].notna()
    else:
        final_label = merged.get("rto_label_orders", pd.Series(0, index=merged.index))
        has_logistics_match = pd.Series(False, index=merged.index)

    merged["rto_label"] = final_label.fillna(0).astype(int)
    merged.drop(columns=["join_key"], errors="ignore", inplace=True)

    n_total = len(merged)
    n_matched = int(has_logistics_match.sum())
    coverage_pct = round(float(n_matched / n_total * 100.0), 2) if n_total > 0 else 0.0

    cod_mask = (merged["payment_method"] == "COD").values
    cod_orders = merged[cod_mask]
    n_cod = len(cod_orders)
    cod_rto_rate = round(float(cod_orders["rto_label"].mean() * 100.0), 2) if n_cod > 0 else 0.0

    metrics = {
        "total_orders": n_total,
        "logistics_matched_orders": n_matched,
        "label_coverage_pct": coverage_pct,
        "unmatched_orders": n_total - n_matched,
        "total_cod_orders": n_cod,
        "cod_rto_rate_pct": cod_rto_rate,
        "overall_rto_rate_pct": round(float(merged["rto_label"].mean() * 100.0), 2),
        "class_balance": {
            "delivered_count": int(np.sum(merged["rto_label"] == 0)),
            "rto_count": int(np.sum(merged["rto_label"] == 1)),
        }
    }

    return merged, metrics
