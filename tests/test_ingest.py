"""
tests/test_ingest.py — Ingestion, Adapter, and Bleed Report Unit & Integration Tests.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.ingest.shopify_csv import ShopifyOrdersImporter, hash_pii, infer_category, clean_pincode
from src.ingest.shiprocket_csv import ShiprocketOrdersImporter
from src.ingest.label_backfill import reconcile_labels
from src.ingest.merchant_calibration import MerchantCalibrator
from src.ingest.bleed_report import BleedReportGenerator

def test_hash_pii_deterministic_and_salted():
    phone = "9876543210"
    h1 = hash_pii(phone, salt="salt_a")
    h2 = hash_pii(phone, salt="salt_a")
    h3 = hash_pii(phone, salt="salt_b")

    assert h1 == h2, "Hashing must be deterministic for identical salt"
    assert h1 != h3, "Hashing must vary across distinct salts"
    assert len(h1) == 16, "Must return 16-character digest prefix"
    assert "9876543210" not in h1, "Raw PII must never be visible in hash"

def test_infer_category_keyword_matching():
    assert infer_category("Cotton Linen Men Shirt") == "Apparel"
    assert infer_category("Fast Charging Type-C Cable") == "Electronics"
    assert infer_category("Running Sneaker Shoes") == "Footwear"
    assert infer_category("Vitamin C Glow Face Serum") == "Beauty"
    assert infer_category("Floral Pattern Bedsheet with Pillows") == "Home"
    assert infer_category("Sterling Silver Ring") == "Jewelry"

    assert infer_category("Mystery Box Unknown Item") == "Apparel"  # Fallback

def test_clean_pincode():
    pin, is_cold = clean_pincode("110001")
    assert pin == "110001" and not is_cold

    pin_space, is_cold = clean_pincode(" 560 038 ")
    assert pin_space == "560038" and not is_cold

    pin_bad, is_cold = clean_pincode("INVALID_PIN")
    assert pin_bad == "110001" and is_cold

    pin_none, is_cold = clean_pincode(None)
    assert pin_none == "110001" and is_cold

def test_shopify_csv_import_end_to_end():
    # Construct synthetic raw Shopify CSV
    sample_data = {
        "Name": ["#1001", "#1001", "#1002", "#1003"],
        "Created at": ["2026-09-01 10:00:00", "2026-09-01 10:00:00", "2026-09-02 12:00:00", "2026-09-02 14:00:00"],
        "Financial Status": ["pending", "pending", "paid", "pending"],
        "Payment Method": ["Cash on Delivery (COD)", "Cash on Delivery (COD)", "Razorpay UPI", "COD"],
        "Total": [1200.0, 1200.0, 2400.0, 850.0],
        "Discount Amount": [100.0, 100.0, 0.0, 50.0],
        "Lineitem quantity": [1, 2, 1, 1],
        "Lineitem name": ["Classic Oxford Shirt", "Blue Denim Jeans", "Wireless Earphones", "Face Serum"],
        "Shipping Zip": ["110001", "110001", "560038", "400001"],
        "Shipping Method": ["Delhivery Surface", "Delhivery Surface", "Blue Dart Air", "Standard"],
        "Shipping Phone": ["9876543210", "9876543210", "9123456789", "9876543210"],
    }
    df_raw = pd.DataFrame(sample_data)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        df_raw.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        importer = ShopifyOrdersImporter()
        orders_df, meta = importer.import_csv(temp_path)

        assert len(orders_df) == 3, "Orders #1001 must be grouped into 1 order row"
        assert meta["cod_orders_count"] == 2
        assert "customer_id" in orders_df.columns
        assert not any("9876543210" in str(x) for x in orders_df["customer_id"])

        # Customer 9876543210 placed #1001 and #1003
        row_1003 = orders_df[orders_df["order_id"] == "#1003"].iloc[0]
        assert row_1003["prior_orders"] == 1, "Order #1003 must have 1 prior order"
        assert row_1003["payment_method"] == "COD"
    finally:
        os.remove(temp_path)

def test_merchant_calibrator_threshold_gate():
    calibrator = MerchantCalibrator(min_sample=300)

    # 1. Under-threshold case (n=50 < 300)
    p_small = np.random.uniform(0.1, 0.5, size=50)
    y_small = (np.random.uniform(0, 1, size=50) < p_small).astype(int)

    probs_out, report = calibrator.fit_and_recalibrate(p_small, y_small)
    assert not report["recalibrated"]
    assert "below threshold" in report["disclosure"]

    # 2. Over-threshold case (n=400 >= 300)
    rng = np.random.default_rng(42)
    p_large = rng.uniform(0.1, 0.6, size=400)
    y_large = (rng.uniform(0, 1, size=400) < (p_large * 0.8)).astype(int)

    probs_recal, report_recal = calibrator.fit_and_recalibrate(p_large, y_large)
    assert report_recal["n_samples"] == 400
    assert "threshold" in report_recal

def test_bleed_report_generator_end_to_end():
    # Create realistic test merchant orders CSV
    rng = np.random.default_rng(42)
    n = 250
    data = {
        "Name": [f"#{1000 + i}" for i in range(n)],
        "Created at": pd.date_range("2026-08-01", periods=n, freq="3h").astype(str),
        "Financial Status": rng.choice(["pending", "paid"], size=n, p=[0.65, 0.35]),
        "Total": rng.uniform(400, 3500, size=n),
        "Lineitem quantity": rng.integers(1, 4, size=n),
        "Lineitem name": rng.choice(["Cotton T-Shirt", "Wireless Headphones", "Sneakers", "Face Wash"], size=n),
        "Shipping Zip": rng.choice(["110001", "560038", "400001", "700001", "500001"], size=n),
        "Shipping Phone": [f"98765{rng.integers(10000, 99999)}" for _ in range(n)],
        "Payment Method": ["Cash on Delivery (COD)"] * n,
    }
    df_sample = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        df_sample.to_csv(f.name, index=False)
        sample_path = f.name

    try:
        gen = BleedReportGenerator()
        rep = gen.generate_report(sample_path, merchant_name="Test Pilot Brand")

        assert "headline_findings" in rep
        assert rep["headline_findings"]["cod_orders_analyzed"] > 0
        assert "measured_rupees_saved_inr" in rep["headline_findings"]
        assert len(rep["static_blacklist_vs_pricing"]["tau_grid"]) == 4

        # Verify saved files exist
        json_path = os.path.join(ROOT, "reports/bleed_reports/test_pilot_brand_bleed_report.json")
        md_path = os.path.join(ROOT, "reports/bleed_reports/test_pilot_brand_bleed_report.md")
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
    finally:
        os.remove(sample_path)
