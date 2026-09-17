"""
tests/test_generator.py
Phase 3: tests for the v1 synthetic data generator.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from src.data.generator import generate


class TestGenerator:
    @pytest.fixture(scope="class")
    def df(self):
        return generate(n_rows=500, seed=42)

    def test_row_count(self, df):
        assert len(df) == 500

    def test_reproducible_seed(self):
        d1 = generate(n_rows=100, seed=42)
        d2 = generate(n_rows=100, seed=42)
        pd.testing.assert_frame_equal(d1, d2)

    def test_different_seeds_differ(self):
        d1 = generate(n_rows=100, seed=42)
        d2 = generate(n_rows=100, seed=99)
        # order_id is sequential, check a stochastic column instead
        assert not d1["order_value"].equals(d2["order_value"])

    def test_required_columns(self, df):
        required = [
            "order_id", "order_date", "order_value", "quantity", "category",
            "discount_pct", "payment_method", "cod_charge", "customer_id",
            "account_age_days", "prior_orders", "prior_rto_count",
            "pincode", "courier_id", "pincode_tier",
            "historical_pincode_rto_rate", "orders_last_24h",
            "device_cluster_size", "rto_label",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_rto_label_binary(self, df):
        assert set(df["rto_label"].unique()).issubset({0, 1})

    def test_cod_charge_zero_for_non_cod(self, df):
        non_cod = df[df["payment_method"] != "COD"]
        assert (non_cod["cod_charge"] == 0.0).all()

    def test_order_value_positive(self, df):
        assert (df["order_value"] > 0).all()

    def test_order_value_in_range(self, df):
        assert df["order_value"].min() >= 50
        assert df["order_value"].max() <= 15000

    def test_pincode_tier_valid(self, df):
        assert df["pincode_tier"].isin([1, 2, 3]).all()

    def test_payment_method_distribution(self, df):
        cod_share = (df["payment_method"] == "COD").mean()
        assert 0.30 <= cod_share <= 0.65  # expected ~48%

    def test_prior_rto_leq_prior_orders(self, df):
        assert (df["prior_rto_count"] <= df["prior_orders"]).all()

    def test_quantity_positive(self, df):
        assert (df["quantity"] >= 1).all()

    def test_no_nulls(self, df):
        assert df.isnull().sum().sum() == 0
