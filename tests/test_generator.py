"""
RTO Shield — Synthetic Data Generator Tests (Stage 1)

Uses a small dynamic run (rows=1000, seed=42) via the generate() function
directly. Does NOT depend on the 100k CSV file.

No ML model tests. No scipy/sklearn/xgboost imports.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import generate, validate, EXPECTED_COLUMNS


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df_small():
    """Generate a small 1000-row dataset for testing."""
    return generate(n_rows=1000, seed=42)


# ── Schema Tests ───────────────────────────────────────────────────────

def test_column_count(df_small):
    assert df_small.shape[1] == 19, f"Expected 19 columns, got {df_small.shape[1]}"


def test_column_names_and_order(df_small):
    assert list(df_small.columns) == EXPECTED_COLUMNS, \
        f"Column mismatch: {list(df_small.columns)}"


def test_row_count(df_small):
    assert len(df_small) == 1000


def test_no_nulls(df_small):
    assert df_small.isnull().sum().sum() == 0


# ── Target ─────────────────────────────────────────────────────────────

def test_rto_label_binary(df_small):
    assert set(df_small['rto_label'].unique()).issubset({0, 1})


def test_rto_rate_in_range(df_small):
    rate = df_small['rto_label'].mean()
    assert 0.15 <= rate <= 0.30, f"RTO rate {rate:.3f} outside [0.15, 0.30]"


# ── Invariants ─────────────────────────────────────────────────────────

def test_prior_rto_leq_prior_orders(df_small):
    assert (df_small['prior_rto_count'] <= df_small['prior_orders']).all()


def test_cod_charge_zero_for_non_cod(df_small):
    non_cod = df_small[df_small['payment_method'] != 'COD']
    assert (non_cod['cod_charge'] == 0).all()


def test_cod_charge_positive_for_cod(df_small):
    cod = df_small[df_small['payment_method'] == 'COD']
    if len(cod) > 0:
        assert (cod['cod_charge'] >= 20).all() and (cod['cod_charge'] <= 100).all()


def test_pincode_string_6digit(df_small):
    assert df_small['pincode'].dtype == object
    assert (df_small['pincode'].str.len() == 6).all()


def test_historical_pincode_rto_rate_range(df_small):
    assert (df_small['historical_pincode_rto_rate'] >= 0).all()
    assert (df_small['historical_pincode_rto_rate'] <= 1).all()


def test_discount_pct_range(df_small):
    assert (df_small['discount_pct'] >= 0).all()
    assert (df_small['discount_pct'] <= 70).all()


def test_quantity_gte_1(df_small):
    assert (df_small['quantity'] >= 1).all()


def test_device_cluster_size_gte_1(df_small):
    assert (df_small['device_cluster_size'] >= 1).all()


def test_pincode_tier_values(df_small):
    assert set(df_small['pincode_tier'].unique()).issubset({1, 2, 3})


def test_timestamp_parseable_and_sorted(df_small):
    ts = pd.to_datetime(df_small['timestamp'])
    assert ts.is_monotonic_increasing


def test_order_value_positive(df_small):
    assert (df_small['order_value'] > 0).all()


def test_account_age_days_range(df_small):
    assert (df_small['account_age_days'] >= 0).all()
    assert (df_small['account_age_days'] <= 3650).all()


def test_new_customer_no_prior_orders(df_small):
    new = df_small[df_small['account_age_days'] == 0]
    if len(new) > 0:
        assert (new['prior_orders'] == 0).all()


# ── Validation function itself ─────────────────────────────────────────

def test_validate_passes(df_small):
    """The full validate() function should pass on the generated data."""
    assert validate(df_small, expected_rows=1000) is True


# ── Reproducibility ───────────────────────────────────────────────────

def test_reproducibility():
    """generate() with same seed must produce identical output."""
    df_a = generate(n_rows=500, seed=7)
    df_b = generate(n_rows=500, seed=7)
    pd.testing.assert_frame_equal(df_a, df_b)
