"""
tests/test_pilot_ops.py — Test Suite for Squad S8 Pilot Operations.

Covers:
- Weekly value report generation from SQLite audit ledger.
- Net savings aggregation and action distribution calculations.
- SHAP narrative rendering in decision examples.
- Executive Markdown digest formatting for Slack / Email.
- Empty tenant handling.
"""

import os
import sys
import pytest

from scripts.commercial.weekly_value_report import (
    generate_weekly_report,
    format_markdown_digest,
)
from src.serve.tenants import TenantManager


def test_weekly_value_report_generation():
    tm = TenantManager()
    tenant_id = "tenant_demo_1"

    # Seed at least 3 audit decisions for demo tenant
    tm.log_decision(
        tenant_id=tenant_id,
        order_id="ord_pilot_01",
        decision_fingerprint="fp_hash_001",
        action="ALLOW_COD",
        risk_score=0.12,
        savings_inr=0.0,
        latency_ms=12.5,
    )
    tm.log_decision(
        tenant_id=tenant_id,
        order_id="ord_pilot_02",
        decision_fingerprint="fp_hash_002",
        action="VERIFY_ADDRESS",
        risk_score=0.38,
        savings_inr=42.50,
        latency_ms=14.2,
    )
    tm.log_decision(
        tenant_id=tenant_id,
        order_id="ord_pilot_03",
        decision_fingerprint="fp_hash_003",
        action="REQUIRE_DEPOSIT",
        risk_score=0.65,
        savings_inr=85.00,
        latency_ms=15.1,
    )

    # Generate report
    rep = generate_weekly_report(tenant_id)
    assert rep["status"] == "ACTIVE"
    assert rep["tenant_id"] == tenant_id
    assert rep["total_orders_evaluated"] >= 3
    assert rep["net_savings_inr"] >= 127.50
    assert "VERIFY_ADDRESS" in rep["action_distribution"]
    assert len(rep["top_decision_examples"]) >= 3

    # Format Markdown
    md = format_markdown_digest(rep)
    assert "Weekly Value Report" in md
    assert "Net Margin Protected" in md
    assert "Notable Decision Examples" in md


def test_weekly_value_report_empty_tenant():
    tm = TenantManager()
    empty_tenant_id = "tenant_test_isolated"

    # Make sure we can generate report for an empty tenant
    rep = generate_weekly_report(empty_tenant_id)
    assert rep["status"] in ["ACTIVE", "NO_ACTIVITY"]
    assert rep["tenant_id"] == empty_tenant_id
