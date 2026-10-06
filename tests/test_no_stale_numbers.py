"""
tests/test_no_stale_numbers.py — Programmatic guard against stale reports and parameter claims.

Enforces:
1. Every JSON report under reports/ that records config_sha256 matches current configs/cost_config.yaml SHA-256.
2. Documentation blocks matching generated narrative match current config parameters.
"""

import glob
import hashlib
import json
import os
import re
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_current_config():
    path = os.path.join(ROOT, "configs", "cost_config.yaml")
    with open(path, "rb") as f:
        raw = f.read()
    h = hashlib.sha256(raw).hexdigest()
    cfg = yaml.safe_load(raw)
    return cfg, h


def test_reports_config_sha256_freshness():
    """Fail if any report JSON has a stale config_sha256 differing from configs/cost_config.yaml."""
    cfg, current_sha = get_current_config()
    json_paths = glob.glob(os.path.join(ROOT, "reports", "*.json"))
    assert len(json_paths) > 0, "No reports/*.json found — run scripts/regenerate_all.py first"

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "config_sha256" in data:
            assert data["config_sha256"] == current_sha, (
                f"Stale config_sha256 in {os.path.basename(jp)}: report has {data['config_sha256']}, "
                f"current config has {current_sha}. Re-run scripts/regenerate_all.py!"
            )


def test_narrative_text_consistent_with_config():
    """Verify generated narrative matches current config values exactly."""
    cfg, current_sha = get_current_config()
    from scripts.generate_narrative_blocks import generate_cost_narrative
    content = generate_cost_narrative(cfg, current_sha)

    # RTO cost
    assert f"₹{cfg['rto_logistics_cost']}" in content
    # Margin
    assert f"{int(cfg['average_margin_pct'] * 100)}%" in content
    # Verify friction
    assert f"`VERIFY_ADDRESS`: Friction ₹{cfg['interventions']['VERIFY_ADDRESS']['friction_cost']:.2f}" in content
    # Deposit friction
    assert f"`REQUIRE_DEPOSIT`: Friction ₹{cfg['interventions']['REQUIRE_DEPOSIT']['friction_cost']:.2f}" in content
    # Deposit drop
    assert f"{int(cfg['interventions']['REQUIRE_DEPOSIT']['success_drop_pct'] * 100)}%" in content
    # Deposit reduction
    assert f"{int(cfg['interventions']['REQUIRE_DEPOSIT']['rto_reduction_pct'] * 100)}%" in content


def test_breakeven_sweep_at_natural_rate_reproduces_headline():
    """Assert that evaluating the break-even sweep at the natural COD RTO rate reproduces headline realized savings."""
    import numpy as np
    hl_path = os.path.join(ROOT, "reports", "headline_numbers.json")
    be_path = os.path.join(ROOT, "reports", "breakeven.json")
    if not (os.path.exists(hl_path) and os.path.exists(be_path)):
        pytest.skip("Reports not yet generated — run scripts/regenerate_all.py")

    with open(hl_path, "r", encoding="utf-8") as f:
        hl = json.load(f)
    with open(be_path, "r", encoding="utf-8") as f:
        be = json.load(f)

    natural_rate = hl["cod_base_rto_rate"]
    headline_sav = hl["realized_savings_inr"]

    sweep = be.get("reweighting_sweep", [])
    rates = [pt["rate"] for pt in sweep]
    savs = [pt["mean_savings_inr"] for pt in sweep]

    interp_sav = float(np.interp(natural_rate, rates, savs))
    assert abs(interp_sav - headline_sav) < 50.0, (
        f"Break-even sweep at natural rate ({natural_rate:.4f}) produced Rs {interp_sav:.2f}, "
        f"differing from headline Rs {headline_sav:.2f}"
    )
