"""
test_sensitivity.py — Executable claims for the deposit-effectiveness
sensitivity analysis (src/eval/deposit_effectiveness_sensitivity.py).

The frozen headline ("2.0x savings") is linear in the unvalidated DEPOSIT
constant rto_reduction_pct=0.80. These tests pin the honest disclosure:
the anchor reproduces the frozen numbers, and the assumption-bound range
is what the README now quotes alongside the point estimate.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SUMMARY_PATH = "reports/deposit_effectiveness_summary.json"
REPORT_PATH = "reports/deposit_effectiveness_sensitivity.md"
PLOT_PATH = "reports/stage4/deposit_effectiveness_heatmap.png"


@pytest.fixture(scope="module")
def summary():
    if not os.path.exists(SUMMARY_PATH):
        pytest.skip("run src/eval/deposit_effectiveness_sensitivity.py to generate artifacts")
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_artifacts_exist():
    assert os.path.exists(REPORT_PATH), "sensitivity report missing"
    assert os.path.exists(PLOT_PATH), "heatmap missing"


def test_anchor_reproduces_frozen_claim(summary):
    # At the claimed constants the script must reproduce the frozen Stage-5
    # numbers exactly: primary 71741, best-binary 35919, ratio ~1.997.
    assert summary["anchor_primary_savings"] == pytest.approx(71741, abs=2)
    assert summary["best_binary_savings"] == pytest.approx(35919, abs=2)
    assert summary["anchor_ratio"] == pytest.approx(1.997, abs=0.01)


def test_assumption_bound_range_is_disclosed(summary):
    lo, hi = summary["ratio_range_at_claimed_d"]
    # The ratio collapses toward ~1.2x at low deposit effectiveness and
    # reaches ~2.4x at high effectiveness. The README quotes this range.
    assert 1.15 <= lo <= 1.35, f"lower bound {lo} outside expected band"
    assert 2.25 <= hi <= 2.55, f"upper bound {hi} outside expected band"
    assert summary["ratio_at_r_0.50"] == pytest.approx(1.28, abs=0.05)


def test_fee_correction_reduces_savings(summary):
    # Including COD-fee revenue on delivered orders must REDUCE savings
    # (interventions become relatively less attractive).
    assert summary["fee_corrected_primary_savings"] < 0.65 * summary["anchor_primary_savings"]


def test_report_states_the_caveat():
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = f.read()
    assert "unvalidated" in report.lower()
    assert "Menu degeneration" in report or "degeneration" in report
    assert "linear" in report.lower()
