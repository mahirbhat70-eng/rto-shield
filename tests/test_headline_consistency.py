"""
tests/test_headline_consistency.py — Verify all docs and reports match headline_numbers.json.
"""

import json
import os
import re
import subprocess
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADLINE_PATH = os.path.join(REPO_ROOT, "reports/headline_numbers.json")


@pytest.fixture(scope="module")
def headline_data():
    assert os.path.exists(HEADLINE_PATH), f"Missing {HEADLINE_PATH}. Run scripts/generate_headline_numbers.py first."
    with open(HEADLINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_readme_test_count_equals_pytest_collected():
    """Assert README badge and text test counts match pytest --collect-only exactly."""
    readme_path = os.path.join(REPO_ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract all test counts mentioned in README
    match_pass = re.search(r"\((\d+)/\d+\s+passed\)", content)
    assert match_pass, "Could not find '(X/X passed)' in README.md"
    readme_pass_count = int(match_pass.group(1))

    match_badge = re.search(r"Tests-(\d+)%20Passing", content)
    assert match_badge, "Could not find 'Tests-X%20Passing' badge in README.md"
    readme_badge_count = int(match_badge.group(1))

    match_summary = re.search(r"\((\d+)\s+passing\s+tests\)", content)
    assert match_summary, "Could not find '(X passing tests)' in README.md"
    readme_summary_count = int(match_summary.group(1))

    # Run pytest --collect-only
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    collect_match = re.search(r"(\d+)\s+tests? collected", res.stdout + res.stderr)
    assert collect_match, f"Could not parse test collection from pytest:\n{res.stdout}"
    collected_count = int(collect_match.group(1))

    assert readme_pass_count == collected_count, (
        f"README '(X/X passed)' is {readme_pass_count}, but pytest collected {collected_count}"
    )
    assert readme_badge_count == collected_count, (
        f"README badge is {readme_badge_count}, but pytest collected {collected_count}"
    )
    assert readme_summary_count == collected_count, (
        f"README summary count is {readme_summary_count}, but pytest collected {collected_count}"
    )


def test_readme_headline_values_match(headline_data):
    """Assert README headline numbers match headline_numbers.json exactly."""
    readme_path = os.path.join(REPO_ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Expected savings (71,741)
    exp_sav = int(round(headline_data["expected_savings_inr"]))
    exp_pattern = rf"₹?{exp_sav:,}"
    assert re.search(exp_pattern, content), f"Expected savings {exp_pattern} missing in README"

    # Realized savings (69,786)
    real_sav = int(round(headline_data["realized_savings_inr"]))
    real_pattern = rf"₹?{real_sav:,}"
    assert re.search(real_pattern, content), f"Realized savings {real_pattern} missing in README"

    # Uplift % (13.1%)
    uplift = f"{headline_data['uplift_pct']:.1f}%"
    assert uplift in content, f"Uplift {uplift} missing in README"

    # PR-AUC (0.3313)
    pr_auc = f"{headline_data['models']['lgbm_isotonic']['pr_auc']:.4f}"
    assert pr_auc in content, f"PR-AUC {pr_auc} missing in README"

    # RTO volume reduction (46.9%)
    rto_vol = f"{headline_data['rtos_prevented_pct']:.1f}%"
    assert rto_vol in content, f"RTO reduction {rto_vol} missing in README"

    # COD RTO drop: 28.3% -> 19.9%
    rto_pre = f"{headline_data['cod_base_rto_pct']:.1f}%"
    rto_post = f"{headline_data['effective_post_policy_rto_pct']:.1f}%"
    assert re.search(rf"{rto_pre}\s*→\s*{rto_post}", content), f"Rate drop {rto_pre} -> {rto_post} missing in README"


def test_claim_matrix_values_match(headline_data):
    """Assert claim-matrix.md numbers match headline_numbers.json exactly."""
    matrix_path = os.path.join(REPO_ROOT, "claim-matrix.md")
    with open(matrix_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Uplift
    assert f"{headline_data['uplift_pct']:.1f}%" in content

    # RTO volume reduction
    assert f"{headline_data['rtos_prevented_pct']:.1f}%" in content

    # Multi-action expected savings
    exp_sav = int(round(headline_data["expected_savings_inr"]))
    assert f"₹{exp_sav:,}" in content

    # Realized savings
    real_sav = int(round(headline_data["realized_savings_inr"]))
    assert f"₹{real_sav:,}" in content

    # PR-AUC
    assert f"{headline_data['models']['lgbm_isotonic']['pr_auc']:.4f}" in content

    # COD-subset RTO rate
    assert f"{headline_data['cod_base_rto_pct']:.2f}%" in content

    # RTOs prevented
    rto_prev = int(round(headline_data["expected_rtos_prevented"]))
    assert str(rto_prev) in content

    # Good customer drops
    drops = int(round(headline_data["good_customer_drops"]))
    assert str(drops) in content

    # Friction spend
    friction = int(round(headline_data["friction_spend_inr"]))
    assert f"₹{friction:,}" in content


def test_stage5_report_values_match(headline_data):
    """Assert reports/stage5_test_results.md matches headline_numbers.json."""
    stage5_path = os.path.join(REPO_ROOT, "reports/stage5_test_results.md")
    with open(stage5_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert f"{headline_data['expected_savings_inr']:,.2f}" in content
    assert f"{headline_data['realized_savings_inr']:,.2f}" in content
    assert f"{headline_data['models']['lgbm_isotonic']['pr_auc']:.4f}" in content
    assert f"{headline_data['expected_rtos_prevented']:.2f}" in content
    assert f"{headline_data['good_customer_drops']:.2f}" in content


def test_judge_qa_values_match(headline_data):
    """Assert docs/JUDGE_QA.md matches headline_numbers.json."""
    qa_path = os.path.join(REPO_ROOT, "docs/JUDGE_QA.md")
    with open(qa_path, "r", encoding="utf-8") as f:
        content = f.read()

    exp_sav = int(round(headline_data["expected_savings_inr"]))
    real_sav = int(round(headline_data["realized_savings_inr"]))
    assert f"₹{exp_sav:,}" in content
    assert f"₹{real_sav:,}" in content
