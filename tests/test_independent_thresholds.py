"""
tests/test_independent_thresholds.py
Direct mathematical and empirical validation of decision thresholds.
"""

import pytest
import numpy as np

def solve_closed_form_threshold(order_value, rto_cost=150.0, margin_pct=0.20,
                                 action_friction=2.0, action_rto_red=0.30, action_drop=0.05):
    M = order_value * margin_pct
    delta_denom = action_rto_red * rto_cost + action_drop * M
    delta_num = action_friction + action_drop * M
    p_star = delta_num / delta_denom
    return p_star

def test_closed_form_threshold_derivation():
    # Verify threshold scaling on low-value orders
    p_low = solve_closed_form_threshold(100.0)
    assert 0.05 <= p_low <= 0.08, f"Expected p* around 0.065 for V=100, got {p_low}"

    # Verify threshold scaling on high-value orders
    p_high = solve_closed_form_threshold(10000.0)
    assert 0.65 <= p_high <= 0.75, f"Expected p* around 0.70 for V=10000, got {p_high}"


def test_readme_test_count_matches_collected_tests():
    """Assert README.md test count matches pytest --collect-only count exactly."""
    import re
    import os
    import subprocess
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    readme_path = os.path.join(repo_root, 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()

    # Extract test count from README (e.g., "(243/243 passed)")
    match = re.search(r'\((\d+)/\d+\s+passed\)', readme_content)
    assert match, "Could not find '(X/X passed)' pattern in README.md"
    readme_count = int(match.group(1))

    # Run pytest --collect-only
    res = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'],
                         cwd=repo_root, capture_output=True, text=True)
    collect_match = re.search(r'(\d+)\s+tests? collected', res.stdout + res.stderr)
    assert collect_match, f"Could not find test collection count in pytest output:\n{res.stdout}"
    actual_count = int(collect_match.group(1))

    assert readme_count == actual_count, (
        f"README.md quotes {readme_count} tests, but pytest collected {actual_count} tests. "
        "Keep documentation and test counts strictly synchronized!"
    )

