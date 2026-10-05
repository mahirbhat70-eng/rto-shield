import os
import ast
import glob
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# We classify every test
results = {}

for tf in sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py"))):
    fname = os.path.basename(tf)
    with open(tf, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=tf)
    
    file_tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # skip fixtures if decorated
            is_fixture = any(
                (isinstance(d, ast.Name) and d.id == "fixture") or
                (isinstance(d, ast.Attribute) and d.attr == "fixture")
                for d in node.decorator_list
            )
            if is_fixture:
                continue
            
            fn_src = ast.get_source_segment(src, node) or ""
            
            # Check C: File-only
            if any(term in fn_src for term in [
                "reports/", "stage5_test_results.md", "stage6_summary.json",
                "deposit_effectiveness_summary.json", "deposit_effectiveness_sensitivity.md",
                "stage6_uplift_ope_results.md", "artifact_hashes.json"
            ]) and ("with open" in fn_src or "os.path.exists" in fn_src):
                category = "C"
                reason = "Compares against static committed report/JSON artifact."
            
            # Check B: Circular
            elif (
                ("eval_multi_action" in fn_src and "eval_binary_policy" in fn_src and "test_stage4_multi_action" in node.name) or
                ("score_order_matches_pipeline" in node.name) or
                ("closed_form_baseline" in node.name and "engine.rto_logistics_cost" in fn_src and "engine.average_margin_pct" in fn_src)
            ):
                category = "B"
                reason = "Expected value derived from the project's own helpers or mathematical tautology."
            
            # Check D: Weak
            elif (
                (node.name == "test_loads_without_error") or
                ("assert 0 <= " in fn_src and "<= 1" in fn_src and "assert np.isclose" not in fn_src and "average_precision" not in fn_src and "test_cold_start" in node.name) or
                ("assert len(" in fn_src and " == len(" in fn_src and len(node.body) <= 5 and "assert proba" in fn_src) or
                ("route_order_still_works" in node.name and "in (" in fn_src)
            ):
                category = "D"
                reason = "Only checks broad interval/membership/shape without verifying precise values."
            
            # Default A: Independent
            else:
                category = "A"
                reason = "Independent constant, hand calculation, schema contract, or independent metric calculation."
                
            file_tests.append((node.name, category, reason, fn_src))
    results[fname] = file_tests

print("Summary of classifications:")
total_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
file_summary = {}

for fname, tests in results.items():
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for tname, cat, reason, _ in tests:
        counts[cat] += 1
        total_counts[cat] += 1
    file_summary[fname] = counts

for f, c in file_summary.items():
    print(f"{f:<30} | A: {c['A']:<3} | B: {c['B']:<3} | C: {c['C']:<3} | D: {c['D']:<3} | Total: {sum(c.values())}")
print("-" * 65)
print(f"{'TOTAL':<30} | A: {total_counts['A']:<3} | B: {total_counts['B']:<3} | C: {total_counts['C']:<3} | D: {total_counts['D']:<3} | Total: {sum(total_counts.values())}")
