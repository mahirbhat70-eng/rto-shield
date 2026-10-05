import os
import ast
import glob

tests_dir = "tests"
test_files = glob.glob(os.path.join(tests_dir, "test_*.py"))

print(f"Auditing {len(test_files)} test files...\n")

file_stats = {}
weak_tests = []
frozen_file_tests = []

for tf in sorted(test_files):
    fname = os.path.basename(tf)
    with open(tf, "r", encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code, filename=tf)
    
    test_funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    
    trivial_count = 0
    file_tests_info = []

    for fn in test_funcs:
        fn_code = ast.get_source_segment(code, fn) or ""
        asserts = [node for node in ast.walk(fn) if isinstance(node, ast.Assert)]
        
        # Check if reads frozen reports
        has_frozen_read = any(term in fn_code for term in ["reports/", "stage5_test_results.md", "stage6_summary.json", "stage4_financial_results.md", "stage3_results.md", "stage2_baseline_results.md", "artifact_hashes.json"])
        if has_frozen_read:
            frozen_file_tests.append((fname, fn.name))

        # Check assertion count & triviality
        is_weak = False
        reasons = []
        if len(asserts) == 0:
            is_weak = True
            reasons.append("0 assertions")
        else:
            # check if only trivial asserts
            trivial_asserts = 0
            for a in asserts:
                a_code = ast.get_source_segment(code, a) or ""
                if a_code in ["assert True", "assert True, True"]:
                    trivial_asserts += 1
                elif "is not None" in a_code and len(asserts) == 1:
                    trivial_asserts += 1
                elif (a_code.startswith("assert len(") and " > 0" in a_code) and len(asserts) == 1:
                    trivial_asserts += 1
            if trivial_asserts == len(asserts):
                is_weak = True
                reasons.append("only trivial assertion(s)")

        if is_weak or has_frozen_read:
            if is_weak:
                trivial_count += 1
            weak_tests.append((fname, fn.name, len(asserts), ", ".join(reasons) if reasons else "frozen file check", fn_code[:200].replace("\n", " ")))

    file_stats[fname] = {
        "total_funcs": len(test_funcs),
        "weak_count": trivial_count,
    }

print("--- WEAK / TRIVIAL TESTS ---")
for f, fn, n_assert, reason, snippet in weak_tests:
    print(f"[{f}::{fn}] (asserts={n_assert}) - Reason: {reason}")

print("\n--- TESTS READING FROZEN RESULT FILES ---")
for f, fn in frozen_file_tests:
    print(f"[{f}::{fn}]")
