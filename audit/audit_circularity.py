import os
import ast
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# We will inspect each file and classify each test
# Let's inspect test functions and their AST structure.

test_files = sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py")))

print(f"Analyzing {len(test_files)} test files for circularity and tautology...\n")

def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=filepath)
    
    records = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            fn_src = ast.get_source_segment(src, node) or ""
            records.append((node.name, fn_src))
    return records

all_tests = {}
for tf in test_files:
    fname = os.path.basename(tf)
    all_tests[fname] = analyze_file(tf)

print(f"Total collected test functions across files: {sum(len(v) for v in all_tests.values())}")
