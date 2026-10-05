import os
import ast
import glob

test_files = glob.glob(os.path.join("tests", "test_*.py"))

print("Investigating tests with 0 AST assert statements...\n")

for tf in sorted(test_files):
    fname = os.path.basename(tf)
    with open(tf, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code, filename=tf)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
            if len(asserts) == 0:
                fn_src = ast.get_source_segment(code, node) or ""
                uses_pytest_raises = "pytest.raises" in fn_src
                uses_np_testing = "assert_" in fn_src or "testing." in fn_src
                is_fixture = any(d.id == "fixture" if isinstance(d, ast.Name) else False for d in node.decorator_list)
                print(f"[{fname}::{node.name}]")
                print(f"  pytest.raises: {uses_pytest_raises} | np/pd testing: {uses_np_testing} | is_fixture: {is_fixture}")
                if not (uses_pytest_raises or uses_np_testing):
                    print("  --> SUSPICIOUS / NO CHECK:")
                    for line in fn_src.splitlines()[:6]:
                        print("     ", line)
