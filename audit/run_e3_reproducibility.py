"""
audit/run_e3_reproducibility.py — Runs the reproduction chain twice with seed 42
and once with seed 999 in isolated directories using the clean virtual environment.
Diffs outputs and artifact hashes between runs.
"""
import os
import sys
import shutil
import tempfile
import subprocess
import hashlib
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_CLEAN = os.path.join(REPO_ROOT, "audit", "clean_env", "Scripts", "python.exe")
if not os.path.exists(PYTHON_CLEAN):
    PYTHON_CLEAN = sys.executable

ARTIFACTS = [
    "data/processed/pincode_rate_lookup.csv",
    "data/processed/test.csv",
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/val_cal.csv",
    "data/processed/val_rep.csv",
    "models/logistic_baseline.pkl",
    "models/tree_model.pkl",
    "models/tree_model_booster.pkl",
    "models/tree_model_calibrated.pkl",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_pipeline(work_dir, seed):
    env = dict(os.environ, PYTHONPATH=work_dir, PYTHONHASHSEED="0", RTO_SHIELD_ENV="development")
    chain = [
        [PYTHON_CLEAN, "src/data/generator.py", "--rows", "100000", "--seed", str(seed)],
        [PYTHON_CLEAN, "src/data/split.py"],
        [PYTHON_CLEAN, "src/serve/lookup.py"],
        [PYTHON_CLEAN, "src/models/logistic_baseline.py"],
        [PYTHON_CLEAN, "src/models/tree_model.py"],
        [PYTHON_CLEAN, "src/eval/calibration.py"],
        [PYTHON_CLEAN, "src/eval/stage5_test_reveal.py"],
    ]
    for cmd in chain:
        r = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Step failed ({cmd[1]}): {r.stderr[:500]}")
            
    digests = {}
    for rel in ARTIFACTS:
        p = os.path.join(work_dir, rel)
        if os.path.exists(p):
            digests[rel] = sha256_file(p)
        else:
            digests[rel] = "MISSING"
    return digests

def main():
    print("=" * 85)
    print("PHASE E3: DETERMINISTIC REPRODUCIBILITY EVALUATION")
    print("=" * 85)
    print(f"Using Clean Environment Python: {PYTHON_CLEAN}")
    
    ignore = shutil.ignore_patterns(".git", ".venv", "clean_env", "__pycache__", ".pytest_cache", "audit")
    
    # Run 1: Seed 42
    print("\n[Run 1] Executing pipeline with seed=42...")
    dir1 = tempfile.mkdtemp(prefix="repro_run1_s42_")
    shutil.copytree(REPO_ROOT, dir1, ignore=ignore, dirs_exist_ok=True)
    d1 = run_pipeline(dir1, seed=42)
    
    # Run 2: Seed 42 (identical seed)
    print("[Run 2] Executing pipeline with seed=42 (identical seed)...")
    dir2 = tempfile.mkdtemp(prefix="repro_run2_s42_")
    shutil.copytree(REPO_ROOT, dir2, ignore=ignore, dirs_exist_ok=True)
    d2 = run_pipeline(dir2, seed=42)
    
    # Run 3: Seed 999 (different seed)
    print("[Run 3] Executing pipeline with seed=999 (different seed)...")
    dir3 = tempfile.mkdtemp(prefix="repro_run3_s999_")
    shutil.copytree(REPO_ROOT, dir3, ignore=ignore, dirs_exist_ok=True)
    d3 = run_pipeline(dir3, seed=999)
    
    print("\n" + "=" * 85)
    print(f"{'Artifact':<38} | {'Run 1 (Seed 42)':<16} | {'Run 2 (Seed 42)':<16} | {'Run1==Run2?':<11} | {'Run1!=Run3?'}")
    print("=" * 85)
    
    all_identical_same_seed = True
    all_different_diff_seed = True
    
    for rel in ARTIFACTS:
        h1 = d1.get(rel, "N/A")[:16]
        h2 = d2.get(rel, "N/A")[:16]
        h3 = d3.get(rel, "N/A")[:16]
        
        same_match = (d1.get(rel) == d2.get(rel))
        diff_match = (d1.get(rel) != d3.get(rel))
        
        if not same_match:
            all_identical_same_seed = False
        if not diff_match:
            all_different_diff_seed = False
            
        print(f"{rel:<38} | {h1:<16} | {h2:<16} | {str(same_match):<11} | {str(diff_match)}")
        
    print("-" * 85)
    print(f"Same-seed run 1 vs run 2 byte-exact match: {all_identical_same_seed}")
    print(f"Diff-seed run 1 vs run 3 expected difference: {all_different_diff_seed}")
    
    # Cleanup temp directories
    shutil.rmtree(dir1, ignore_errors=True)
    shutil.rmtree(dir2, ignore_errors=True)
    shutil.rmtree(dir3, ignore_errors=True)
    
    if all_identical_same_seed:
        print("\nSUCCESS: Complete deterministic bitwise reproducibility confirmed across identical seeds!")
    else:
        print("\nFAILURE: Mismatch detected between identical-seed runs!")

if __name__ == '__main__':
    main()
