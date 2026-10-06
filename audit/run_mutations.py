import os
import sys
import shutil
import subprocess
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKUP_DIR = os.path.join(REPO_ROOT, "audit", "_backup")

def backup_files(paths):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for p in paths:
        src = os.path.join(REPO_ROOT, p)
        dst = os.path.join(BACKUP_DIR, p.replace("/", "_").replace("\\", "_"))
        shutil.copy2(src, dst)

def restore_files(paths):
    for p in paths:
        src = os.path.join(BACKUP_DIR, p.replace("/", "_").replace("\\", "_"))
        dst = os.path.join(REPO_ROOT, p)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)

def run_pytest(test_targets=None):
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if test_targets:
        cmd.extend(test_targets)
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    failed_tests = []
    for line in res.stdout.splitlines() + res.stderr.splitlines():
        if "FAILED" in line:
            parts = line.split()
            for part in parts:
                if "::" in part:
                    failed_tests.append(part)
    return res.returncode, failed_tests, res.stdout, res.stderr

def replace_in_file(rel_path, old_str, new_str):
    full_path = os.path.join(REPO_ROOT, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_str not in content:
        raise ValueError(f"Target string not found in {rel_path}:\n{old_str[:100]}")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.replace(old_str, new_str, 1))

mutations = [
    {
        "id": 1,
        "name": "Expected-loss formula: flip sign of margin term",
        "file": "src/policy/cost_engine.py",
        "old": "- (p_success * margin)",
        "new": "+ (p_success * margin)",
        "extra_file": "src/policy/cost_engine.py",
        "extra_old": "- ((1.0 - P) * (1.0 - drop)) * margin",
        "extra_new": "+ ((1.0 - P) * (1.0 - drop)) * margin",
        "tests": ["tests/test_cost_engine.py", "tests/test_stage4.py", "tests/test_serving_validation.py"]
    },
    {
        "id": 2,
        "name": "Expected-loss formula: RTO cost 150 -> 15 (then 1500)",
        "file": "configs/cost_config.yaml",
        "old": "rto_logistics_cost: 150",
        "new": "rto_logistics_cost: 15",
        "tests": ["tests/test_cost_engine.py", "tests/test_stage4.py"]
    },
    {
        "id": 3,
        "name": "Margin rate: 20% -> 2% (then 90%)",
        "file": "configs/cost_config.yaml",
        "old": "average_margin_pct: 0.20",
        "new": "average_margin_pct: 0.02",
        "tests": ["tests/test_cost_engine.py", "tests/test_stage4.py"]
    },
    {
        "id": 4,
        "name": "Action selection: replace argmin with argmax",
        "file": "src/policy/cost_engine.py",
        "old": "best_idx = np.argmin(mat, axis=0)",
        "new": "best_idx = np.argmax(mat, axis=0)",
        "extra_file": "src/serve/scorer.py",
        "extra_old": "recommended_action = min(el_table, key=el_table.get)",
        "extra_new": "recommended_action = max(el_table, key=el_table.get)",
        "tests": ["tests/test_cost_engine.py", "tests/test_scorer.py", "tests/test_serving_validation.py"]
    },
    {
        "id": 5,
        "name": "Thresholds: change VERIFY 0.20 -> 0.90, PREPAID 0.48 -> 0.05",
        "file": "src/eval/stage5_test_reveal.py",
        "old": 'eval_binary_policy(test_cod, p_cal_cod, engine, "PREPAID_ONLY", 0.48)',
        "new": 'eval_binary_policy(test_cod, p_cal_cod, engine, "PREPAID_ONLY", 0.05)',
        "extra_file": "src/eval/stage5_test_reveal.py",
        "extra_old": 'eval_binary_policy(test_cod, p_cal_cod, engine, "VERIFY_ADDRESS", 0.20)',
        "extra_new": 'eval_binary_policy(test_cod, p_cal_cod, engine, "VERIFY_ADDRESS", 0.90)',
        "tests": ["tests/test_stage5.py"]
    },
    {
        "id": 6,
        "name": "Friction/verification cost: set to 0, then 1000",
        "file": "configs/cost_config.yaml",
        "old": "friction_cost: 2",
        "new": "friction_cost: 1000",
        "tests": ["tests/test_cost_engine.py", "tests/test_serving_validation.py"]
    },
    {
        "id": 7,
        "name": "Calibration: bypass calibrator (return raw uncalibrated scores)",
        "file": "src/serve/scorer.py",
        "old": "p = float(tree_cal.predict_proba(df)[0, 1])",
        "new": "p = float(tree_uncal.predict_proba(df)[0, 1])",
        "tests": ["tests/test_scorer.py", "tests/test_app_logic.py", "tests/test_serving_validation.py"]
    },
    {
        "id": 8,
        "name": "Features: drop historical pincode RTO (fill with constant 0.0)",
        "file": "src/serve/scorer.py",
        "old": "row_dict['historical_pincode_rto_rate'] = float(pin_info['historical_pincode_rto_rate'])",
        "new": "row_dict['historical_pincode_rto_rate'] = 0.0",
        "tests": ["tests/test_scorer.py", "tests/test_app_logic.py", "tests/test_serving_validation.py"]
    },
    {
        "id": 9,
        "name": "Split: make temporal split random (shuffle before splitting)",
        "file": "src/data/split.py",
        "old": "train_df = df[df['_ts'] <= train_cutoff].copy()",
        "new": "df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)\n    train_df = df[df['_ts'] <= train_cutoff].copy()",
        "tests": ["tests/test_stage2.py"]
    },
    {
        "id": 10,
        "name": "Labels: shuffle y_train randomly before fitting",
        "file": "src/models/logistic_baseline.py",
        "old": "pipeline.fit(X_train, y_train)",
        "new": "pipeline.fit(X_train, y_train.sample(frac=1.0, random_state=42).values)",
        "tests": ["tests/test_stage2.py"]
    },
    {
        "id": 11,
        "name": "Serving path: make single-order scoring apply different scaling than batch",
        "file": "src/serve/scorer.py",
        "old": "df = _build_model_row(clean, pin_info)",
        "new": "df = _build_model_row(clean, pin_info)\n    df['order_value'] = df['order_value'] * 10.0",
        "tests": ["tests/test_serving_validation.py"]
    },
    {
        "id": 12,
        "name": "Fallback: unseen-pincode fallback return 0 instead of global prior",
        "file": "src/serve/scorer.py",
        "old": "return dict(COLD_START_PRIOR), True",
        "new": "return {'historical_pincode_rto_rate': 0.0, 'pincode_tier': 1}, True",
        "tests": ["tests/test_serving_validation.py"]
    },
    {
        "id": 13,
        "name": "Good-customer drop-off: set drop probability to 0",
        "file": "configs/cost_config.yaml",
        "old": "success_drop_pct: 0.05",
        "new": "success_drop_pct: 0.0",
        "tests": ["tests/test_cost_engine.py", "tests/test_stage4.py", "tests/test_stage5.py"]
    },
    {
        "id": 14,
        "name": "Audit log: stop writing decision fingerprint",
        "file": "src/serve/audit.py",
        "old": '"decision_id": decision_id,',
        "new": '"decision_id": "",',
        "tests": ["tests/test_audit.py", "tests/test_serving_validation.py"]
    },
    {
        "id": 15,
        "name": "LightGBM max_depth=1 (degenerate decision stumps)",
        "file": "src/models/tree_model.py",
        "old": "'max_depth': [4, 6]",
        "new": "'max_depth': [1, 1]",
        "tests": ["tests/test_stage3.py"]
    },
    {
        "id": 16,
        "name": "Scorer applies a different category encoding",
        "file": "src/serve/scorer.py",
        "old": "'category': clean['category'],",
        "new": "'category': 'UnknownCategory',",
        "tests": ["tests/test_serving_validation.py"]
    },
    {
        "id": 17,
        "name": "Calibrator fitted on test data (data leakage)",
        "file": "src/eval/calibration.py",
        "old": 'val_cal_df = pd.read_csv("data/processed/val_cal.csv", dtype={\'pincode\': str})',
        "new": 'val_cal_df = pd.read_csv("data/processed/test.csv", dtype={\'pincode\': str})',
        "tests": ["tests/test_stage3.py"]
    },
    {
        "id": 18,
        "name": "Pincode feature computed from full dataset",
        "file": "src/serve/lookup.py",
        "old": "os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'train.csv')",
        "new": "os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'synthetic_orders.csv')",
        "tests": ["tests/test_data_realism.py"]
    }
]

def main():
    print("=" * 80)
    print(f"STARTING 18-MUTATION SABOTAGE SUITE (Total: {len(mutations)})")
    print("=" * 80 + "\n")

    results = []

    for m in mutations:
        m_id = m["id"]
        m_name = m["name"]
        files_to_touch = [m["file"]]
        if "extra_file" in m and m["extra_file"] not in files_to_touch:
            files_to_touch.append(m["extra_file"])

        backup_files(files_to_touch)
        try:
            replace_in_file(m["file"], m["old"], m["new"])
            if "extra_file" in m:
                replace_in_file(m["extra_file"], m["extra_old"], m["extra_new"])
            
            ret, failed, stdout, stderr = run_pytest(m["tests"])
            verdict = "CAUGHT" if ret != 0 or len(failed) > 0 else "SURVIVED"
            results.append({
                "id": m_id,
                "name": m_name,
                "file": m["file"],
                "failed_count": len(failed),
                "failed_tests": failed[:3],
                "verdict": verdict,
                "raw_err": stderr[:200] if stderr else stdout[:200]
            })
            print(f"[{m_id:>2}/{len(mutations)}] {verdict:<8}: {m_name}")
            if failed:
                print(f"         Failing test(s): {', '.join(failed[:2])}")
        except Exception as e:
            print(f"[{m_id:>2}/{len(mutations)}] ERROR applying mutation: {e}")
            results.append({
                "id": m_id,
                "name": m_name,
                "file": m["file"],
                "failed_count": 0,
                "failed_tests": [],
                "verdict": f"ERROR: {e}",
                "raw_err": str(e)
            })
        finally:
            restore_files(files_to_touch)

    print("\n" + "=" * 90)
    print(f"{'ID':<3} | {'Mutation Description':<50} | {'Verdict':<10} | {'First Failing Test'}")
    print("=" * 90)
    for r in results:
        f_str = r["failed_tests"][0] if r["failed_tests"] else "-"
        print(f"{r['id']:<3} | {r['name'][:50]:<50} | {r['verdict']:<10} | {f_str}")
    print("=" * 90)
    caught = sum(1 for r in results if r["verdict"] == "CAUGHT")
    total = len(results)
    pct = (caught / total) * 100.0
    print(f"\nFinal Summary: {caught}/{total} mutations caught ({pct:.1f}% caught).")
    survivors = [r for r in results if r["verdict"] != "CAUGHT"]
    if survivors:
        print("Survivors:")
        for s in survivors:
            print(f"  - [{s['id']}] {s['name']}: {s['verdict']}")
    else:
        print("All mutations successfully caught by test suite!")

if __name__ == '__main__':
    main()
