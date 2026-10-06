"""
audit/prompt12/repro_check.py — Prompt 12: reproducibility / build integrity.

Rebuilds every pinned artifact from scratch inside an ISOLATED COPY of the repo
(the working tree is never modified), N times, and reports:
  * run-vs-run determinism  (same seed => same bytes?)
  * run-vs-committed parity (does a rebuild reproduce models/artifact_hashes.json?)

Usage:
  python audit/prompt12/repro_check.py --runs 2 [--out audit/prompt12/result_local.json]
Exit code 0 only if every run matches every other run AND the committed hashes.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHAIN = [
    ["src/data/generator.py"],
    ["src/data/generator_v2.py"],
    ["src/data/split.py"],
    ["src/serve/lookup.py"],
    ["src/models/logistic_baseline.py"],
    ["src/models/tree_model.py"],
    ["src/eval/calibration.py"],
]

IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "audit",
    "models_data_backup", "*.pyc", "node_modules",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "joblib", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover
            out[mod] = f"ERR {e}"
    return out


def one_run(idx, committed):
    work = tempfile.mkdtemp(prefix=f"rto_repro_{idx}_")
    shutil.copytree(REPO, work, ignore=IGNORE, dirs_exist_ok=True)
    for rel in committed:
        p = os.path.join(work, rel)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ, PYTHONPATH=work, PYTHONHASHSEED="0")
    t0 = time.time()
    for step in CHAIN:
        r = subprocess.run([sys.executable] + step, cwd=work, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[run {idx}] FAILED at {step[0]}\n{r.stderr[-2000:]}")
            sys.exit(2)
        print(f"[run {idx}] ok  {step[0]}")
    digests = {}
    for rel in committed:
        p = os.path.join(work, rel)
        digests[rel] = sha256(p) if os.path.exists(p) else "MISSING"
    print(f"[run {idx}] rebuilt in {time.time() - t0:.1f}s  (workdir {work})")
    shutil.rmtree(work, ignore_errors=True)
    return digests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(os.path.join(REPO, "models", "artifact_hashes.json")) as f:
        committed = json.load(f)

    print("ENV:", json.dumps(versions()))
    runs = [one_run(i + 1, committed) for i in range(a.runs)]

    print()
    hdr = f"{'artifact':<40} | {'committed':<12} | " + " | ".join(f"run{i+1:<9}" for i in range(a.runs)) + " | run==run | ==committed"
    print(hdr)
    print("-" * len(hdr))
    det_ok = parity_ok = True
    for rel, ref in committed.items():
        ds = [r[rel] for r in runs]
        same = len(set(ds)) == 1
        par = all(d == ref for d in ds)
        det_ok &= same
        parity_ok &= par
        print(f"{rel:<40} | {ref[:12]} | " + " | ".join(d[:12] for d in ds)
              + f" | {'YES' if same else 'NO ':<8} | {'YES' if par else 'NO'}")
    print("-" * len(hdr))
    print(f"DETERMINISM (run vs run): {'PASS' if det_ok else 'FAIL'}")
    print(f"PARITY (rebuild vs committed hashes): {'PASS' if parity_ok else 'FAIL'}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"env": versions(), "committed": committed, "runs": runs,
                       "determinism": det_ok, "parity": parity_ok}, f, indent=2)
    sys.exit(0 if (det_ok and parity_ok) else 1)


if __name__ == "__main__":
    main()
