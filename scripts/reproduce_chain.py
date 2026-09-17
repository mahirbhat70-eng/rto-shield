import os
import sys
import shutil
import hashlib
import json
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PINNED_ARTIFACTS = [
    "data/processed/pincode_rate_lookup.csv",
    "data/processed/test.csv",
    "data/processed/train.csv",
    "data/processed/val_cal.csv",
    "data/processed/val_rep.csv",
    "models/logistic_baseline.pkl",
    "models/tree_model.pkl",
    "models/tree_model_booster.pkl",
    "models/tree_model_calibrated.pkl",
]

REFERENCE_DIGESTS = {
    "data/processed/pincode_rate_lookup.csv": "5e4b2f90662a3a697e9a109ec44ddfae16bcc6b3224935ec84c026266e68428f",
    "data/processed/test.csv": "cf29991737868ff0281f506d44ce6a240e33330a438cf12e16ff42124064adeb",
    "data/processed/train.csv": "689392d2425a4ae41dac3621b9f10236fc989e0639c6fb0dab2b0d2842292435",
    "data/processed/val_cal.csv": "85b8a63f2d4b37362e1842ea94b8be775e1dfb3ec7434fca9d24b8c2988f743f",
    "data/processed/val_rep.csv": "d44d195e549b3f774e9fe719133693e24a8031467ef078ed4ad1a268f6dc83d0",
    "models/logistic_baseline.pkl": "8530ca0636751d03b9c4cba5532d4a39908e084e75050630a5959ad0b23fd29d",
    "models/tree_model.pkl": "e75478885e620297d17f08cc9e7211aa1d1385f33cfb73560bbf916a77fa7477",
    "models/tree_model_booster.pkl": "db099b05d7af60a74c5f4a788b435cc2ef1c89a1026a2de81258875df796f23a",
    "models/tree_model_calibrated.pkl": "2e6b0c5198dd56b2053263ea3d25835bcd1ee95b57c2c982de5650a297538b2c",
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()

def main():
    # 1. Load reference digests
    ref_digests = REFERENCE_DIGESTS

    backup_dir = os.path.join(REPO_ROOT, "models_data_backup")
    os.makedirs(backup_dir, exist_ok=True)

    print("--- STEP 1: BACKING UP ---")
    for rel in PINNED_ARTIFACTS:
        src = os.path.join(REPO_ROOT, rel)
        dst = os.path.join(backup_dir, rel.replace("/", "_"))
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Backed up: {rel}")

    print("\n--- STEP 2: DELETING ARTIFACTS ---")
    for rel in PINNED_ARTIFACTS:
        src = os.path.join(REPO_ROOT, rel)
        if os.path.exists(src):
            os.remove(src)
            print(f"Deleted: {rel}")

    print("\n--- STEP 3: RUNNING FULL CHAIN ---")
    chain = [
        [sys.executable, "src/data/generator.py"],
        [sys.executable, "src/data/split.py"],
        [sys.executable, "src/serve/lookup.py"],
        [sys.executable, "src/models/logistic_baseline.py"],
        [sys.executable, "src/models/tree_model.py"],
        [sys.executable, "src/eval/calibration.py"],
        [sys.executable, "scripts/freeze_artifact_hashes.py"],
    ]

    for cmd in chain:
        print(f"\n>> Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED: {res.stderr}")
            # restore
            for rel in PINNED_ARTIFACTS:
                dst = os.path.join(backup_dir, rel.replace("/", "_"))
                src = os.path.join(REPO_ROOT, rel)
                if os.path.exists(dst) and not os.path.exists(src):
                    shutil.copy2(dst, src)
            sys.exit(1)
        print(res.stdout.strip()[-300:] if len(res.stdout) > 300 else res.stdout.strip())

    print("\n--- STEP 4: DIGEST COMPARISON ---")
    all_match = True
    print(f"{'Artifact':<42} | {'New Digest':<16} | {'Ref Digest':<16} | {'Match?'}")
    print("-" * 85)
    for rel in PINNED_ARTIFACTS:
        p = os.path.join(REPO_ROOT, rel)
        new_d = sha256_file(p)
        ref_d = ref_digests.get(rel, "MISSING")
        match = (new_d == ref_d)
        if not match:
            all_match = False
        print(f"{rel:<42} | {new_d[:16]} | {ref_d[:16]} | {'MATCH' if match else 'MISMATCH'}")

    # cleanup backup
    shutil.rmtree(backup_dir)

    print("-" * 85)
    if all_match:
        print("RESULT: Byte-exact reproduction confirmed! All 9 digests match.")
    else:
        print("RESULT: Digest mismatch detected.")

if __name__ == "__main__":
    main()
