"""
freeze_artifact_hashes.py — Record SHA-256 digests of frozen binary artifacts.

Writes models/artifact_hashes.json, consumed by tests/test_stage3.py
(test_stage3_artifact_check) and tests/test_artifact_integrity.py so that the
committed model blobs are byte-pinned. Previously the "artifact check" only
asserted file existence (see WHAT_BROKE.md Bug 4 note in IMPROVEMENTS.md).

Usage:
    python scripts/freeze_artifact_hashes.py
"""

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "models", "artifact_hashes.json")

PINNED_ARTIFACTS = [
    "models/tree_model.pkl",
    "models/tree_model_calibrated.pkl",
    "models/tree_model_booster.pkl",
    "models/logistic_baseline.pkl",
    "data/processed/pincode_rate_lookup.csv",
    "data/processed/test.csv",
    "data/processed/val_cal.csv",
    "data/processed/val_rep.csv",
    "data/processed/train.csv",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {}
    for rel in PINNED_ARTIFACTS:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            print(f"SKIP (missing): {rel}")
            continue
        digest = sha256_file(path)
        manifest[rel] = digest
        print(f"{digest[:16]}…  {rel}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nWrote {len(manifest)} digests -> {os.path.relpath(MANIFEST_PATH, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
