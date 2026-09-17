"""
test_artifact_integrity.py — Byte-level integrity of frozen artifacts.

WHAT_BROKE.md Bug 4 claimed model integrity was regression-tested; it wasn't
(the old check called os.path.exists only). This test pins the SHA-256 of
every frozen binary artifact against models/artifact_hashes.json, so any
byte drift (sync tools, LFS re-encoding, accidental rebuilds) fails CI.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MANIFEST = os.path.join(os.path.dirname(__file__), '..', 'models', 'artifact_hashes.json')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_exists_and_covers_models():
    assert os.path.exists(MANIFEST), "models/artifact_hashes.json missing — run scripts/freeze_artifact_hashes.py"
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    for required in ("models/tree_model.pkl", "models/tree_model_calibrated.pkl",
                     "models/tree_model_booster.pkl"):
        assert required in manifest, f"manifest missing {required}"


def test_frozen_artifacts_match_pinned_digests():
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    mismatches = []
    for rel_path, expected in sorted(manifest.items()):
        path = os.path.join(os.path.dirname(__file__), '..', rel_path)
        assert os.path.exists(path), f"artifact missing: {rel_path}"
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(f"{rel_path}: pinned {expected[:12]}…, actual {actual[:12]}…")
    assert not mismatches, (
        "Frozen artifact(s) drifted from pinned digests:\n" + "\n".join(mismatches)
        + "\nIf the change is intentional, re-run scripts/freeze_artifact_hashes.py "
          "and re-freeze the reports in the same commit."
    )


def test_model_version_matches_manifest():
    # The serving layer's MODEL_VERSION must be derived from the pinned bytes.
    from src.serve.scorer import MODEL_VERSION  # noqa: E402
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    pinned = manifest["models/tree_model_calibrated.pkl"]
    assert MODEL_VERSION == pinned[:12], (
        "scorer.MODEL_VERSION does not match the pinned calibrated artifact digest"
    )


def test_byte_flip_fails_integrity(tmp_path):
    # Verify that tampering/flipping even a single byte causes SHA-256 mismatch
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    rel_path = "models/tree_model_calibrated.pkl"
    src_path = os.path.join(os.path.dirname(__file__), '..', rel_path)
    with open(src_path, 'rb') as f:
        data = bytearray(f.read())
    
    # Flip the first byte
    data[0] ^= 0xFF
    
    tampered_file = tmp_path / "tampered.pkl"
    with open(tampered_file, 'wb') as f:
        f.write(data)
    
    tampered_digest = sha256_file(str(tampered_file))
    assert tampered_digest != manifest[rel_path], "Byte-flip MUST change sha256 digest"

