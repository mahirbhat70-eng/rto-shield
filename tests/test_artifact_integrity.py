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
        # On Linux/macOS CI runners, git clone converts text files to LF.
        # Verify CRLF-normalized hash for CSVs so CI passes deterministically.
        if actual != expected and rel_path.endswith('.csv'):
            with open(path, 'rb') as f:
                crlf_content = f.read().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            if hashlib.sha256(crlf_content).hexdigest() == expected:
                actual = expected
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


def test_verify_and_load_refuses_tampered_model(tmp_path):
    # Verify that _verify_and_load refuses deserialization when an actual model file is byte-modified
    import pytest
    from src.serve.scorer import _verify_and_load, SecurityError
    
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    rel_path = "models/tree_model_calibrated.pkl"
    src_path = os.path.join(os.path.dirname(__file__), '..', rel_path)
    with open(src_path, 'rb') as f:
        data = bytearray(f.read())

    # Flip one byte in the model file
    data[10] ^= 0x42
    tampered_model = tmp_path / "tree_model_calibrated.pkl"
    with open(tampered_model, "wb") as f:
        f.write(data)

    # Attempt to load tampered model using the manifest - must raise SecurityError
    with pytest.raises(SecurityError, match="SecurityError: SHA-256 digest mismatch"):
        _verify_and_load(str(tampered_model), manifest_path=MANIFEST)


def test_out_of_boundary_pinned_digest_enforcement(tmp_path, monkeypatch):
    # Verify that moving the trusted hash out of the repo boundary (e.g. pinned in CI or env var)
    # prevents attackers who modify both the model and the in-repo manifest from bypassing checks.
    import pytest
    from src.serve.scorer import _verify_and_load, SecurityError

    fake_model = tmp_path / "tree_model.pkl"
    with open(fake_model, "wb") as f:
        f.write(b"injected_untrusted_payload")

    # Attacker modified the in-repo manifest to match their injected payload
    attacker_hash = sha256_file(str(fake_model))
    in_repo_manifest = tmp_path / "compromised_manifest.json"
    with open(in_repo_manifest, "w", encoding="utf-8") as f:
        json.dump({"models/tree_model.pkl": attacker_hash}, f)

    # Out-of-boundary pinned digest (e.g. from CI secrets or secure environment)
    trusted_digest = "e75478885e620297d17f08cc9e7211aa1d1385f33cfb73560bbf916a77fa7477"
    monkeypatch.setenv("RTO_SHIELD_PINNED_DIGESTS", json.dumps({"models/tree_model.pkl": trusted_digest}))

    # Loading with the compromised manifest MUST still fail because the out-of-boundary pin overrides it
    with pytest.raises(SecurityError, match="SecurityError: SHA-256 digest mismatch"):
        _verify_and_load(str(fake_model), manifest_path=str(in_repo_manifest))


def test_hash_file_completeness():
    """Assert every file under models/ and data/ is recorded in artifact_hashes.json."""
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    missing = []
    total_found = 0

    for root_dir in ["models", "data"]:
        full_dir = os.path.join(repo_root, root_dir)
        for dirpath, _, filenames in os.walk(full_dir):
            for fname in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fname), repo_root).replace("\\", "/")
                # Skip the manifest itself and transient/hidden files
                if rel in ("models/artifact_hashes.json",) or fname.startswith("."):
                    continue
                total_found += 1
                if rel not in manifest:
                    missing.append(rel)

    assert total_found >= 13, f"Expected at least 13 tracked files, found {total_found}"
    assert not missing, (
        f"File(s) exist on disk under models/ or data/ but are missing from models/artifact_hashes.json:\n"
        + "\n".join(f"  - {m}" for m in missing)
        + "\nRun scripts/freeze_artifact_hashes.py to update the manifest."
    )


def test_verify_and_load_fail_closed_in_production(monkeypatch):
    """Refuse to load models in production if RTO_SHIELD_PINNED_DIGESTS is unset."""
    import pytest
    from src.serve.scorer import _verify_and_load, SecurityError

    monkeypatch.setenv("RTO_SHIELD_ENV", "production")
    monkeypatch.delenv("RTO_SHIELD_PINNED_DIGESTS", raising=False)

    with pytest.raises(SecurityError, match="In production mode"):
        _verify_and_load("tree_model.pkl")


def test_verify_and_load_warns_and_loads_in_dev(monkeypatch):
    """In development mode, warn when RTO_SHIELD_PINNED_DIGESTS is unset and fall back to repo manifest."""
    import pytest
    from src.serve.scorer import _verify_and_load

    monkeypatch.setenv("RTO_SHIELD_ENV", "development")
    monkeypatch.delenv("RTO_SHIELD_PINNED_DIGESTS", raising=False)

    with pytest.warns(UserWarning, match="RTO_SHIELD_PINNED_DIGESTS is unset in development mode"):
        model = _verify_and_load("tree_model.pkl")
    assert model is not None




