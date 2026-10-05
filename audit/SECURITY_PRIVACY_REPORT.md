# RTO Shield — Privacy and Security Review (Prompt 16)

**Auditor:** Adversarial Security Audit  
**Date:** October 2026  
**Scope:** Data ingestion, model storage, serving path, audit logging, and runtime dependencies.

---

## 1. PII Exposure Surface Analysis

| Surface | Present in Repo? | Risk Level | Details |
|---|---|---|---|
| **Raw Input Payloads** | Yes | **HIGH** | `src/serve/audit.py:build_audit_record` stores the entire raw `payload` dict directly into the audit record (`"payload": payload`). If merchant integration passes phone, name, email, or address fields in the request, they are stored in plaintext in the JSONL audit trail without redaction or masking. |
| **Pincode / Geolocation** | Yes | **LOW-MEDIUM** | Pincodes (6-digit) are stored in plaintext. Indian pincodes represent postal areas (~10,000–50,000 residents), not specific individuals, but combined with order timestamp and basket value could facilitate re-identification. |
| **Customer Identifiers** | Yes (`CUST000001`...) | **MEDIUM** | In synthetic data, customer IDs are pseudonymous. In production, raw merchant customer IDs (or emails/phones used as IDs) MUST be salted and hashed before logging. |
| **Error Logs** | Yes | **MEDIUM** | `src/serve/scorer.py:validate_and_clean` includes raw field values in exception messages (e.g. `Field 'order_value' is below minimum... got -50.0`). Unsanitized error strings logged to Datadog/CloudWatch could leak user input. |
| **Model Artifacts** | Clean | **NONE** | Pickles contain LightGBM decision trees and isotonic regression thresholds; no training customer PII is memorized in plaintext. |

---

## 2. Model Serialization & Deserialization (Pickle Vulnerabilities)

1. **Unsafe `joblib.load()` Execution:**
   - In `src/serve/scorer.py`, model pickles (`tree_model.pkl`, `tree_model_calibrated.pkl`, `tree_model_booster.pkl`) are loaded directly using `joblib.load()`.
   - Python's standard `pickle` module allows arbitrary code execution during deserialization.
2. **Missing Pre-Load Cryptographic Verification:**
   - Although `models/artifact_hashes.json` exists in the repo with exact SHA-256 digests for all 9 core artifacts, `src/serve/scorer.py` **never verifies the pickle hashes before calling `joblib.load()`**.
   - If an attacker gains write access to `models/`, they can replace `tree_model.pkl` with a malicious pickle payload that executes shell commands upon API startup.
   - **Recommendation:** Implement mandatory pre-load digest verification: read file bytes, compute SHA-256, verify against `models/artifact_hashes.json`, and abort if hashes mismatch.

---

## 3. Dependency Vulnerability & Supply Chain

- **Pinned Versions:** `requirements.txt` contains pinned major/minor libraries (`scikit-learn>=1.3.0`, `lightgbm>=4.0.0`, `shap>=0.42.0`).
- **SHAP Warning Spam:** TreeSHAP generates thousands of deprecation warnings on newer LightGBM versions.
- **Supply Chain:** Standard PyPI packages; no internal package repository or hash verification in `requirements.txt`.

---

## 4. Production Security Checklist Before Pilot Rollout

- [ ] Strip or salt-hash any identifier (merchant customer ID, IP, device ID) before audit logging.
- [ ] Enforce schema whitelist in `scorer.py`: discard any unlisted key (such as `name`, `phone`, `address`) before building the audit record.
- [ ] Add SHA-256 integrity check inside `scorer.py` before `joblib.load()`.
- [ ] Add rate limiting at API gateway / reverse proxy to mitigate denial-of-service via TreeSHAP calculation.
