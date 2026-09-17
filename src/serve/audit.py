"""
src/serve/audit.py
WHY: Immutable, append-only audit trail for every scoring decision.
     Each record contains a cryptographic decision_id derived from the
     payload content (NOT wall-clock time), making it deterministic and
     idempotent — the same inputs always produce the same decision_id.

     This supports:
     - Compliance: every decision is traceable to an exact model version.
     - Tamper detection: verify_audit_record() recomputes and compares.
     - Replay testing: identical payloads can be re-scored and matched.
"""

import hashlib
import json
import os
from typing import Any


def _payload_hash(payload: dict, result: dict) -> str:
    """
    Deterministic SHA-256 hash of (payload, result).
    Excludes wall-clock timestamps to keep the hash idempotent.
    """
    canonical = json.dumps(
        {"payload": payload, "probability": result.get("probability"),
         "recommended_action": result.get("recommended_action"),
         "model_version": result.get("model_version")},
        sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_audit_record(payload: dict, result: dict, latency_ms: float = None) -> dict:
    """
    Build a single audit record dict.

    Args:
        payload:    The raw input features dict.
        result:     The dict returned by score_order().
        latency_ms: Optional end-to-end latency in milliseconds.

    Returns:
        dict suitable for JSON serialization and JSONL appending.
    """
    decision_id = _payload_hash(payload, result)
    record = {
        "decision_id":        decision_id,
        "model_version":      result.get("model_version"),
        "payload":            payload,
        "probability":        result.get("probability"),
        "el_table":           result.get("el_table"),
        "recommended_action": result.get("recommended_action"),
        "shap_top_factors":   result.get("shap_top_factors"),
        "warnings":           result.get("warnings", []),
        "latency_ms":         latency_ms,
    }
    return record


def verify_audit_record(record: dict) -> bool:
    """
    Verify that a stored audit record has not been tampered with.

    Recomputes the decision_id from payload and result fields and
    compares to the stored value.

    Returns:
        True if the record is intact, False if tampered.
    """
    stored_id = record.get("decision_id")
    payload = record.get("payload", {})
    result  = {
        "probability":        record.get("probability"),
        "recommended_action": record.get("recommended_action"),
        "model_version":      record.get("model_version"),
    }
    recomputed = _payload_hash(payload, result)
    return recomputed == stored_id


def to_jsonl(records: list[dict]) -> str:
    """Serialize a list of audit records to JSONL format."""
    return "\n".join(json.dumps(r, default=str) for r in records) + "\n"


def append_to_file(record: dict, path: str) -> None:
    """Append a single audit record to a JSONL file (creates if missing)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
