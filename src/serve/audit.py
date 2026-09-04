"""
audit.py — Audit trail serialization and replay fingerprinting for RTO Shield decisions.
"""

import json
import hashlib
import datetime
from typing import Dict, Any, List, Optional


def compute_decision_fingerprint(payload: Dict[str, Any], timestamp_str: str) -> str:
    """
    Computes a deterministic 16-character SHA-256 hex digest fingerprint
    over the canonical sorted JSON payload and ISO timestamp.
    Serves as an immutable replay fingerprint.
    """
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    raw_str = f"{canonical_payload}|{timestamp_str}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def build_audit_record(
    payload: Dict[str, Any],
    res: Dict[str, Any],
    latency_ms: float,
    timestamp: Optional[datetime.datetime] = None
) -> Dict[str, Any]:
    """
    Builds a standardized, validated audit record containing the replay fingerprint,
    latency, input payload, output prediction, expected loss table, and SHAP drivers.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    timestamp_iso = timestamp.isoformat()
    decision_id = compute_decision_fingerprint(payload, timestamp_iso)

    return {
        "decision_id": decision_id,
        "timestamp": timestamp_iso,
        "latency_ms": round(float(latency_ms), 2),
        "payload": payload,
        "probability": round(float(res["probability"]), 4),
        "recommended_action": res["recommended_action"],
        "expected_losses": {k: round(float(v), 2) for k, v in res.get("el_table", {}).items()},
        "top_factors": [
            {"feature": name, "shap_val": round(float(val), 4)}
            for name, val in res.get("shap_top_factors", [])
        ]
    }


def to_jsonl(records: List[Dict[str, Any]]) -> str:
    """Serializes a list of audit records to newline-delimited JSON (JSONL)."""
    return "\n".join(json.dumps(r, separators=(',', ':')) for r in records)


def verify_audit_record(record: Dict[str, Any]) -> bool:
    """
    Verifies that the audit record's decision_id matches its payload and timestamp.
    Returns True if valid, False if tampered or corrupt.
    """
    try:
        expected_id = compute_decision_fingerprint(record["payload"], record["timestamp"])
        return record["decision_id"] == expected_id
    except (KeyError, TypeError):
        return False
