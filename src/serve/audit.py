"""
audit.py — Audit trail serialization and replay fingerprinting for RTO Shield decisions.

Changes vs v1 (see IMPROVEMENTS.md):
  * decision_id now hashes (canonical payload || model_version) instead of
    (payload || timestamp). Replaying the identical order against the same
    frozen model now produces the SAME fingerprint — the ID is replay-stable
    AND tamper-evident, and ties each decision to the exact model artifact.
  * Records carry an explicit `model_version` field (digest of the frozen
    calibrated model, provided by src/serve/scorer.py).
"""

import json
import hashlib
import datetime
from typing import Dict, Any, List, Optional


def compute_decision_fingerprint(payload: Dict[str, Any], model_version: str, action: str = "") -> str:
    """
    Deterministic 16-character SHA-256 hex digest over the canonical sorted
    JSON payload, model version, and recommended action.
    """
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    raw_str = f"{canonical_payload}|{model_version}"
    if action:
        raw_str += f"|{action}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


_payload_hash = compute_decision_fingerprint


def build_audit_record(
    payload: Dict[str, Any],
    res: Dict[str, Any],
    latency_ms: float = 0.0,
    timestamp: Optional[datetime.datetime] = None
) -> Dict[str, Any]:
    """
    Standardized audit record: replay fingerprint, model version, latency,
    input payload, prediction, expected-loss table, and SHAP drivers.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    timestamp_iso = timestamp.isoformat()
    model_version = str(res.get("model_version", "unknown"))
    action = str(res.get("recommended_action", ""))
    decision_id = compute_decision_fingerprint(payload, model_version, action)

    return {
        "decision_id": decision_id,
        "timestamp": timestamp_iso,
        "model_version": model_version,
        "latency_ms": round(float(latency_ms), 2),
        "payload": payload,
        "probability": round(float(res["probability"]), 4) if res.get("probability") is not None else 0.0,
        "recommended_action": action,
        "expected_losses": {k: round(float(v), 2) for k, v in res.get("el_table", {}).items()},
        "el_table": res.get("el_table", {}),
        "top_factors": [
            {"feature": name, "shap_val": round(float(val), 4)}
            for name, val in res.get("shap_top_factors", [])
        ],
        "shap_top_factors": res.get("shap_top_factors", []),
        "warnings": res.get("warnings", []),
    }


def to_jsonl(records: List[Dict[str, Any]]) -> str:
    """Newline-delimited JSON serialization (strict: NaN/Infinity rejected)."""
    return "\n".join(
        json.dumps(r, separators=(',', ':'), allow_nan=False) for r in records
    )


def verify_audit_record(record: Dict[str, Any]) -> bool:
    """
    True iff the record's decision_id matches its payload + model_version + action.
    Detects tampering with the payload, action, or a model-version swap after the fact.
    """
    try:
        expected_id = compute_decision_fingerprint(
            record["payload"],
            record.get("model_version", "unknown"),
            record.get("recommended_action", "")
        )
        return record["decision_id"] == expected_id
    except (KeyError, TypeError):
        return False

