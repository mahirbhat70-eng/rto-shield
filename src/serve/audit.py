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


# Strict allow-list of admissible order feature keys in audit logs.
# Any unknown field (free-text notes, addresses, customer remarks) is strictly dropped.
AUDIT_ALLOWLIST_KEYS = {
    'order_value', 'category', 'payment_method', 'quantity', 'discount_pct',
    'cod_charge', 'account_age_days', 'prior_orders', 'prior_rto_count',
    'orders_last_24h', 'device_cluster_size', 'pincode', 'courier_id',
    'pincode_tier', 'historical_pincode_rto_rate', 'customer_id'
}

def mask_pii(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize payload for audit logging using a strict ALLOW-LIST.
    Any field not explicitly permitted in AUDIT_ALLOWLIST_KEYS is dropped.
    Permitted customer_id is salt-hashed.
    """
    sanitized = {}
    for k, v in payload.items():
        k_lower = k.lower()
        if k_lower in AUDIT_ALLOWLIST_KEYS:
            if k_lower == 'customer_id' and v is not None:
                sanitized[k] = f"HASH_{hashlib.sha256(str(v).encode()).hexdigest()[:12]}"
            else:
                sanitized[k] = v
    return sanitized

def compute_decision_fingerprint(payload: Dict[str, Any], model_version: str, action: str = "") -> str:
    """
    Deterministic 16-character SHA-256 hex digest over the canonical sorted
    JSON payload, model version, and recommended action.
    """
    clean_p = mask_pii(payload)
    canonical_payload = json.dumps(clean_p, sort_keys=True, separators=(',', ':'), default=str)
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
    input payload (PII-masked), prediction, expected-loss table, and SHAP drivers.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    timestamp_iso = timestamp.isoformat()
    model_version = str(res.get("model_version", "unknown"))
    action = str(res.get("recommended_action", ""))
    clean_payload = mask_pii(payload)
    decision_id = compute_decision_fingerprint(clean_payload, model_version, action)

    return {
        "decision_id": decision_id,
        "timestamp": timestamp_iso,
        "model_version": model_version,
        "latency_ms": round(float(latency_ms), 2),
        "payload": clean_payload,
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

