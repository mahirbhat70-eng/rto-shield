"""
src/serve/tenants.py — Multi-Tenant Registry & Tenant Isolation for FlipPrice AI.

Implements:
- Standard library SQLite persistence (`data/tenants.db`).
- DPDP Act 2023 compliant audit storage (zero raw PII).
- Salted SHA-256 API key authentication.
- Strict tenant isolation guards.
"""

import os
import sqlite3
import hashlib
import hmac
import json
import time
from typing import Optional, Dict, Any, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(ROOT, "data", "tenants.db")
SECRET_SALT = os.getenv("FLIPPRICE_AUTH_SALT", "flipprice_auth_salt_v2_2026")

class TenantManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    api_key_hash TEXT UNIQUE NOT NULL,
                    api_key_prefix TEXT NOT NULL,
                    merchant_name TEXT NOT NULL,
                    rate_limit_rpm INTEGER DEFAULT 600,
                    is_active INTEGER DEFAULT 1,
                    custom_config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    decision_fingerprint TEXT NOT NULL,
                    action TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    expected_savings_inr REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_tenant_order ON audit_logs(tenant_id, order_id)")

            # Seed default test tenants if empty
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tenants")
            if cursor.fetchone()[0] == 0:
                self._seed_default_tenants(conn)

    def _hash_api_key(self, api_key: str) -> str:
        return hmac.new(SECRET_SALT.encode("utf-8"), api_key.strip().encode("utf-8"), hashlib.sha256).hexdigest()

    def _seed_default_tenants(self, conn: sqlite3.Connection):
        demo_keys = [
            ("tenant_demo_1", "fp_live_demo_merchant_123", "Demo Apparel D2C", 600),
            ("tenant_pilot_alpha", "fp_live_pilot_alpha_456", "Pilot Alpha Brand", 1200),
            ("tenant_test_isolated", "fp_test_isolated_789", "Isolated Test Store", 600),
        ]
        for tid, raw_key, name, rpm in demo_keys:
            h = self._hash_api_key(raw_key)
            prefix = raw_key[:12] + "..."
            conn.execute("""
                INSERT OR IGNORE INTO tenants (tenant_id, api_key_hash, api_key_prefix, merchant_name, rate_limit_rpm, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (tid, h, prefix, name, rpm))
        conn.commit()

    def register_tenant(self, tenant_id: str, raw_api_key: str, merchant_name: str, rate_limit_rpm: int = 600, custom_config: dict = None) -> Dict[str, Any]:
        """Register a new merchant tenant."""
        h = self._hash_api_key(raw_api_key)
        prefix = raw_api_key[:12] + "..."
        cfg_json = json.dumps(custom_config) if custom_config else None
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO tenants (tenant_id, api_key_hash, api_key_prefix, merchant_name, rate_limit_rpm, is_active, custom_config_json)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (tenant_id, h, prefix, merchant_name, rate_limit_rpm, cfg_json))
            conn.commit()
        return {"tenant_id": tenant_id, "merchant_name": merchant_name, "api_key_prefix": prefix}

    def authenticate(self, raw_api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return tenant context. Returns None if invalid or inactive."""
        if not raw_api_key:
            return None
        h = self._hash_api_key(raw_api_key)
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tenants WHERE api_key_hash = ? AND is_active = 1", (h,)).fetchone()
            if not row:
                return None
            return {
                "tenant_id": row["tenant_id"],
                "merchant_name": row["merchant_name"],
                "rate_limit_rpm": row["rate_limit_rpm"],
                "custom_config": json.loads(row["custom_config_json"]) if row["custom_config_json"] else None,
            }

    def log_decision(self, tenant_id: str, order_id: str, decision_fingerprint: str, action: str, risk_score: float, savings_inr: float, latency_ms: float):
        """Append an audit log under tenant isolation."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO audit_logs (tenant_id, order_id, decision_fingerprint, action, risk_score, expected_savings_inr, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (tenant_id, str(order_id), decision_fingerprint, action, risk_score, savings_inr, latency_ms))
            conn.commit()

    def log_decisions_batch(self, logs: List[tuple]):
        """Bulk append audit logs in a single atomic transaction for high throughput."""
        if not logs:
            return
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO audit_logs (tenant_id, order_id, decision_fingerprint, action, risk_score, expected_savings_inr, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, logs)
            conn.commit()

    def get_order_decision(self, tenant_id: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetch past order decision strictly scoped to the requesting tenant."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM audit_logs
                WHERE tenant_id = ? AND order_id = ?
                ORDER BY id DESC LIMIT 1
            """, (tenant_id, str(order_id))).fetchone()
            if not row:
                return None
            return dict(row)

    def list_recent_decisions(self, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List tenant decisions with tenant isolation."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM audit_logs
                WHERE tenant_id = ?
                ORDER BY id DESC LIMIT ?
            """, (tenant_id, limit)).fetchall()
            return [dict(r) for r in rows]
