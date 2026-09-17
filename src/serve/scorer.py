"""
src/serve/scorer.py
WHY: The production serving entry point. Converts raw order fields into a calibrated
     risk probability, expected-loss action table, recommended action, SHAP explanation,
     model version hash, and any data-quality warnings.

     Key design decisions (correcting v1 bugs):
     1. validate_and_clean() raises ONE ValueError listing ALL problems.
     2. is_cod() (from cost_engine) is the single source of truth.
     3. Non-COD orders → PREPAID_PASSTHROUGH immediately, ZERO SHAP calls.
        This eliminates the v1 bug where SHAP ran on every order, making
        PREPAID 2.4× slower than COD for no reason.
     4. Unknown pincodes fall back to global prior (fail-open with warning),
        NOT a hard ValueError. The v1 behaviour was a production stopper.
     5. MODEL_VERSION is the first 12 hex chars of the SHA-256 of the calibrated
        model file — identical across machines for the same artifact.
"""

import os
import sys
import html
import hashlib
import warnings as _warnings

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.policy.cost_engine import CostEngine, is_cod
from src.serve.lookup import resolve as _resolve_pincode
# IsoCalibratedPipeline must be importable here so joblib.load can unpickle it
from src.models.tree_model import IsoCalibratedPipeline  # noqa: F401

# ── Paths (absolute, CWD-independent) ──────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_MODELS    = os.path.join(_HERE, "..", "..", "models")
_CAL_PATH  = os.path.join(_MODELS, "tree_model_calibrated.pkl")
_PIPE_PATH = os.path.join(_MODELS, "tree_model.pkl")
_BOOS_PATH = os.path.join(_MODELS, "tree_model_booster.pkl")


# ── Model version hash ──────────────────────────────────────────────────────────
def _model_version(path: str = _CAL_PATH) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:12]
    except FileNotFoundError:
        return "unloaded"


MODEL_VERSION: str = _model_version()


# ── Lazy model loading ─────────────────────────────────────────────────────────
_tree_uncal  = None
_tree_cal    = None
_booster     = None
_explainer   = None
_engine      = None


def _load_models():
    global _tree_uncal, _tree_cal, _booster, _explainer, _engine
    if _tree_cal is None:
        _tree_uncal = joblib.load(_PIPE_PATH)
        _tree_cal   = joblib.load(_CAL_PATH)
        _booster    = joblib.load(_BOOS_PATH)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            _explainer = shap.TreeExplainer(_booster)
        _engine = CostEngine()


# ── Vocabulary constants ───────────────────────────────────────────────────────
_KNOWN_CATEGORIES = {"electronics", "apparel", "footwear", "beauty", "home", "jewelry"}
_KNOWN_COURIERS   = {"courier_a", "courier_b", "courier_c", "courier_d", "courier_e"}
_KNOWN_PAYMENTS   = {"cod", "upi", "credit card", "debit card", "net banking"}

# Numeric bounds: (inclusive_lower, inclusive_upper)
_NUMERIC_BOUNDS = {
    "order_value":        (1.0,   25000.0),
    "quantity":           (1,     50),
    "discount_pct":       (0.0,   100.0),
    "cod_charge":         (0.0,   500.0),
    "account_age_days":   (0,     5000),
    "prior_orders":       (0,     100),
    "prior_rto_count":    (0,     100),
    "orders_last_24h":    (0,     50),
    "device_cluster_size":(1,     50),
}

_REQUIRED_FIELDS = list(_NUMERIC_BOUNDS.keys()) + [
    "category", "payment_method", "cod_charge", "pincode", "courier_id"
]
# Deduplicate while preserving order
seen = set()
_REQUIRED_FIELDS = [x for x in _REQUIRED_FIELDS if not (x in seen or seen.add(x))]

_ORACLE_FIELDS = {"historical_pincode_rto_rate", "pincode_tier"}

# Model feature columns (must match training order)
_CATEGORICAL_FEATURES = ["category", "payment_method", "courier_id"]
_NUMERIC_FEATURES     = [
    "order_value", "quantity", "discount_pct", "cod_charge",
    "account_age_days", "prior_orders", "prior_rto_count",
    "orders_last_24h", "device_cluster_size",
    "historical_pincode_rto_rate", "pincode_tier",
]


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_and_clean(features: dict) -> tuple[dict, list[str]]:
    """
    Validate and clean raw feature dict.

    Accumulates ALL errors and raises ONE ValueError listing all of them.
    Appends non-fatal issues to the returned warnings list.

    Returns:
        (clean_features: dict, warnings: list[str])

    Raises:
        ValueError: if any hard constraint is violated.
    """
    errors:   list[str] = []
    warn:     list[str] = []
    clean:    dict      = {}

    # ── Oracle-field injection guard ──────────────────────────────────────────
    injected = _ORACLE_FIELDS & set(features.keys())
    if injected:
        errors.append(
            f"Oracle features must not be provided by the caller: {sorted(injected)}. "
            "They are resolved from the pincode lookup internally."
        )

    # ── Required field presence ───────────────────────────────────────────────
    missing = [f for f in _REQUIRED_FIELDS if f not in features]
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # ── Numeric coerce-then-check ─────────────────────────────────────────────
    for field, (lo, hi) in _NUMERIC_BOUNDS.items():
        if field not in features:
            continue  # already caught above
        raw = features[field]

        # Reject booleans first (bool is a subclass of int in Python)
        if isinstance(raw, bool):
            errors.append(f"Field '{field}': got bool, expected numeric value.")
            continue

        # Attempt float coercion
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Field '{field}': cannot convert {type(raw).__name__} '{raw}' to float.")
            continue

        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            errors.append(f"Field '{field}': value is None/NaN.")
            continue

        if not np.isfinite(val):
            errors.append(f"Field '{field}': value is non-finite ({val}).")
            continue

        if val < lo or val > hi:
            errors.append(
                f"Field '{field}': value {val} out of allowed range [{lo}, {hi}]."
            )
            continue

        clean[field] = val

    # ── Payment method ────────────────────────────────────────────────────────
    if "payment_method" in features:
        pm_raw = features["payment_method"]
        pm_str = str(pm_raw).strip() if pm_raw is not None else ""
        pm_lower = pm_str.lower()

        if pm_lower == "prepaid":
            # PREPAID was never seen in training — map to non-COD passthrough with OOD warning
            warn.append(
                "payment_method='PREPAID' is OOD (never seen in training). "
                "Treating as non-COD passthrough."
            )
            clean["payment_method"] = "PREPAID"
        elif pm_lower not in _KNOWN_PAYMENTS:
            warn.append(
                f"payment_method='{html.escape(pm_str)}' is an unknown value. "
                "Scored as-is; prediction may be unreliable (OOD extrapolation)."
            )
            clean["payment_method"] = pm_str
        else:
            # Canonical form
            clean["payment_method"] = pm_str

        # Cross-field: cod_charge must be 0 for non-COD
        if not is_cod(clean.get("payment_method", "")):
            cc = clean.get("cod_charge", features.get("cod_charge", 0))
            if float(cc) != 0.0:
                errors.append(
                    f"cod_charge must be 0.0 for non-COD orders "
                    f"(payment_method='{html.escape(pm_str)}', cod_charge={cc})."
                )

    # ── Category ──────────────────────────────────────────────────────────────
    if "category" in features:
        cat_raw = features["category"]
        cat_str = str(cat_raw).strip() if cat_raw is not None else ""
        if cat_str.lower() not in _KNOWN_CATEGORIES:
            warn.append(
                f"category='{html.escape(cat_str)}' is unknown/OOD. "
                "Scored as-is; prediction may be unreliable."
            )
        clean["category"] = cat_str

    # ── Courier ───────────────────────────────────────────────────────────────
    if "courier_id" in features:
        cour_raw = features["courier_id"]
        cour_str = str(cour_raw).strip() if cour_raw is not None else ""
        if cour_str.lower() not in _KNOWN_COURIERS:
            warn.append(
                f"courier_id='{html.escape(cour_str)}' is unknown/OOD. "
                "Scored as-is; prediction may be unreliable."
            )
        clean["courier_id"] = cour_str

    # ── Pincode normalization ─────────────────────────────────────────────────
    if "pincode" in features:
        pin_raw = features["pincode"]
        # Normalize int / float-int / str
        if isinstance(pin_raw, (int, float)):
            pin_str = f"{int(pin_raw):06d}"
        else:
            pin_str = str(pin_raw).strip()
        # Remove trailing .0 if any
        if pin_str.endswith(".0"):
            pin_str = pin_str[:-2]
        import re
        if not re.fullmatch(r"\d{6}", pin_str):
            errors.append(
                f"pincode '{html.escape(str(pin_raw))}' is not a valid 6-digit code "
                f"(got {len(pin_str)} chars after normalization)."
            )
        else:
            clean["pincode"] = pin_str

    # ── Cross-field logical rules ─────────────────────────────────────────────
    if "prior_rto_count" in clean and "prior_orders" in clean:
        if clean["prior_rto_count"] > clean["prior_orders"]:
            errors.append(
                f"prior_rto_count ({clean['prior_rto_count']}) cannot exceed "
                f"prior_orders ({clean['prior_orders']})."
            )

    if "account_age_days" in clean and "prior_orders" in clean:
        if clean["account_age_days"] == 0 and clean["prior_orders"] > 0:
            warn.append(
                "account_age_days=0 but prior_orders>0 — possible data-contract violation."
            )

    # ── Raise if any hard errors ───────────────────────────────────────────────
    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    return clean, warn


# ── Main scoring function ──────────────────────────────────────────────────────

def score_order(features: dict) -> dict:
    """
    Score a single order.

    Args:
        features: dict with the 14 required input fields.

    Returns:
        dict with keys:
            probability       : float — calibrated P(RTO)
            el_table          : dict[action → expected_loss]
            recommended_action: str
            shap_top_factors  : list[(feature, shap_value)] or [] for non-COD
            model_version     : str — SHA-256[:12] of calibrated model file
            warnings          : list[str]
    """
    _load_models()

    clean, warn = validate_and_clean(features)

    # ── Resolve pincode ────────────────────────────────────────────────────────
    pin = clean.get("pincode", "000000")
    pin_features, cold_start = _resolve_pincode(pin)
    if cold_start:
        warn.append(
            f"Pincode '{pin}' not found in lookup. "
            "Using global prior (mean RTO rate + modal tier). "
            "Prediction may be less accurate for new/uncommon pincodes."
        )
    clean["historical_pincode_rto_rate"] = pin_features["historical_pincode_rto_rate"]
    clean["pincode_tier"]                = pin_features["pincode_tier"]

    # ── Build feature row ──────────────────────────────────────────────────────
    row = {f: clean.get(f, 0) for f in _CATEGORICAL_FEATURES + _NUMERIC_FEATURES}
    df  = pd.DataFrame([row])

    # ── Non-COD fast path (ZERO SHAP, ZERO EL) ────────────────────────────────
    if not is_cod(clean.get("payment_method", "")):
        return {
            "probability":        None,
            "el_table":           {k: 0.0 for k in _engine.interventions},
            "recommended_action": "PREPAID_PASSTHROUGH",
            "shap_top_factors":   [],
            "model_version":      MODEL_VERSION,
            "warnings":           warn,
        }

    # ── COD path: calibrated probability ──────────────────────────────────────
    p = float(_tree_cal.predict_proba(df)[0, 1])

    # ── EL table + argmin ──────────────────────────────────────────────────────
    el_table = _engine.evaluate_interventions(clean["order_value"], p)
    recommended_action = min(el_table, key=el_table.get)

    # ── TreeSHAP (COD only) ────────────────────────────────────────────────────
    X_transformed = _tree_uncal.named_steps["preprocessor"].transform(df)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        sv = _explainer.shap_values(X_transformed)

    # Handle both old and new SHAP output formats
    if isinstance(sv, list):
        shap_vals = np.asarray(sv[1])[0]
    elif np.asarray(sv).ndim == 3:
        shap_vals = np.asarray(sv)[0, :, 1]
    else:
        shap_vals = np.asarray(sv)[0]

    feature_names = _tree_uncal.named_steps["preprocessor"].get_feature_names_out()
    shap_dict     = dict(zip(feature_names, shap_vals.tolist()))
    top_factors   = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "probability":        p,
        "el_table":           el_table,
        "recommended_action": recommended_action,
        "shap_top_factors":   top_factors,
        "model_version":      MODEL_VERSION,
        "warnings":           warn,
    }


def route_order(features: dict) -> str:
    """Convenience wrapper — returns only the recommended action string."""
    return score_order(features)["recommended_action"]
