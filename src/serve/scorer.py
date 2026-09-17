"""
scorer.py — Serving entrypoint for RTO Shield (hardened).

Changes vs v1 (see IMPROVEMENTS.md):
  * One COD predicate shared with the batch policy (src.policy.cost_engine.is_cod)
    — every non-COD payment is a passthrough, in both single and batch paths.
  * validate_features now coerces-then-checks: type/bool/finite validation,
    lower AND upper bounds, categorical canonicalization (case-insensitive,
    out-of-vocabulary warning), pincode normalization + cold-start fallback.
  * SHAP is computed only for COD orders (passthrough orders skip it).
  * Every result carries `model_version` (digest of the frozen calibrated
    artifact) and a `warnings` list surfaced by the UI and the audit trail.
"""

import os
import re
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd
import shap

# Load models and configs at module level to simulate a serving environment
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.policy.cost_engine import CostEngine, is_cod

# Load lookup table
lookup_df = pd.read_csv(os.path.join(DATA_DIR, 'pincode_rate_lookup.csv'), dtype={'pincode': str})
PINCODE_LOOKUP = lookup_df.set_index('pincode').to_dict(orient='index')

# Cold-start prior: global mean historical rate + modal tier (used when a
# pincode is absent from the train-derived lookup, e.g. a new serviceable area).
COLD_START_PRIOR = {
    'historical_pincode_rto_rate': float(lookup_df['historical_pincode_rto_rate'].mean()),
    'pincode_tier': int(lookup_df['pincode_tier'].mode().iloc[0]),
}

# Load models
try:
    tree_uncal = joblib.load(os.path.join(MODEL_DIR, 'tree_model.pkl'))  # contains encoder pipeline
    tree_cal = joblib.load(os.path.join(MODEL_DIR, 'tree_model_calibrated.pkl'))
    booster = joblib.load(os.path.join(MODEL_DIR, 'tree_model_booster.pkl'))
except Exception as exc:  # pragma: no cover - actionable message if artifacts are moved
    raise RuntimeError(
        f"Could not load frozen model artifacts from '{MODEL_DIR}'. "
        f"Run from the repository root or restore models/*.pkl. Original error: {exc}"
    ) from exc

explainer = shap.TreeExplainer(booster)

# Initialize Cost Engine (absolute path — CWD independent)
engine = CostEngine(config_path=os.path.join(CONFIG_DIR, 'cost_config.yaml'))

# Digest of the frozen calibrated artifact: stamped into every score + audit
# record so decisions are attributable to an exact model version.
with open(os.path.join(MODEL_DIR, 'tree_model_calibrated.pkl'), 'rb') as _f:
    MODEL_VERSION = hashlib.sha256(_f.read()).hexdigest()[:12]


REQUIRED_INPUTS = [
    'order_value', 'quantity', 'category', 'discount_pct', 'payment_method',
    'cod_charge', 'account_age_days', 'prior_orders', 'prior_rto_count',
    'pincode', 'courier_id', 'orders_last_24h', 'device_cluster_size'
]

NUMERIC_FIELDS = [
    'order_value', 'quantity', 'discount_pct', 'cod_charge',
    'account_age_days', 'prior_orders', 'prior_rto_count',
    'orders_last_24h', 'device_cluster_size'
]

# (lower, upper) inclusive bounds — P99.9 guardrails from the training contract.
NUMERIC_BOUNDS = {
    'order_value': (1.0, 25000.0),
    'quantity': (1, 50),
    'discount_pct': (0.0, 100.0),
    'cod_charge': (0.0, 500.0),
    'account_age_days': (0, 5000),
    'prior_orders': (0, 100),
    'prior_rto_count': (0, 100),
    'orders_last_24h': (0, 50),
    'device_cluster_size': (1, 50),
}

# Canonical vocabularies (case-insensitive lookup). Unknown values are still
# scored (OneHotEncoder handle_unknown='ignore') but flagged as OOD warnings.
ALLOWED_CATEGORIES = ['Electronics', 'Apparel', 'Footwear', 'Beauty', 'Home', 'Jewelry']
ALLOWED_PAYMENT = ['COD', 'UPI', 'Credit Card', 'Debit Card', 'Net Banking']
ALLOWED_COURIERS = ['Courier_A', 'Courier_B', 'Courier_C', 'Courier_D', 'Courier_E']

_CANON = {
    'category': {v.lower(): v for v in ALLOWED_CATEGORIES},
    'payment_method': {v.lower(): v for v in ALLOWED_PAYMENT + ['PREPAID']},
    'courier_id': {v.lower(): v for v in ALLOWED_COURIERS},
}

_PINCODE_RE = re.compile(r'^\d{6}$')


def _canonicalize_categorical(field, value, warnings):
    """Case-insensitive canonicalization + OOD detection for string features."""
    if value is None:
        raise ValueError(f"Field '{field}' must be a non-null string.")
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        raise ValueError(f"Field '{field}' must be a non-empty string.")
    canonical = _CANON[field].get(value.lower())
    if canonical is not None:
        if field == 'payment_method' and canonical == 'PREPAID':
            warnings.append(
                "'PREPAID' was never observed in training data — the probability is an "
                "out-of-distribution extrapolation (treated as non-COD passthrough)."
            )
        return canonical
    warnings.append(
        f"Unknown {field} '{value}' — not in the training vocabulary; the model treats it "
        "as an unseen category (all-zero one-hot). Verify the spelling."
    )
    return value


def _normalize_pincode(value):
    """Normalize int/float/str pincode input to a 6-digit string."""
    if value is None:
        raise ValueError("Field 'pincode' must be provided.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("Field 'pincode' must be a 6-digit string or integer.")
    value = value.strip()
    if not _PINCODE_RE.match(value):
        raise ValueError(
            f"Pincode '{value}' is not a valid 6-digit code (got {len(value)} characters)."
        )
    return value


def resolve_pincode_info(pincode):
    """
    Resolve a normalized pincode to (feature_info, is_cold).
    Cold pincodes fall back to the global prior — matching the documented
    behavior in docs/JUDGE_QA.md (previously the code raised; see IMPROVEMENTS.md).
    """
    info = PINCODE_LOOKUP.get(pincode)
    if info is not None:
        return info, False
    return dict(COLD_START_PRIOR), True


def _coerce_numeric(field, value):
    """Coerce to float with strict type/finite checks. Raises ValueError."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(f"Field '{field}' must not be NaN or None.")
    if isinstance(value, bool):
        raise ValueError(f"Field '{field}' must be numeric (got bool).")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Field '{field}' must be numeric (got {type(value).__name__}: {value!r})."
        )
    if not np.isfinite(x):
        raise ValueError(f"Field '{field}' must be a finite number (got {x}).")
    return x


def validate_and_clean(features: dict):
    """
    Full input validation. Returns (clean_features, warnings).
    Raises ValueError with all problems listed (message is user-facing).
    """
    errors = []
    warnings = []

    if not isinstance(features, dict):
        raise ValueError("features must be a dict of order fields.")

    # 1. Missing fields
    missing = [f for f in REQUIRED_INPUTS if f not in features]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")

    # Oracle-feature injection guard (these must come from the lookup, not the caller)
    provided_oracle = [f for f in ('historical_pincode_rto_rate', 'pincode_tier') if f in features]
    if provided_oracle:
        errors.append(
            "Do not provide historical_pincode_rto_rate or pincode_tier; "
            "these are looked up via pincode."
        )

    clean = {}
    if not missing:
        # 2. Pincode (normalize + cold-start detection)
        try:
            pincode = _normalize_pincode(features['pincode'])
            clean['pincode'] = pincode
        except ValueError as e:
            errors.append(str(e))

        # 3. Categoricals (canonicalize; unknown -> warning)
        try:
            clean['category'] = _canonicalize_categorical(
                'category', features['category'], warnings)
            clean['payment_method'] = _canonicalize_categorical(
                'payment_method', features['payment_method'], warnings)
            clean['courier_id'] = _canonicalize_categorical(
                'courier_id', features['courier_id'], warnings)
        except ValueError as e:
            errors.append(str(e))

        # 4. Numerics: coerce once, then bound-check BOTH sides
        for f in NUMERIC_FIELDS:
            try:
                x = _coerce_numeric(f, features[f])
                lo, hi = NUMERIC_BOUNDS[f]
                if x < lo:
                    errors.append(f"Field '{f}' is below the minimum allowed value of {lo} (got {x}).")
                elif x > hi:
                    errors.append(f"Field '{f}' exceeds maximum allowed value of {hi} (got {x}).")
                else:
                    clean[f] = x
            except ValueError as e:
                errors.append(str(e))

        # 5. Logical constraints (only meaningful when numerics parsed)
        pr = clean.get('prior_rto_count')
        po = clean.get('prior_orders')
        if pr is not None and po is not None and pr > po:
            errors.append("prior_rto_count cannot be greater than prior_orders.")
        age = clean.get('account_age_days')
        if age is not None and age < 0:  # unreachable via bounds; kept for message stability
            errors.append("account_age_days cannot be negative.")
        if age == 0 and po not in (None, 0):
            warnings.append(
                "account_age_days == 0 with prior_orders > 0 violates the data contract "
                "(new customers have no prior orders)."
            )

        # 6. Cross-field: COD charge only applies to COD orders (training contract)
        cc = clean.get('cod_charge')
        if cc is not None and clean.get('payment_method') != 'COD' and cc != 0:
            errors.append(
                f"cod_charge must be 0 for non-COD orders (got {cc}); "
                "non-COD orders carry no COD fee in training data."
            )

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

    return clean, warnings


def validate_features(features: dict):
    """Backward-compatible API: raises ValueError on invalid input."""
    validate_and_clean(features)


def _build_model_row(clean: dict, pin_info: dict) -> pd.DataFrame:
    """Assemble the single-row DataFrame the sklearn pipeline expects."""
    row_dict = {
        'category': clean['category'],
        'payment_method': clean['payment_method'],
        'courier_id': clean['courier_id'],
    }
    for f in NUMERIC_FIELDS:
        row_dict[f] = clean[f]
    row_dict['historical_pincode_rto_rate'] = float(pin_info['historical_pincode_rto_rate'])
    row_dict['pincode_tier'] = pin_info['pincode_tier']
    return pd.DataFrame([row_dict])


def _explain(df):
    """TreeSHAP top-3 drivers for a COD order (class-1 contributions)."""
    X_transformed = tree_uncal.named_steps['preprocessor'].transform(df)
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        shap_vals = np.asarray(sv[1])[0]      # older SHAP: [class0, class1] -> class 1
    elif getattr(sv, 'ndim', 2) == 3:
        shap_vals = sv[0, :, 1]               # new SHAP: (n, features, classes)
    else:
        shap_vals = np.asarray(sv)[0]         # single-output array
    feature_names = tree_uncal.named_steps['preprocessor'].get_feature_names_out()
    shap_dict = {feature_names[i]: float(shap_vals[i]) for i in range(len(feature_names))}
    return sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]


def score_order(features: dict):
    """
    Score a single order end-to-end.

    Returns dict with keys:
      probability, el_table, recommended_action, shap_top_factors,
      model_version, warnings
    Non-COD orders are passthroughs (zero EL, no SHAP) — consistent with
    CostEngine.get_optimal_policy.
    """
    clean, warnings = validate_and_clean(features)

    pincode = clean['pincode']
    pin_info, is_cold = resolve_pincode_info(pincode)
    if is_cold:
        warnings.append(
            f"Pincode '{pincode}' not in the train-derived lookup (cold start) — "
            f"using global prior rate {COLD_START_PRIOR['historical_pincode_rto_rate']:.4f} "
            f"and tier {COLD_START_PRIOR['pincode_tier']}."
        )

    df = _build_model_row(clean, pin_info)

    # Probability for every order (UPI/CC/DC/NB are in-vocabulary; PREPAID is OOD).
    p = float(tree_cal.predict_proba(df)[0, 1])

    payment = clean['payment_method']
    if not is_cod(payment):
        # Passthrough: no COD exposure — skip the cost engine and SHAP entirely.
        return {
            'probability': p,
            'el_table': {k: 0.0 for k in engine.interventions.keys()},
            'recommended_action': 'PREPAID_PASSTHROUGH',
            'shap_top_factors': [],
            'model_version': MODEL_VERSION,
            'warnings': warnings,
        }

    # COD decision: expected-loss argmin + SHAP explanation
    el_table = engine.evaluate_interventions(clean['order_value'], p)
    recommended_action = min(el_table, key=el_table.get)
    top_factors = _explain(df)

    return {
        'probability': p,
        'el_table': el_table,
        'recommended_action': recommended_action,
        'shap_top_factors': top_factors,
        'model_version': MODEL_VERSION,
        'warnings': warnings,
    }


def route_order(features: dict):
    res = score_order(features)
    return res['recommended_action']
