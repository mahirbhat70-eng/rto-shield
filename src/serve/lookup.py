"""
src/serve/lookup.py
WHY: Serving-time counterpart to src/data/lookup.py.
     Loads the pre-built pincode lookup CSV into memory and provides
     a resolve() function for use by score_order().

     Design: the lookup is loaded once at module import time (module-level singleton)
     for low-latency serving. The cold-start global prior is computed lazily
     from the same lookup, so it stays consistent with whatever training data
     built the lookup.
"""

import os
import pandas as pd

_LOOKUP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/processed/pincode_rate_lookup.csv")
)


def _load() -> tuple[dict, dict]:
    """Load lookup CSV and compute global prior. Returns (lookup_dict, global_prior)."""
    df = pd.read_csv(_LOOKUP_PATH, dtype={"pincode": str})
    lookup = df.set_index("pincode").to_dict(orient="index")
    global_prior = {
        "historical_pincode_rto_rate": float(df["historical_pincode_rto_rate"].mean()),
        "pincode_tier": int(df["pincode_tier"].mode().iloc[0]),
    }
    return lookup, global_prior


try:
    PINCODE_LOOKUP, GLOBAL_PRIOR = _load()
except FileNotFoundError:
    PINCODE_LOOKUP, GLOBAL_PRIOR = {}, {"historical_pincode_rto_rate": 0.28, "pincode_tier": 2}


def resolve(pincode: str) -> tuple[dict, bool]:
    """
    Resolve pincode to lookup features.

    Args:
        pincode: 6-digit string.

    Returns:
        (features_dict, is_cold_start): features_dict has keys
        historical_pincode_rto_rate and pincode_tier. is_cold_start=True
        means the pincode was unseen and the global prior was used.
    """
    if pincode in PINCODE_LOOKUP:
        return PINCODE_LOOKUP[pincode], False
    return GLOBAL_PRIOR, True
