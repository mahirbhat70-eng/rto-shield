"""
lookup.py — Rebuild the pincode feature lookup from TRAIN rows only.

Change vs v1 (see IMPROVEMENTS.md): the historical RTO rate now uses the
TRAIN MEAN per pincode instead of the first observed row. Each pincode has
~70 train rows; averaging cuts the per-pincode rate noise from sigma~0.03
(single draw) to sigma~0.004, which is what a production estimator would do.
Tier is constant per pincode, so 'first' remains correct for it.
Cold pincodes (never seen in train) are NOT added here — the scorer falls
back to a global prior at serving time (see scorer.resolve_pincode_info).
"""

import os
import pandas as pd

DEFAULT_TRAIN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'train.csv')
)
DEFAULT_OUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'pincode_rate_lookup.csv')
)


def build_lookup(train_path=DEFAULT_TRAIN_PATH, out_path=DEFAULT_OUT_PATH):
    train_df = pd.read_csv(train_path, dtype={'pincode': str})

    # Mean over train rows per pincode (noise-averaged); tier is fixed per pincode.
    lookup = train_df.groupby('pincode', as_index=False).agg({
        'historical_pincode_rto_rate': 'mean',
        'pincode_tier': 'first'
    })
    lookup['historical_pincode_rto_rate'] = lookup['historical_pincode_rto_rate'].round(4)

    # Sanity: rate must remain a valid probability and tier in {1,2,3}
    assert (lookup['historical_pincode_rto_rate'].between(0, 1)).all(), "rate outside [0,1]"
    assert lookup['pincode_tier'].isin([1, 2, 3]).all(), "tier outside {1,2,3}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lookup.to_csv(out_path, index=False)
    print(f"Pincode lookup built with {len(lookup)} rows (train-mean rate estimator).")
    return lookup


if __name__ == "__main__":
    build_lookup()
