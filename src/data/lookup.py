"""
src/data/lookup.py
WHY: Builds the pincode → (historical_rto_rate, tier) lookup table used at serving time.
     This MUST be built from TRAIN data only — using val/test data here would leak
     future label information into the feature pipeline (target leakage).

     The lookup is stored as a CSV and loaded at model-serving time. Unknown pincodes
     trigger a cold-start fallback (global prior) — see src/serve/scorer.py.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROCESSED_DIR = "data/processed"


def build_lookup(train_path: str = None, output_path: str = None) -> pd.DataFrame:
    """
    Build the pincode rate lookup from TRAIN split only.

    Uses the per-pincode MEAN of historical_pincode_rto_rate (stored in the data)
    rather than 'first', which would introduce arbitrary row-ordering noise since
    each row stores a slightly different rolling rate snapshot.

    Args:
        train_path:  Path to train.csv. Defaults to data/processed/train.csv.
        output_path: Where to write lookup CSV. Defaults to data/processed/pincode_rate_lookup.csv.

    Returns:
        DataFrame with columns: pincode, historical_pincode_rto_rate, pincode_tier.
    """
    if train_path is None:
        train_path = os.path.join(PROCESSED_DIR, "train.csv")
    if output_path is None:
        output_path = os.path.join(PROCESSED_DIR, "pincode_rate_lookup.csv")

    train = pd.read_csv(train_path, dtype={"pincode": str})

    lookup = (
        train.groupby("pincode", as_index=False)
        .agg(
            historical_pincode_rto_rate=("historical_pincode_rto_rate", "mean"),
            pincode_tier=("pincode_tier", "first"),
        )
    )
    lookup["historical_pincode_rto_rate"] = lookup["historical_pincode_rto_rate"].round(4)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    lookup.to_csv(output_path, index=False)

    print(f"Pincode lookup: {len(lookup)} pincodes -> {output_path}")
    print(f"  Rate range: [{lookup['historical_pincode_rto_rate'].min():.4f}, "
          f"{lookup['historical_pincode_rto_rate'].max():.4f}]")
    return lookup


# Global prior (computed lazily from the lookup at serving time — see scorer.py)
def compute_global_prior(lookup_df: pd.DataFrame) -> dict:
    """Return the cold-start fallback: mean rate + modal tier."""
    return {
        "historical_pincode_rto_rate": float(lookup_df["historical_pincode_rto_rate"].mean()),
        "pincode_tier": int(lookup_df["pincode_tier"].mode().iloc[0]),
    }


if __name__ == "__main__":
    build_lookup()
