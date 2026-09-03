import os
import pandas as pd

def build_lookup():
    train_df = pd.read_csv("data/processed/train.csv", dtype={'pincode': str})
    
    # We want unique mapping of pincode to historical_pincode_rto_rate and pincode_tier.
    # Grouping by pincode and taking the first should be enough since they are fixed per pincode
    lookup = train_df.groupby('pincode', as_index=False).agg({
        'historical_pincode_rto_rate': 'first',
        'pincode_tier': 'first'
    })
    
    os.makedirs('data/processed', exist_ok=True)
    lookup.to_csv("data/processed/pincode_rate_lookup.csv", index=False)
    print(f"Pincode lookup built with {len(lookup)} rows.")

if __name__ == "__main__":
    build_lookup()
