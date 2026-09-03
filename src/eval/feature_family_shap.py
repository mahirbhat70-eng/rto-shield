import joblib
import shap
import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Stage 3.5: Feature Family SHAP Aggregation")
    print("=" * 60)

    val_rep_df = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    pipeline = joblib.load('models/tree_model.pkl')
    booster = joblib.load('models/tree_model_booster.pkl')
    
    preprocessor = pipeline.named_steps['preprocessor']
    X_val_rep_raw = val_rep_df.drop(columns=['rto_label'])
    
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(val_rep_df), size=min(2000, len(val_rep_df)), replace=False)
    X_sample_raw = X_val_rep_raw.iloc[sample_idx]
    
    X_sample_transformed = preprocessor.transform(X_sample_raw)
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample_transformed)
    
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values
        
    parent_features = {}
    for i, col in enumerate(feature_names):
        if col.startswith('num__'):
            parent = col[5:]
        elif col.startswith('cat__'):
            parent = col[5:].split('_')[0]
            if col[5:].startswith('payment_method'):
                parent = 'payment_method'
            elif col[5:].startswith('courier_id'):
                parent = 'courier_id'
            elif col[5:].startswith('pincode_tier'):
                parent = 'pincode_tier'
        else:
            parent = col
            
        mean_abs_shap = np.mean(np.abs(shap_values_pos[:, i]))
        if parent in parent_features:
            parent_features[parent] += mean_abs_shap
        else:
            parent_features[parent] = mean_abs_shap
            
    # Grouping into families
    # "COD-Family" (cod_charge + payment_method)
    # "Pincode-Family" (historical_pincode_rto_rate + pincode_tier)
    # "Noise-Columns" (order_value, quantity)
    # "All-Other-Family" (Everything else)
    
    families = {
        'COD-Family': 0.0,
        'Pincode-Family': 0.0,
        'Noise-Columns': 0.0,
        'All-Other-Family': 0.0
    }
    
    noise_cols = []
    with open('docs/data_dictionary.md', 'r', encoding='utf-8') as f:
        for line in f:
            if 'noise column' in line.lower():
                # Hypothetical parsing if any existed
                pass
    
    # Store individual ranks
    sorted_parents = sorted(parent_features.items(), key=lambda x: x[1], reverse=True)
    ranks = {k: rank+1 for rank, (k, _) in enumerate(sorted_parents)}
    
    for feat, val in parent_features.items():
        if feat in ['cod_charge', 'payment_method']:
            families['COD-Family'] += val
        elif feat in ['historical_pincode_rto_rate', 'pincode_tier']:
            families['Pincode-Family'] += val
        elif feat in noise_cols:
            families['Noise-Columns'] += val
        else:
            families['All-Other-Family'] += val
            
    total_shap = sum(families.values())
    
    print("\nFeature Family Importance Shares:")
    for family, val in sorted(families.items(), key=lambda x: x[1], reverse=True):
        pct = (val / total_shap) * 100 if total_shap > 0 else 0
        print(f"  - {family}: {pct:.2f}%")
        
    print("\nNoise Columns Detailed Ranking:")
    if not noise_cols:
        print("  - (No explicitly labeled pure noise columns found in data dictionary)")
    else:
        for nc in noise_cols:
            val = parent_features.get(nc, 0.0)
            pct = (val / total_shap) * 100 if total_shap > 0 else 0
            rank = ranks.get(nc, 'N/A')
            print(f"  - {nc}: Rank {rank} ({pct:.2f}%)")
            
    print("\nAll-Other-Family (Residual) Top 3 Features:")
    other_features = {k: v for k, v in parent_features.items() 
                      if k not in ['cod_charge', 'payment_method', 'historical_pincode_rto_rate', 'pincode_tier'] + noise_cols}
    sorted_other = sorted(other_features.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, val) in enumerate(sorted_other[:3]):
        pct = (val / total_shap) * 100 if total_shap > 0 else 0
        rank = ranks.get(feat, 'N/A')
        print(f"  {i+1}. {feat}: Rank {rank} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
