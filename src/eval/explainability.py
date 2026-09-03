import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Stage 3: SHAP Explainability")
    print("=" * 60)

    val_rep_df = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    pipeline = joblib.load('models/tree_model.pkl')
    booster = joblib.load('models/tree_model_booster.pkl')
    
    # Preprocess
    preprocessor = pipeline.named_steps['preprocessor']
    X_val_rep_raw = val_rep_df.drop(columns=['rto_label'])
    y_val_rep = val_rep_df['rto_label'].values
    
    # Sample 2000 rows for SHAP
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(val_rep_df), size=min(2000, len(val_rep_df)), replace=False)
    
    X_sample_raw = X_val_rep_raw.iloc[sample_idx]
    y_sample = y_val_rep[sample_idx]
    
    X_sample_transformed = preprocessor.transform(X_sample_raw)
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample_transformed)
    
    # If lightgbm binary classification, shap_values is sometimes a list of arrays (one per class) 
    # or a single array (for the positive class, depending on version).
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values
        
    os.makedirs('reports/stage3', exist_ok=True)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values_pos, X_sample_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('reports/stage3/shap_summary.png')
    plt.close()
    
    # Text reporting: aggregate one-hot encoded columns back to parent features
    parent_features = {}
    
    # Clean feature names from column transformer e.g. "num__historical_pincode_rto_rate", "cat__category_Electronics"
    for i, col in enumerate(feature_names):
        # determine parent feature
        if col.startswith('num__'):
            parent = col[5:]
        elif col.startswith('cat__'):
            # e.g., cat__category_Electronics -> category
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
            
    total_shap = sum(parent_features.values())
    sorted_parents = sorted(parent_features.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop Parent Features by Mean |SHAP|:")
    for rank, (feat, val) in enumerate(sorted_parents[:10], 1):
        pct = (val / total_shap) * 100 if total_shap > 0 else 0
        print(f"{rank}. {feat}: {pct:.2f}%")
        
    # Waterfall plots for high/low risk
    proba_sample = pipeline.predict_proba(X_sample_raw)[:, 1]
    
    high_risk_idx = np.where(proba_sample > 0.7)[0]
    low_risk_idx = np.where(proba_sample < 0.2)[0]
    
    if len(high_risk_idx) == 0:
        print("No prediction > 0.7 found. Using the highest risk prediction in sample instead.")
        high_risk_idx = [np.argmax(proba_sample)]
        
    if len(high_risk_idx) > 0:
        high_idx = high_risk_idx[0]
        # In modern SHAP, waterfall plot requires an Explanation object
        exp_high = shap.Explanation(
            values=shap_values_pos[high_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample_transformed[high_idx],
            feature_names=feature_names
        )
        plt.figure()
        shap.plots.waterfall(exp_high, show=False)
        plt.tight_layout()
        plt.savefig('reports/stage3/shap_waterfall_high.png')
        plt.close()
        print("Saved shap_waterfall_high.png")
    else:
        print("No prediction > 0.7 found in sample for high risk waterfall plot.")
        
    if len(low_risk_idx) > 0:
        low_idx = low_risk_idx[0]
        exp_low = shap.Explanation(
            values=shap_values_pos[low_idx],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample_transformed[low_idx],
            feature_names=feature_names
        )
        plt.figure()
        shap.plots.waterfall(exp_low, show=False)
        plt.tight_layout()
        plt.savefig('reports/stage3/shap_waterfall_low.png')
        plt.close()
        print("Saved shap_waterfall_low.png")
    else:
        print("No prediction < 0.2 found in sample for low risk waterfall plot.")

if __name__ == "__main__":
    main()
