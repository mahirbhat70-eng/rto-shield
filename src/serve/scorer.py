import os
import joblib
import numpy as np
import pandas as pd
import yaml
import shap
import sys

# Load models and configs at module level to simulate a serving environment
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.policy.cost_engine import CostEngine

# Load lookup table
lookup_df = pd.read_csv(os.path.join(DATA_DIR, 'pincode_rate_lookup.csv'), dtype={'pincode': str})
PINCODE_LOOKUP = lookup_df.set_index('pincode').to_dict(orient='index')

# Load models
tree_uncal = joblib.load(os.path.join(MODEL_DIR, 'tree_model.pkl')) # contains encoder pipeline
tree_cal = joblib.load(os.path.join(MODEL_DIR, 'tree_model_calibrated.pkl'))
booster = joblib.load(os.path.join(MODEL_DIR, 'tree_model_booster.pkl'))
explainer = shap.TreeExplainer(booster)

# Initialize Cost Engine
engine = CostEngine()

REQUIRED_INPUTS = [
    'order_value', 'quantity', 'category', 'discount_pct', 'payment_method',
    'cod_charge', 'account_age_days', 'prior_orders', 'prior_rto_count', 
    'pincode', 'courier_id', 'orders_last_24h', 'device_cluster_size'
]

def validate_features(features: dict):
    errors = []
    # 1. Missing fields
    missing = [f for f in REQUIRED_INPUTS if f not in features]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
    
    # Check extra fields (for lookup)
    if 'historical_pincode_rto_rate' in features or 'pincode_tier' in features:
        errors.append("Do not provide historical_pincode_rto_rate or pincode_tier; these are looked up via pincode.")

    # Only do further checks if required fields are present
    if not missing:
        # 2. Pincode exists in lookup
        pincode = str(features.get('pincode'))
        if pincode not in PINCODE_LOOKUP:
            errors.append(f"Pincode '{pincode}' not found in lookup table.")
            
        # 3. NaNs in numeric features
        numeric_fields = ['order_value', 'quantity', 'discount_pct', 'cod_charge', 
                          'account_age_days', 'prior_orders', 'prior_rto_count', 
                          'orders_last_24h', 'device_cluster_size']
        for f in numeric_fields:
            val = features.get(f)
            if val is None or pd.isna(val):
                errors.append(f"Field '{f}' must not be NaN or None.")
            else:
                try:
                    float(val)
                except ValueError:
                    errors.append(f"Field '{f}' must be numeric.")
                    
        # 4. Logical constraints and Guardrails
        if features.get('prior_rto_count', 0) > features.get('prior_orders', 0):
            errors.append("prior_rto_count cannot be greater than prior_orders.")
        
        if features.get('account_age_days', 0) < 0:
            errors.append("account_age_days cannot be negative.")
            
        # Upper bounds (P99.9 guardrails)
        bounds = {
            'order_value': 25000,
            'quantity': 50,
            'discount_pct': 100,
            'cod_charge': 500,
            'account_age_days': 5000,
            'prior_orders': 100,
            'orders_last_24h': 50,
            'device_cluster_size': 50
        }
        for f, m_val in bounds.items():
            if features.get(f, 0) > m_val:
                errors.append(f"{f} exceeds maximum allowed value of {m_val}.")

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))
        
def score_order(features: dict):
    validate_features(features)
    
    # 1. Feature Assembly
    pincode = str(features['pincode'])
    lookup_info = PINCODE_LOOKUP[pincode]
    
    # Explicit column hygiene - only take required inputs (excluding pincode which is replaced by tier/rate)
    row_dict = {
        'category': features['category'],
        'payment_method': features['payment_method'],
        'courier_id': features['courier_id']
    }
    
    numeric_fields = ['order_value', 'quantity', 'discount_pct', 'cod_charge', 
                      'account_age_days', 'prior_orders', 'prior_rto_count', 
                      'orders_last_24h', 'device_cluster_size']
    for f in numeric_fields:
        row_dict[f] = float(features[f])
        
    row_dict['historical_pincode_rto_rate'] = float(lookup_info['historical_pincode_rto_rate'])
    row_dict['pincode_tier'] = lookup_info['pincode_tier']
    
    # Create DataFrame (models expect pandas DF)
    # Exclude non-model features
    df = pd.DataFrame([row_dict])
    # The pipeline in tree_uncal handles preprocessing, dropping 'pincode' etc.
    
    # 2. Score Probability
    p = tree_cal.predict_proba(df)[0, 1]
    
    # 3. Cost Engine EL
    el_table = engine.evaluate_interventions(features['order_value'], p)
    recommended_action = min(el_table, key=el_table.get)
    
    # 4. SHAP Explanation
    # We must run the encoder part of the pipeline to get raw features for SHAP
    # The pipeline step 'preprocessor' handles scaling/encoding
    X_transformed = tree_uncal.named_steps['preprocessor'].transform(df)
    
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        shap_vals = np.asarray(sv[1])[0]      # older SHAP: [class0, class1] → class 1
    elif getattr(sv, 'ndim', 2) == 3:
        shap_vals = sv[0, :, 1]               # new SHAP: (n, features, classes)
    else:
        shap_vals = np.asarray(sv)[0]         # single-output array
    
    # Get feature names from preprocessor
    feature_names = tree_uncal.named_steps['preprocessor'].get_feature_names_out()
    
    # Pair and sort
    shap_dict = {feature_names[i]: float(shap_vals[i]) for i in range(len(feature_names))}
    top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    if features.get('payment_method') == 'PREPAID':
        return {
            'probability': float(p),
            'el_table': {k: 0.0 for k in el_table.keys()},
            'recommended_action': 'PREPAID_PASSTHROUGH',
            'shap_top_factors': top_factors
        }
    
    return {
        'probability': float(p),
        'el_table': el_table,
        'recommended_action': recommended_action,
        'shap_top_factors': top_factors
    }

def route_order(features: dict):
    res = score_order(features)
    return res['recommended_action']
