import joblib
import pandas as pd
import numpy as np

def main():
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    
    mask = val_rep['payment_method'] == 'COD'
    cod_subset = val_rep[mask].copy().reset_index(drop=True)
    
    median_val = cod_subset['order_value'].median()
    print(f"Median order_value of COD subset: {median_val:.2f}")
    
    # Evaluate Primary Policy on COD subset
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.policy.cost_engine import CostEngine
    from src.eval.stage4_evaluate import eval_multi_action
    
    engine = CostEngine()
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    proba_cod = tree_cal.predict_proba(cod_subset.drop(columns=['rto_label']))[:, 1]
    
    actions, _ = eval_multi_action(cod_subset, proba_cod, engine)
    cod_subset['action'] = actions
    
    cod_subset['quartile'] = pd.qcut(cod_subset['order_value'], q=4)
    
    print("\nDEPOSIT share within each order_value quartile:")
    for q, group in cod_subset.groupby('quartile', observed=False):
        deposit_share = (group['action'] == 'REQUIRE_DEPOSIT').mean() * 100
        print(f"  {q}: {deposit_share:.1f}%")

if __name__ == "__main__":
    main()
