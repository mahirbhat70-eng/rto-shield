import os
import joblib
import pandas as pd
import numpy as np

def main():
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.policy.cost_engine import CostEngine
    from src.eval.stage4_evaluate import eval_multi_action, get_cod_subset
    
    val_rep = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})
    tree_cal = joblib.load('models/tree_model_calibrated.pkl')
    proba_cod = tree_cal.predict_proba(val_rep[val_rep['payment_method'] == 'COD'].drop(columns=['rto_label']))[:, 1]
    
    val_rep_cod, _ = get_cod_subset(val_rep, np.zeros(len(val_rep)))
    
    print("=" * 60)
    print("DEPOSIT Drop-Rate Sensitivity Analysis (Primary Policy)")
    print("=" * 60)
    
    drop_pcts = [0.25, 0.40, 0.50]
    
    for dp in drop_pcts:
        engine = CostEngine()
        # Override the config
        engine.interventions['REQUIRE_DEPOSIT']['success_drop_pct'] = dp
        
        actions, losses = eval_multi_action(val_rep_cod, proba_cod, engine)
        
        total_loss = np.sum(losses)
        deposit_share = np.mean(actions == 'REQUIRE_DEPOSIT') * 100
        
        print(f"REQUIRE_DEPOSIT success_drop_pct: {dp:.2f}")
        print(f"  - Total Loss (INR): {total_loss:.2f}")
        print(f"  - DEPOSIT Share (COD Orders): {deposit_share:.1f}%\n")

if __name__ == "__main__":
    main()
