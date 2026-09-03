import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_binary_policy, eval_multi_action, calc_operational_metrics

test = pd.read_csv('data/processed/test.csv')
tree_cal = joblib.load('models/tree_model_calibrated.pkl')
p_cal = tree_cal.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
test_cod, p_cal_cod = get_cod_subset(test, p_cal)

engine = CostEngine()

prepaid_actions, _ = eval_binary_policy(test_cod, p_cal_cod, engine, 'PREPAID_ONLY', 0.48)
verify_actions, _ = eval_binary_policy(test_cod, p_cal_cod, engine, 'VERIFY_ADDRESS', 0.20)
primary_actions, _ = eval_multi_action(test_cod, p_cal_cod, engine)

def get_metrics(name, actions):
    t, prev, drops, fric = calc_operational_metrics(test_cod, p_cal_cod, actions, engine)
    print(f'{name}: Prev {prev:.2f}, Drops {drops:.2f}')

get_metrics('Binary PREPAID', prepaid_actions)
get_metrics('Binary VERIFY', verify_actions)
get_metrics('Primary', primary_actions)

print(f'COD Subset RTO Rate: {test_cod["rto_label"].mean():.4f}')
