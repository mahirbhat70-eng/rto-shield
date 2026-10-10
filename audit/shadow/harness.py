"""
audit/shadow/harness.py — Shadow-Mode Evaluation Harness for Real Merchant Data.

Processes merchant CSV exports, runs them through the production serving scorer,
and generates an executive-ready shadow validation report.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.serve.scorer import score_order, COLD_START_PRIOR
from src.policy.cost_engine import CostEngine

def run_shadow_replay(csv_path: str, output_report: str = "audit/shadow/shadow_report.md"):
    print(f">> Loading merchant export: {csv_path}")
    df = pd.read_csv(csv_path, dtype={'pincode': str})
    
    total_orders = len(df)
    cod_mask = df['payment_method'].str.upper() == 'COD'
    cod_orders = int(cod_mask.sum())
    cod_share = cod_orders / total_orders if total_orders > 0 else 0
    
    has_labels = 'rto_label' in df.columns or 'final_status' in df.columns
    if 'final_status' in df.columns and 'rto_label' not in df.columns:
        df['rto_label'] = df['final_status'].apply(lambda x: 1 if str(x).upper() in ['RTO', 'RETURNED', 'REJECTED'] else 0)
    
    observed_rto_rate = float(df[cod_mask]['rto_label'].mean()) if has_labels and cod_orders > 0 else None
    
    # Process orders
    actions = []
    probabilities = []
    warnings_list = []
    cold_start_count = 0
    
    for i, row in df.iterrows():
        order_dict = {
            'order_value': float(row.get('order_value', 500.0)),
            'quantity': int(row.get('quantity', 1)),
            'category': str(row.get('category', 'Apparel')),
            'discount_pct': float(row.get('discount_pct', 0.0)),
            'payment_method': str(row.get('payment_method', 'COD')),
            'cod_charge': float(row.get('cod_charge', 49.0 if row.get('payment_method') == 'COD' else 0.0)),
            'account_age_days': int(row.get('account_age_days', 90)),
            'prior_orders': int(row.get('prior_orders', 1)),
            'prior_rto_count': int(row.get('prior_rto_count', 0)),
            'orders_last_24h': int(row.get('orders_last_24h', 0)),
            'device_cluster_size': int(row.get('device_cluster_size', 1)),
            'pincode': str(row.get('pincode', '110001')).zfill(6),
            'courier_id': str(row.get('courier_id', 'Courier_A'))
        }
        
        try:
            res = score_order(order_dict)
            actions.append(res['recommended_action'])
            probabilities.append(res['probability'])
            if any('cold start' in str(w).lower() for w in res['warnings']):
                cold_start_count += 1
            if res['warnings']:
                warnings_list.extend(res['warnings'])
        except Exception as e:
            actions.append("ERROR_ALLOW")
            probabilities.append(COLD_START_PRIOR['historical_pincode_rto_rate'])
            warnings_list.append(f"Scoring error on row {i}: {e}")

    # Action distribution
    action_series = pd.Series(actions)
    act_counts = action_series.value_counts()
    
    # Financial projection on COD
    engine = CostEngine()
    total_savings_el = 0.0
    cod_indices = np.where(cod_mask)[0]
    for idx in cod_indices:
        row = df.iloc[idx]
        V = float(row.get('order_value', 500.0))
        p = probabilities[idx]
        act = actions[idx]
        losses = engine.evaluate_interventions(V, p)
        base_el = losses['ALLOW_COD']
        pol_el = losses.get(act, base_el)
        total_savings_el += max(0.0, base_el - pol_el)
        
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("# RTO Shield — Shadow Mode Evaluation Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Orders Processed:** {total_orders:,}\n")
        f.write(f"- **COD Order Volume:** {cod_orders:,} ({cod_share*100:.1f}%)\n")
        if observed_rto_rate is not None:
            f.write(f"- **Historical COD RTO Rate:** {observed_rto_rate*100:.1f}%\n")
        f.write(f"- **Cold-Start Pincode Rate:** {cold_start_count / max(1, total_orders) * 100:.1f}% ({cold_start_count:,} orders)\n")
        f.write(f"- **Projected Net Savings (EL):** Rs {total_savings_el:,.2f}\n")
        f.write(f"- **Estimated Savings Per COD Order:** Rs {total_savings_el / max(1, cod_orders):.2f}\n\n")
        
        f.write("## 2. Policy Action Distribution (COD Orders)\n")
        f.write("| Action | Count | Percentage |\n|---|---|---|\n")
        for a, cnt in act_counts.items():
            f.write(f"| `{a}` | {cnt:,} | {cnt / total_orders * 100:.1f}% |\n")
        f.write("\n")
        
        f.write("## 3. Data Quality & Safety Warnings\n")
        f.write(f"- Total warnings emitted: {len(warnings_list):,}\n")
        for w in list(set(warnings_list))[:10]:
            f.write(f"  - `{w}`\n")
        f.write("\n")
        
    print(f">> Shadow report written to {output_report}")
    return output_report

if __name__ == "__main__":
    # Test harness on sample test slice
    print("Testing shadow harness on 500 rows of test.csv...")
    test_slice = pd.read_csv("data/processed/test.csv", dtype={'pincode': str}).head(500)
    test_slice.to_csv("audit/shadow/sample_merchant_input.csv", index=False)
    run_shadow_replay("audit/shadow/sample_merchant_input.csv")
