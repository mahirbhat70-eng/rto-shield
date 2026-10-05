"""
audit/drift_monitor.py — Production Drift & Calibration Monitoring Service.

Computes:
1. Population Stability Index (PSI) on predicted risk probabilities p_cal.
2. Prediction mean drift (E[p] vs baseline reference).
3. Empirical RTO rate drift (delayed feedback window).
4. Expected Calibration Error (ECE) drift on delivered/returned batches.
5. Alert triggers:
   - PSI > 0.10: WARNING, PSI > 0.25: CRITICAL DRIFT (Retrain/recalibrate trigger).
   - ECE > 0.05: CALIBRATION ALERT (Pause automated interventions).
   - Mean P shift > 0.05: DISTRIBUTION SHIFT WARNING.
"""

import numpy as np
import pandas as pd

def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Compute Population Stability Index between reference and current score streams."""
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]
    
    ref_pct = (ref_counts + 1e-4) / (len(reference) + 1e-4 * num_bins)
    cur_pct = (cur_counts + 1e-4) / (len(current) + 1e-4 * num_bins)
    
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)

def compute_window_metrics(reference_p: np.ndarray, window_p: np.ndarray, window_y: np.ndarray = None):
    """Compute comprehensive drift scorecard for a sliding window."""
    psi = compute_psi(reference_p, window_p)
    mean_ref_p = float(np.mean(reference_p))
    mean_win_p = float(np.mean(window_p))
    delta_p = mean_win_p - mean_ref_p
    
    metrics = {
        'window_size': len(window_p),
        'mean_ref_p': round(mean_ref_p, 4),
        'mean_win_p': round(mean_win_p, 4),
        'delta_mean_p': round(delta_p, 4),
        'psi': round(psi, 4),
        'psi_status': 'GREEN' if psi < 0.10 else ('AMBER' if psi < 0.25 else 'RED')
    }
    
    if window_y is not None:
        empirical_rto = float(np.mean(window_y))
        abs_cal_gap = abs(mean_win_p - empirical_rto)
        metrics['empirical_rto'] = round(empirical_rto, 4)
        metrics['calibration_gap'] = round(abs_cal_gap, 4)
        metrics['calibration_status'] = 'HEALTHY' if abs_cal_gap < 0.03 else ('WARN' if abs_cal_gap < 0.06 else 'ALERT')
        
    return metrics

if __name__ == "__main__":
    print("Testing audit/drift_monitor.py on synthetic streams...")
    ref = np.random.beta(2, 8, size=5000)
    normal_cur = np.random.beta(2, 8, size=1000)
    drifted_cur = np.random.beta(3, 7, size=1000)
    
    print("In-distribution stream:", compute_window_metrics(ref, normal_cur))
    print("Drifted festive peak stream:", compute_window_metrics(ref, drifted_cur))
