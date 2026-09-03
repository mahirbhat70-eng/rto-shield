"""
RTO Shield -- Stage 2: Evaluation

Evaluates rule baseline and logistic regression baseline on val and test splits.
Reports: Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix.
Saves comparison table to reports/stage2_baseline_results.md
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, roc_auc_score
)

# Import project modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.rule_baseline import derive_thresholds, predict_rules
from src.models.logistic_baseline import train_model, predict, print_top_coefficients


def evaluate_predictions(y_true, y_pred, y_proba=None, model_name="", split_name=""):
    """Compute and return metrics dict."""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'model': model_name,
        'split': split_name,
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'flag_rate': round(np.mean(y_pred), 4),
        'pr_auc': None,
        'pr_auc_prov': '',
        'roc_auc': None,
        'prec_at_rule_recall': None,
    }

    if y_proba is not None:
        metrics['pr_auc'] = round(average_precision_score(y_true, y_proba), 4)
        metrics['pr_auc_prov'] = 'sklearn'
        metrics['roc_auc'] = round(roc_auc_score(y_true, y_proba), 4)
    else:
        # For a binary scorer (rule baseline), PR-AUC can be approximated using closed form:
        # AP = R*P + (1-R)*prevalence
        prevalence = np.mean(y_true)
        ap_approx = (rec * prec) + ((1 - rec) * prevalence)
        ap_sklearn = average_precision_score(y_true, y_pred)
        assert abs(ap_approx - ap_sklearn) < 1e-9, "Closed-form AP mismatch"
        metrics['pr_auc'] = round(ap_approx, 4)
        metrics['pr_auc_prov'] = 'implied (closed-form, binary scorer)'

    return metrics, cm


def print_results(metrics, cm, label=""):
    """Print metrics and confusion matrix."""
    print(f"\n  {label}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    if metrics['pr_auc'] is not None:
        print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"      FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    print(f"    FP Rate: {fpr:.4f}")


def save_results_table(all_metrics, output_path="reports/stage2_baseline_results.md"):
    """Save comparison table as markdown."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Stage 2 Baseline Results",
        "",
        "> Default 0.5 threshold is a placeholder. Real operating threshold",
        "> will be chosen in a later stage via the cost matrix.",
        "",
        "| Model | Split | Flag Rate | Precision | Recall | F1 | PR-AUC | PR-AUC Provenance | ROC-AUC | Prec @ Rule Recall |",
        "|-------|-------|-----------|-----------|--------|----|--------|-------------------|---------|-------------------|",
    ]
    for m in all_metrics:
        pr_auc = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        roc_auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        p_at_r = f"{m['prec_at_rule_recall']:.4f}" if m.get('prec_at_rule_recall') is not None else "N/A"
        lines.append(
            f"| {m['model']} | {m['split']} | {m['flag_rate']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {pr_auc} | {m['pr_auc_prov']} | {roc_auc} | {p_at_r} |"
        )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nResults table saved to: {output_path}")


def main():
    print("=" * 60)
    print("RTO Shield -- Stage 2 Evaluation")
    print("=" * 60)

    # Load splits
    train_df = pd.read_csv("data/processed/train.csv", dtype={'pincode': str})
    val_df = pd.read_csv("data/processed/val.csv", dtype={'pincode': str})
    test_df = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})

    all_metrics = []

    # ── 1. Rule Baseline ──────────────────────────────────────────
    print("\n--- Rule Baseline ---")
    thresholds = derive_thresholds(train_df)
    print("Thresholds (train-derived):", thresholds)

    rule_recalls = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        preds = predict_rules(df, thresholds)
        y_true = df['rto_label'].values
        # Rule baseline has no probability output, so PR-AUC/ROC-AUC = N/A
        metrics, cm = evaluate_predictions(
            y_true, preds, y_proba=None,
            model_name="Rule Baseline", split_name=split_name
        )
        rule_recalls[split_name] = metrics['recall']
        all_metrics.append(metrics)
        print_results(metrics, cm, label=f"Rule Baseline ({split_name})")

    # ── 2. Logistic Regression ────────────────────────────────────
    print("\n--- Logistic Regression Baseline ---")
    pipeline = train_model()

    for split_name, df in [("val", val_df), ("test", test_df)]:
        preds, proba = predict(pipeline, df)
        y_true = df['rto_label'].values
        metrics, cm = evaluate_predictions(
            y_true, preds, y_proba=proba,
            model_name="Logistic Regression", split_name=split_name
        )
        # Interpolate LR precision at rule recall
        from sklearn.metrics import precision_recall_curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, proba)
        rule_rec = rule_recalls[split_name]
        p_at_r = np.interp(rule_rec, rec_curve[:-1][::-1], prec_curve[:-1][::-1])
        metrics['prec_at_rule_recall'] = round(float(p_at_r), 4)

        all_metrics.append(metrics)
        print_results(metrics, cm, label=f"Logistic Regression ({split_name})")

    # ── 3. Coefficient sanity check ───────────────────────────────
    feature_names, coefs = print_top_coefficients(pipeline)

    # ── 4. Split balance report ───────────────────────────────────
    print("\n--- Split RTO Rate Summary ---")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = df['rto_label'].mean()
        ts = pd.to_datetime(df['timestamp'])
        print(f"  {name:5s}: {len(df):>6,} rows | "
              f"{ts.min().date()} -> {ts.max().date()} | "
              f"RTO rate: {rate:.4f}")

    # ── 5. Save results ──────────────────────────────────────────
    save_results_table(all_metrics)

    print("\n" + "=" * 60)
    print("Stage 2 Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
