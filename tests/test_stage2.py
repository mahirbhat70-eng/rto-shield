"""
RTO Shield -- Stage 2 Tests

Tests for temporal split, rule baseline, logistic baseline, and evaluation.
Uses small synthetic samples -- does not depend on the 100k CSV.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    """Generate a small dataset mimicking the 19-column schema."""
    rng = np.random.default_rng(42)
    n = 300
    timestamps = pd.date_range("2026-03-01", periods=n, freq="6h")
    df = pd.DataFrame({
        'order_id': [f"ORD{i:09d}" for i in range(n)],
        'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S'),
        'order_value': rng.lognormal(6.4, 0.8, n).clip(50, 15000).round(2),
        'quantity': rng.poisson(1.2, n) + 1,
        'category': rng.choice(['Electronics', 'Apparel', 'Footwear'], n),
        'discount_pct': (rng.beta(2, 5, n) * 70).round(2),
        'payment_method': rng.choice(['COD', 'UPI', 'Credit Card'], n, p=[0.5, 0.3, 0.2]),
        'cod_charge': np.zeros(n),
        'customer_id': [f"CUST{rng.integers(1, 100):06d}" for _ in range(n)],
        'account_age_days': rng.integers(0, 1000, n),
        'prior_orders': rng.poisson(3, n),
        'prior_rto_count': np.zeros(n, dtype=int),
        'pincode': [f"{rng.integers(100000, 999999)}" for _ in range(n)],
        'courier_id': rng.choice(['Courier_A', 'Courier_B', 'Courier_C'], n),
        'pincode_tier': rng.choice([1, 2, 3], n),
        'historical_pincode_rto_rate': rng.beta(2, 8, n).round(4),
        'orders_last_24h': rng.poisson(1.5, n),
        'device_cluster_size': rng.integers(1, 5, n),
        'rto_label': rng.binomial(1, 0.20, n),
    })
    # Fix constraints
    df['prior_rto_count'] = np.minimum(
        rng.poisson(0.5, n), df['prior_orders']
    ).astype(int)
    cod_mask = df['payment_method'] == 'COD'
    df.loc[cod_mask, 'cod_charge'] = rng.choice([29.0, 49.0, 79.0], cod_mask.sum())
    return df


# ── Test: Temporal Split ──────────────────────────────────────────────

def test_temporal_split_no_overlap(small_dataset, tmp_path):
    """Split produces non-overlapping temporal ranges."""
    # Save small dataset
    input_path = str(tmp_path / "test_data.csv")
    small_dataset.to_csv(input_path, index=False)

    from src.data.split import temporal_split
    train, val, test = temporal_split(
        input_path=input_path,
        output_dir=str(tmp_path / "processed")
    )

    train_max = pd.to_datetime(train['timestamp']).max()
    val_min = pd.to_datetime(val['timestamp']).min()
    val_max = pd.to_datetime(val['timestamp']).max()
    test_min = pd.to_datetime(test['timestamp']).min()

    assert train_max < val_min, "Train/val temporal overlap"
    assert val_max < test_min, "Val/test temporal overlap"
    assert len(train) + len(val) + len(test) == len(small_dataset)


# ── Test: Rule Baseline ──────────────────────────────────────────────

def test_rule_baseline_binary_predictions(small_dataset):
    """Rule baseline returns binary predictions without needing rto_label."""
    from src.models.rule_baseline import derive_thresholds, predict_rules

    thresholds = derive_thresholds(small_dataset)
    # Predict on data without rto_label column
    df_no_label = small_dataset.drop(columns=['rto_label'])
    preds = predict_rules(df_no_label, thresholds)

    assert set(np.unique(preds)).issubset({0, 1}), "Predictions not binary"
    assert len(preds) == len(small_dataset)


# ── Test: Logistic Baseline ──────────────────────────────────────────

def test_logistic_pipeline_fit_transform(small_dataset):
    """Preprocessing pipeline fits on train, transforms val without error,
    and scaler.mean_ matches only train data."""
    from src.models.logistic_baseline import build_pipeline, NUMERICAL_FEATURES

    n = len(small_dataset)
    train = small_dataset.iloc[:int(n * 0.7)]
    val = small_dataset.iloc[int(n * 0.7):]

    pipeline = build_pipeline()
    X_train = train.drop(columns=['rto_label'])
    y_train = train['rto_label']
    pipeline.fit(X_train, y_train)

    # Transform val without error
    X_val = val.drop(columns=['rto_label'])
    proba = pipeline.predict_proba(X_val)
    assert proba.shape[0] == len(val)
    assert proba.shape[1] == 2

    # Verify scaler.mean_ matches train stats (not val stats)
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
    train_means = X_train[NUMERICAL_FEATURES].mean().values
    np.testing.assert_allclose(scaler.mean_, train_means, rtol=1e-5,
                                err_msg="Scaler fitted on non-train data")


# ── Test: Evaluation Metrics ─────────────────────────────────────────

def test_precision_recall_matches_sklearn():
    """Our evaluate function matches sklearn on a small hand-computed example."""
    from src.eval.evaluate import evaluate_predictions

    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])

    metrics, cm = evaluate_predictions(y_true, y_pred)

    expected_prec = precision_score(y_true, y_pred)
    expected_rec = recall_score(y_true, y_pred)

    assert abs(metrics['precision'] - round(expected_prec, 4)) < 1e-4
    assert abs(metrics['recall'] - round(expected_rec, 4)) < 1e-4
    assert cm[1, 1] == 3  # TP
    assert cm[0, 1] == 1  # FP
