"""
RTO Shield -- Stage 2: Logistic Regression Baseline

Preprocessing + LogisticRegression trained on temporal TRAIN split only.
- Numerical features: StandardScaler (fit on train only)
- Categorical features: One-hot encoding for low-cardinality features
  (category, payment_method, courier_id, pincode_tier).
  For pincode (~1000 values): DROPPED -- its signal is already captured by
  historical_pincode_rto_rate and pincode_tier. Including 1000 one-hot columns
  would add noise, increase dimensionality, and overfit on a synthetic dataset.
- Fitted pipeline saved to models/logistic_baseline.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Feature lists from configs/data_config.yaml (hardcoded to avoid yaml dep)
NUMERICAL_FEATURES = [
    'order_value', 'quantity', 'discount_pct', 'cod_charge',
    'account_age_days', 'prior_orders', 'prior_rto_count',
    'historical_pincode_rto_rate', 'orders_last_24h', 'device_cluster_size'
]

# Low-cardinality categoricals get one-hot encoding
ONEHOT_CATEGORICALS = ['category', 'payment_method', 'courier_id', 'pincode_tier']

# pincode is DROPPED (see docstring above)
TARGET = 'rto_label'


def build_pipeline():
    """Build the sklearn preprocessing + logistic regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             ONEHOT_CATEGORICALS),
        ],
        remainder='drop'  # drops order_id, customer_id, timestamp, pincode
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            C=1.0,  # default regularization; can tune on val later
            random_state=42,
        ))
    ])
    return pipeline


def train_model(train_path="data/processed/train.csv",
                model_path="models/logistic_baseline.pkl"):
    """Train LR on train split, save pipeline."""
    train_df = pd.read_csv(train_path, dtype={'pincode': str})
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Logistic regression pipeline trained and saved to: {model_path}")

    return pipeline


def predict(pipeline, df):
    """Return (binary predictions at 0.5 threshold, probabilities)."""
    X = df.drop(columns=[TARGET], errors='ignore')
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return preds, proba


def print_top_coefficients(pipeline, n=10):
    """Print top N coefficients by absolute magnitude."""
    clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()
    # Clean names
    feature_names = [n.replace('num__', '').replace('cat__', '')
                     for n in feature_names]

    coefs = clf.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    print(f"\nTop {n} Logistic Regression Coefficients (by |magnitude|):")
    print("-" * 55)
    for i in range(min(n, len(coefs))):
        idx = sorted_idx[i]
        name = feature_names[idx]
        coef = coefs[idx]
        sign = "+" if coef > 0 else "-"
        print(f"  {sign} {name:40s}  {coef:+.4f}")

    return feature_names, coefs


def main():
    pipeline = train_model()

    # Quick prediction summary
    for split_name in ['val', 'test']:
        df = pd.read_csv(f"data/processed/{split_name}.csv", dtype={'pincode': str})
        preds, proba = predict(pipeline, df)
        print(f"\n{split_name.upper()}: predicted {preds.sum()} RTO out of {len(preds)}")

    print_top_coefficients(pipeline)


if __name__ == "__main__":
    main()
