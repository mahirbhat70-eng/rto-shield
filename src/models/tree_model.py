import os
import joblib
import itertools
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

NUMERICAL_FEATURES = [
    'order_value', 'quantity', 'discount_pct', 'cod_charge',
    'account_age_days', 'prior_orders', 'prior_rto_count',
    'historical_pincode_rto_rate', 'orders_last_24h', 'device_cluster_size'
]
ONEHOT_CATEGORICALS = ['category', 'payment_method', 'courier_id', 'pincode_tier']
TARGET = 'rto_label'

def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ONEHOT_CATEGORICALS),
        ],
        remainder='drop'
    )

def main():
    print("=" * 60)
    print("Stage 3: Gradient Boosting Model (LightGBM)")
    print("=" * 60)

    # Load splits
    train_df = pd.read_csv("data/processed/train.csv", dtype={'pincode': str})
    val_cal_df = pd.read_csv("data/processed/val_cal.csv", dtype={'pincode': str})
    val_rep_df = pd.read_csv("data/processed/val_rep.csv", dtype={'pincode': str})

    preprocessor = get_preprocessor()

    # Fit preprocessor on train only
    X_train_raw = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_train = preprocessor.fit_transform(X_train_raw)

    # Transform val_cal for tuning and early stopping
    X_val_cal_raw = val_cal_df.drop(columns=[TARGET])
    y_val_cal = val_cal_df[TARGET]
    X_val_cal = preprocessor.transform(X_val_cal_raw)

    # Grid search on val_cal ONLY
    grid = {
        'n_estimators': [400, 800],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6]
    }
    keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    best_pr_auc = -1
    best_params = None
    best_model = None

    print(f"Starting grid search on {len(combinations)} configurations using val_cal...")
    for combo in combinations:
        params = dict(zip(keys, combo))
        params['random_state'] = 42
        params['n_jobs'] = -1
        # Set max_depth but also keep num_leaves consistent if needed, 
        # LGBM default num_leaves is 31. For max_depth 4, max leaves is 16.
        params['num_leaves'] = 2 ** params['max_depth'] - 1

        clf = LGBMClassifier(**params)
        
        # We'll use early stopping on val_cal
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val_cal, y_val_cal)],
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
        )

        proba = clf.predict_proba(X_val_cal)[:, 1]
        score = average_precision_score(y_val_cal, proba)

        print(f"  {params} -> PR-AUC: {score:.4f} (best_iter: {clf.best_iteration_})")

        if score > best_pr_auc:
            best_pr_auc = score
            best_params = params
            best_model = clf

    print(f"\nBest Config: {best_params}")
    print(f"Best val_cal PR-AUC: {best_pr_auc:.4f}")

    # Retrain best model on train, evaluating on val_cal for early stopping
    print("\nRefitting best model...")
    best_clf = LGBMClassifier(**best_params)
    best_clf.fit(
        X_train, y_train,
        eval_set=[(X_val_cal, y_val_cal)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_clf)
    ])

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/tree_model.pkl')
    joblib.dump(best_clf, 'models/tree_model_booster.pkl')
    print("Saved models/tree_model.pkl and models/tree_model_booster.pkl")

    # Diagnostic check on val_rep
    X_val_rep_raw = val_rep_df.drop(columns=[TARGET])
    y_val_rep = val_rep_df[TARGET]
    proba_rep = pipeline.predict_proba(X_val_rep_raw)[:, 1]
    rep_pr_auc = average_precision_score(y_val_rep, proba_rep)

    print(f"\nval_rep PR-AUC (Diagnostic): {rep_pr_auc:.4f}")

if __name__ == "__main__":
    main()
