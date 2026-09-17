"""
src/models/tree_model.py
WHY: Trains the main LightGBM risk model inside a sklearn Pipeline (encoder + classifier),
     then applies isotonic calibration on val_cal to produce well-calibrated probabilities.

     Three artifacts are saved:
       - tree_model.pkl              : full pipeline (encoder + uncalibrated LGBM)
       - tree_model_calibrated.pkl   : CalibratedClassifierCV wrapping tree_model
       - tree_model_booster.pkl      : raw LightGBM booster for SHAP TreeExplainer

     Why isotonic over Platt scaling? Isotonic is non-parametric and works better
     when the raw scores are already monotone but non-linear (typical for LGBM).
     val_cal is used so the calibration set is disjoint from training AND from the
     evaluation window (val_rep / test).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

MODELS_DIR   = "models"
PROCESSED_DIR = "data/processed"
SEED = 42

# ── Feature definitions ───────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["category", "payment_method", "courier_id"]
# pincode_tier treated as numeric ordinal (already 1/2/3)
NUMERIC_FEATURES = [
    "order_value", "quantity", "discount_pct", "cod_charge",
    "account_age_days", "prior_orders", "prior_rto_count",
    "orders_last_24h", "device_cluster_size",
    "historical_pincode_rto_rate", "pincode_tier",
]
TARGET = "rto_label"
DROP_COLS = ["order_id", "order_date", "customer_id", "pincode", TARGET]


def _drop_extra(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")


class IsoCalibratedPipeline:
    """
    Module-level picklable wrapper: sklearn Pipeline -> IsotonicRegression calibration.
    Placed at module scope so joblib.dump/load works across processes.
    """
    def __init__(self, base_pipe: Pipeline, iso_reg: IsotonicRegression):
        self._pipe = base_pipe
        self._iso  = iso_reg
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        raw = self._pipe.predict_proba(X)[:, 1]
        cal = self._iso.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def named_steps(self):
        """Forward named_steps so explainability code can access the preprocessor."""
        return self._pipe.named_steps


def build_pipeline() -> Pipeline:
    """Build the sklearn pipeline with OneHot encoder + LGBMClassifier."""
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def train(
    train_path: str = None,
    val_cal_path: str = None,
    models_dir: str = None,
) -> dict:
    """
    Train, calibrate, and save model artifacts.

    Returns dict with paths to saved artifacts.
    """
    if train_path is None:
        train_path = f"{PROCESSED_DIR}/train.csv"
    if val_cal_path is None:
        val_cal_path = f"{PROCESSED_DIR}/val_cal.csv"
    if models_dir is None:
        models_dir = MODELS_DIR

    os.makedirs(models_dir, exist_ok=True)

    # Load data
    train = pd.read_csv(train_path, dtype={"pincode": str})
    val_cal = pd.read_csv(val_cal_path, dtype={"pincode": str})

    X_train = _drop_extra(train)
    y_train = train[TARGET].values
    X_val   = _drop_extra(val_cal)
    y_val   = val_cal[TARGET].values

    print(f"Training on {len(train):,} rows, calibrating on {len(val_cal):,} rows")

    # ── Train uncalibrated pipeline ────────────────────────────────────────────
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # ── Extract raw booster for SHAP ───────────────────────────────────────────
    booster = pipe.named_steps["classifier"].booster_

    # ── Isotonic calibration on val_cal ───────────────────────────────────────
    raw_proba_val = pipe.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_proba_val, y_val)

    # Wrap with module-level class (picklable across processes)
    cal = IsoCalibratedPipeline(pipe, iso)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    paths = {
        "uncalibrated": os.path.join(models_dir, "tree_model.pkl"),
        "calibrated":   os.path.join(models_dir, "tree_model_calibrated.pkl"),
        "booster":      os.path.join(models_dir, "tree_model_booster.pkl"),
    }
    joblib.dump(pipe,    paths["uncalibrated"])
    joblib.dump(cal,     paths["calibrated"])
    joblib.dump(booster, paths["booster"])

    # ── Quick eval ────────────────────────────────────────────────────────────
    from sklearn.metrics import average_precision_score, roc_auc_score
    proba_cal = cal.predict_proba(X_val)[:, 1]
    print(f"Val-cal  AUC-ROC : {roc_auc_score(y_val, proba_cal):.4f}")
    print(f"Val-cal  PR-AUC  : {average_precision_score(y_val, proba_cal):.4f}")
    print(f"Val-cal  mean p  : {proba_cal.mean():.4f}  (true rate: {y_val.mean():.4f})")
    print(f"Artifacts saved -> {models_dir}/")
    for k, v in paths.items():
        print(f"  {k:15s}: {v}")

    return paths


if __name__ == "__main__":
    train()
