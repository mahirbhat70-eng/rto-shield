"""Probe: does a depth-1 LightGBM actually fail the relative PR-AUC floor? (Task 4c evidence)"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score
from src.models.tree_model import get_preprocessor, TARGET
from src.eval.bayes_ceiling import get_true_p

tr = pd.read_csv("data/processed/train.csv", dtype={"pincode": str})
vc = pd.read_csv("data/processed/val_cal.csv", dtype={"pincode": str})
vr = pd.read_csv("data/processed/val_rep.csv", dtype={"pincode": str})
pre = get_preprocessor(); Xt = pre.fit_transform(tr.drop(columns=[TARGET]))
Xc, Xr = pre.transform(vc.drop(columns=[TARGET])), pre.transform(vr.drop(columns=[TARGET]))
ceil = average_precision_score(vr[TARGET], get_true_p(vr))
for d in [1, 2, 4, 6]:
    clf = LGBMClassifier(n_estimators=800, learning_rate=0.05, max_depth=d, num_leaves=max(2, 2**d - 1),
                         random_state=42, verbose=-1)
    clf.fit(Xt, tr[TARGET], eval_set=[(Xc, vc[TARGET])], callbacks=[early_stopping(50, verbose=False)])
    pr = average_precision_score(vr[TARGET], clf.predict_proba(Xr)[:, 1])
    print(f"max_depth={d}: val_rep PR-AUC={pr:.4f} ceiling={ceil:.4f} ratio={pr/ceil:.3f} iters={clf.best_iteration_}")
