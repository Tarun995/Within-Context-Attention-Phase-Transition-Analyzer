"""Reconcile the 0.832 (single split, unscaled probe) with the repeated-split estimate.
Run from the repo root after probe_position has written probe_pos_gpt2.npz."""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

path = sys.argv[1] if len(sys.argv) > 1 else "probe_pos_gpt2.npz"
layers = [int(x) for x in sys.argv[2:]] or [3, 6]
z = np.load(path, allow_pickle=True)
y = z["labels"]
for layer in layers:
    X = z[f"X_{layer}_end"]
    # your exact setup in linear_probe.py: unscaled LR, one split, random_state=42
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    one = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"\nlayer {layer}, end-of-answer state")
    print(f"  your exact setup (unscaled, seed 42, one split): AUC {one:.3f}")
    for name, make in [("unscaled", lambda: LogisticRegression(max_iter=2000)),
                       ("scaled", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))]:
        aucs = []
        for tr, te in StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=0).split(X, y):
            m = make().fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
        print(f"  {name:>8}, 20 random splits: AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
              f"(min {min(aucs):.3f}, max {max(aucs):.3f})")