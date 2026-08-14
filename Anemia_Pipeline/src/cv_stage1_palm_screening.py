"""
K-fold cross-validation for Stage-1 (Palm screening).

Addresses the "no cross-validation" gap: instead of a single 80:20 split,
this runs stratified k-fold CV and reports mean +/- std for accuracy,
precision, recall, and F1, which is what reviewers expect instead of a
single point estimate.

Usage:
    python -m src.cv_stage1_palm_screening --data-dir /path/to/Palm --k 5 --output-dir ./artifacts
"""

import argparse
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from . import config, utils
from .stage1_palm_screening import build_feature_extractor, load_images_and_labels


def run_cv(data_dir, output_dir, k=5):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    X, y = load_images_and_labels(data_dir)
    print(f"Loaded {len(X)} images total.")

    # Extract features once (frozen EfficientNetB0 is not being trained, so
    # this is safe/valid to do outside the fold loop and saves a lot of time).
    feature_extractor = build_feature_extractor()
    print("Extracting features for all images...")
    X_feat = feature_extractor.predict(X, batch_size=32, verbose=1)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y), start=1):
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBClassifier(**config.XGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = {
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        fold_metrics.append(m)
        print(f"Fold {fold}: accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} f1={m['f1']:.4f}")

    summary = {}
    for key in ("accuracy", "precision", "recall", "f1"):
        values = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    print(f"\n=== Stage-1 {k}-fold CV summary ===")
    for key, stats in summary.items():
        print(f"{key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    with open(os.path.join(output_dir, "stage1_cv_results.json"), "w") as f:
        json.dump({"folds": fold_metrics, "summary": summary}, f, indent=2)
    print(f"Saved CV results to {os.path.join(output_dir, 'stage1_cv_results.json')}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    run_cv(args.data_dir, args.output_dir, args.k)
