"""
K-fold cross-validation for Stage-2 (RBC anemia detection) and Stage-3
(morphology classification).

Retraining a full CNN k times is expensive, so this script exposes
--epochs-per-fold to let you trade off thoroughness vs. runtime (e.g., use
fewer epochs per fold than the full training run, since the goal here is an
honest variance estimate across folds, not a final deployable model — train
your final deployed model separately with the full epoch budget via
stage2_rbc_anemia_detection.py / stage3_morphology_classification.py).

Each fold still uses an internal validation split (carved out of that
fold's training data) to select the decision threshold, and only evaluates
on that fold's held-out test data — no leakage.

Usage:
    python -m src.cv_deep_stage --stage 2 \
        --class0-dir /path/to/Healthy_individuals/RGB_segmented \
        --class1-dir /path/to/Anemic_individuals/RGB_segmented \
        --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts

    python -m src.cv_deep_stage --stage 3 \
        --class0-dir /path/to/normal --class1-dir /path/to/abnormal \
        --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts
"""

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from . import config, utils


def get_model_builder(stage):
    if stage == 2:
        from .stage2_rbc_anemia_detection import build_model
    elif stage == 3:
        from .stage3_morphology_classification import build_model
    else:
        raise ValueError("--stage must be 2 or 3")
    return build_model


def run_cv(stage, class0_dir, class1_dir, output_dir, cap_per_class, k, epochs_per_fold):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)
    build_model = get_model_builder(stage)

    paths, labels = utils.load_labeled_filepaths(class0_dir, class1_dir, cap_per_class=cap_per_class)
    paths, labels = np.array(paths), np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels), start=1):
        train_paths, test_paths = paths[train_idx].tolist(), paths[test_idx].tolist()
        train_labels, test_labels = labels[train_idx].tolist(), labels[test_idx].tolist()

        # Carve a validation split out of this fold's training data for threshold selection
        tr_paths, val_paths, tr_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.15, stratify=train_labels,
            random_state=config.RANDOM_STATE,
        )

        train_ds = utils.make_dataset(tr_paths, tr_labels, shuffle=True)
        val_ds = utils.make_dataset(val_paths, val_labels, shuffle=False)
        test_ds = utils.make_dataset(test_paths, test_labels, shuffle=False)

        model, _ = build_model()
        model.compile(optimizer=Adam(config.RBC_INITIAL_LR), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            train_ds, validation_data=val_ds, epochs=epochs_per_fold,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            verbose=1,
        )

        val_probs, val_true = [], []
        for imgs, lbls in val_ds:
            val_probs.extend(model.predict(imgs, verbose=0).ravel())
            val_true.extend(lbls.numpy())
        best_thresh, _ = utils.select_best_threshold(val_true, val_probs, low=0.3, high=0.7, steps=100)

        test_probs, test_true = [], []
        for imgs, lbls in test_ds:
            test_probs.extend(model.predict(imgs, verbose=0).ravel())
            test_true.extend(lbls.numpy())

        result = utils.evaluate_and_report(test_true, test_probs, best_thresh, label=f"Fold {fold}")
        fold_results.append({"fold": fold, "threshold": best_thresh, "accuracy": result["accuracy"]})

    accs = [r["accuracy"] for r in fold_results]
    summary = {"mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs))}
    print(f"\n=== Stage-{stage} {k}-fold CV summary ===")
    print(f"Accuracy: {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")

    out_path = os.path.join(output_dir, f"stage{stage}_cv_results.json")
    with open(out_path, "w") as f:
        json.dump({"folds": fold_results, "summary": summary}, f, indent=2)
    print(f"Saved CV results to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True, choices=[2, 3])
    parser.add_argument("--class0-dir", required=True, help="Healthy dir (stage 2) or normal-morphology dir (stage 3)")
    parser.add_argument("--class1-dir", required=True, help="Anemic dir (stage 2) or abnormal-morphology dir (stage 3)")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--cap-per-class", type=int, default=6000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs-per-fold", type=int, default=8,
                         help="Lower than full training epochs to keep CV runtime reasonable.")
    args = parser.parse_args()
    cap = args.cap_per_class if args.cap_per_class and args.cap_per_class > 0 else None
    run_cv(args.stage, args.class0_dir, args.class1_dir, args.output_dir, cap, args.k, args.epochs_per_fold)
