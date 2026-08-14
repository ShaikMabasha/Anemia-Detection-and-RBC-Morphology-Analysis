"""
K-fold cross-validation for the multi-class morphology classifier
(stage3_morphology_multiclass.py), trained on real Chula-RBC-12 labels.

This matters more here than for the binary stages: with 13 imbalanced
classes (6,330 Normal down to 307 Teardrop), a single train/test split can
land luckily or unluckily on the small classes just by chance. Reviewers at
a Q1/Q2 journal will expect variance estimates, not a single point value,
especially for imbalanced multi-class results.

Reports macro-averaged precision/recall/F1 per fold (macro-average matters
here because it weighs every class equally, so it won't hide poor
performance on rare classes the way accuracy or a weighted average would),
plus per-class F1 averaged across folds so you can see exactly which
morphology classes are hardest to classify.

Usage:
    python -m src.cv_stage3_multiclass --data-dir ./chula_cropped \
        --k 5 --epochs-per-fold 15 --output-dir ./artifacts
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from . import config, utils
from .stage3_morphology_multiclass import build_multiclass_model, load_multiclass_dataset, make_multiclass_dataset


def run_cv(data_dir, output_dir, k, epochs_per_fold):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    paths, labels, class_names = load_multiclass_dataset(data_dir)
    paths, labels = np.array(paths), np.array(labels)
    num_classes = len(class_names)
    print(f"{num_classes} classes: {class_names}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    fold_results = []
    per_class_f1_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels), start=1):
        train_paths_full = paths[train_idx].tolist()
        train_labels_full = labels[train_idx].tolist()
        test_paths = paths[test_idx].tolist()
        test_labels = labels[test_idx].tolist()

        # Carve a validation split out of this fold's training data
        tr_paths, val_paths, tr_labels, val_labels = train_test_split(
            train_paths_full, train_labels_full, test_size=0.15,
            stratify=train_labels_full, random_state=config.RANDOM_STATE,
        )

        class_weight_values = compute_class_weight(
            class_weight="balanced", classes=np.arange(num_classes), y=tr_labels
        )
        class_weights = {i: w for i, w in enumerate(class_weight_values)}

        train_ds = make_multiclass_dataset(tr_paths, tr_labels, shuffle=True)
        val_ds = make_multiclass_dataset(val_paths, val_labels, shuffle=False)
        test_ds = make_multiclass_dataset(test_paths, test_labels, shuffle=False)

        model, base_model = build_multiclass_model(num_classes)
        model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3),
        ]
        model.fit(
            train_ds, validation_data=val_ds, epochs=epochs_per_fold,
            class_weight=class_weights, callbacks=callbacks, verbose=1,
        )

        y_true, y_pred = [], []
        for imgs, lbls in test_ds:
            preds = model.predict(imgs, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(lbls.numpy())

        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=np.arange(num_classes))

        print(f"\n=== Fold {fold} ===")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

        fold_results.append({
            "fold": fold,
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
        })
        per_class_f1_all_folds.append(per_class_f1)

    macro_f1_values = [r["macro_f1"] for r in fold_results]
    macro_precision_values = [r["macro_precision"] for r in fold_results]
    macro_recall_values = [r["macro_recall"] for r in fold_results]

    per_class_f1_array = np.array(per_class_f1_all_folds)  # shape: (k, num_classes)
    per_class_f1_mean = per_class_f1_array.mean(axis=0)
    per_class_f1_std = per_class_f1_array.std(axis=0)

    summary = {
        "macro_precision": {"mean": float(np.mean(macro_precision_values)), "std": float(np.std(macro_precision_values))},
        "macro_recall": {"mean": float(np.mean(macro_recall_values)), "std": float(np.std(macro_recall_values))},
        "macro_f1": {"mean": float(np.mean(macro_f1_values)), "std": float(np.std(macro_f1_values))},
        "per_class_f1": {
            class_names[i]: {"mean": float(per_class_f1_mean[i]), "std": float(per_class_f1_std[i])}
            for i in range(num_classes)
        },
    }

    print(f"\n=== Stage-3 Multi-class {k}-fold CV summary ===")
    print(f"Macro F1: {summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['std']:.4f}")
    print(f"Macro Precision: {summary['macro_precision']['mean']:.4f} +/- {summary['macro_precision']['std']:.4f}")
    print(f"Macro Recall: {summary['macro_recall']['mean']:.4f} +/- {summary['macro_recall']['std']:.4f}")
    print("\nPer-class F1 (mean +/- std across folds):")
    for cls, stats in summary["per_class_f1"].items():
        print(f"  {cls}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    out_path = os.path.join(output_dir, "stage3_multiclass_cv_results.json")
    with open(out_path, "w") as f:
        json.dump({"folds": fold_results, "summary": summary}, f, indent=2)
    print(f"\nSaved CV results to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Output of crop_chula_dataset.py")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs-per-fold", type=int, default=15)
    args = parser.parse_args()
    run_cv(args.data_dir, args.output_dir, args.k, args.epochs_per_fold)
