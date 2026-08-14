"""
Ablation study for Stage-1: does the EfficientNetB0 + XGBoost hybrid
actually beat simpler alternatives on the same data/split?

Compares three configurations, all on the identical train/test split so the
comparison is fair:
  A) EfficientNetB0 (frozen) + XGBoost           <- what the paper reports
  B) EfficientNetB0 (frozen) + simple Dense head  <- isolates XGBoost's contribution
  C) End-to-end CNN trained from scratch (no transfer learning) <- isolates
     the contribution of ImageNet transfer learning itself

Usage:
    python -m src.ablation_stage1 --data-dir /path/to/Palm --output-dir ./artifacts
"""

import argparse
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from . import config, utils
from .stage1_palm_screening import build_feature_extractor, load_images_and_labels
from .statistical_tests import mcnemar_test


def eval_binary(y_true, y_pred, label):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        # Saved (not just printed) so statistical_tests.py's McNemar test can be
        # run between any two configs afterward on these exact, paired predictions.
        "y_true": np.asarray(y_true).astype(int).tolist(),
        "y_pred": np.asarray(y_pred).astype(int).tolist(),
    }
    print(f"\n=== {label} ===")
    for k, v in m.items():
        if k not in ("y_true", "y_pred"):
            print(f"{k}: {v:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return m


def config_a_efficientnet_xgboost(X_train_feat, y_train, X_test_feat, y_test):
    model = xgb.XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train_feat, y_train)
    y_pred = model.predict(X_test_feat)
    return eval_binary(y_test, y_pred, "Config A: EfficientNetB0 (frozen) + XGBoost")


def config_b_efficientnet_dense(X_train_feat, y_train, X_test_feat, y_test):
    inputs = layers.Input(shape=(X_train_feat.shape[1],))
    x = Dense(128, activation="relu")(inputs)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_feat, np.array(y_train), epochs=50, batch_size=32, verbose=0,
        validation_split=0.15,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    y_pred = (model.predict(X_test_feat, verbose=0).ravel() > 0.5).astype(int)
    return eval_binary(y_test, y_pred, "Config B: EfficientNetB0 (frozen) + simple Dense head")


def config_c_end_to_end_cnn(X_train_img, y_train, X_test_img, y_test):
    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_img, np.array(y_train), epochs=30, batch_size=32, verbose=0,
        validation_split=0.15,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    y_pred = (model.predict(X_test_img, verbose=0).ravel() > 0.5).astype(int)
    return eval_binary(y_test, y_pred, "Config C: End-to-end CNN trained from scratch (no transfer learning)")


def main(data_dir, output_dir):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    X, y = load_images_and_labels(data_dir)
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X, y, test_size=config.PALM_TEST_SPLIT, random_state=config.RANDOM_STATE, stratify=y
    )

    feature_extractor = build_feature_extractor()
    X_train_feat = feature_extractor.predict(X_train_img, batch_size=32, verbose=0)
    X_test_feat = feature_extractor.predict(X_test_img, batch_size=32, verbose=0)

    results = {
        "config_a_efficientnet_xgboost": config_a_efficientnet_xgboost(
            X_train_feat, y_train, X_test_feat, y_test
        ),
        "config_b_efficientnet_dense": config_b_efficientnet_dense(
            X_train_feat, y_train, X_test_feat, y_test
        ),
        "config_c_end_to_end_cnn": config_c_end_to_end_cnn(
            X_train_img, y_train, X_test_img, y_test
        ),
    }

    print("\n=== Ablation summary (same train/test split for all configs) ===")
    for name, m in results.items():
        print(f"{name}: accuracy={m['accuracy']:.4f} f1={m['f1']:.4f}")

    # Significance testing: McNemar's test between the paper's reported config (A)
    # and each alternative, using the exact paired predictions saved above. This is
    # what a Q1 reviewer will expect before "97.7% > 96.8%" is treated as a real
    # difference rather than noise on this test set.
    print("\n=== Significance testing (McNemar's test vs. Config A) ===")
    significance = {}
    y_true_a = results["config_a_efficientnet_xgboost"]["y_true"]
    for other in ("config_b_efficientnet_dense", "config_c_end_to_end_cnn"):
        print(f"\n-- Config A vs {other} --")
        significance[f"config_a_vs_{other}"] = mcnemar_test(
            y_true_a,
            results["config_a_efficientnet_xgboost"]["y_pred"],
            results[other]["y_pred"],
        )

    with open(os.path.join(output_dir, "stage1_ablation_results.json"), "w") as f:
        json.dump({"results": results, "significance": significance}, f, indent=2)
    print(f"Saved ablation results to {os.path.join(output_dir, 'stage1_ablation_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
