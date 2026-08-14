"""
Stage-3: Normal vs. abnormal RBC morphology classification.

Fixes applied relative to the original notebook (MP_finalCode_89_86.ipynb,
"Phase-2" cell):
  1. CRITICAL — the original code swept the decision threshold directly
     against the *test* set and reported accuracy at the best test-set
     threshold. That is data leakage: the threshold is fit to the test
     labels, so the reported accuracy is optimistically biased and not a
     valid estimate of generalization performance. This version selects
     the threshold on the validation set only, then applies it once to the
     untouched test set.
  2. `confusion_matrix` / `classification_report` were used without being
     imported — fixed.
  3. Dataset size is explicit via `--cap-per-class` instead of a hardcoded
     `[:3000]` slice, so it can be set to match whatever the paper
     ultimately reports.

IMPORTANT CAVEAT (please read):
  This script currently derives normal/abnormal morphology labels from the
  same Healthy_individuals / Anemic_individuals folder split used for Stage-2
  anemia detection, because that is what the original notebook did. If your
  actual ground truth for morphology (normal vs. abnormal cell shape) comes
  from a different, independently annotated source, point --healthy-dir /
  --anemic-dir (or replace utils.load_labeled_filepaths entirely) at that
  source instead. Using the anemic/healthy label as a stand-in for
  normal/abnormal morphology should be explicitly stated as a limitation in
  the paper if that is in fact what was done.

Usage:
    python -m src.stage3_morphology_classification \
        --normal-dir /path/to/normal_morphology_images \
        --abnormal-dir /path/to/abnormal_morphology_images \
        --cap-per-class 6000 \
        --output-dir ./artifacts
"""

import argparse
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: F401 (used via utils)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from . import config, utils


def build_model():
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    for layer in base_model.layers[: -config.MORPH_UNFREEZE_LAST_N]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base_model.input, outputs=out), base_model


def main(normal_dir, abnormal_dir, output_dir, cap_per_class):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    paths, labels = utils.load_labeled_filepaths(normal_dir, abnormal_dir, cap_per_class=cap_per_class)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(1 - config.MORPH_TRAIN_SPLIT),
        stratify=labels, random_state=config.RANDOM_STATE,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=config.RANDOM_STATE
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = utils.make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = utils.make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = utils.make_dataset(test_paths, test_labels, shuffle=False)

    model, base_model = build_model()
    model.compile(optimizer=Adam(config.MORPH_INITIAL_LR), loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=config.MORPH_HEAD_EPOCHS, callbacks=callbacks)

    # Fine-tuning
    for layer in base_model.layers[-config.MORPH_FINETUNE_UNFREEZE_LAST_N :]:
        layer.trainable = True
    model.compile(optimizer=Adam(config.MORPH_FINETUNE_LR), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=config.MORPH_FINETUNE_EPOCHS)

    # --- Threshold selection: VALIDATION SET ONLY (this is the leakage fix) ---
    val_probs, val_true = [], []
    for imgs, lbls in val_ds:
        val_probs.extend(model.predict(imgs, verbose=0).ravel())
        val_true.extend(lbls.numpy())
    best_thresh, val_acc = utils.select_best_threshold(
        val_true, val_probs,
        low=config.MORPH_THRESHOLD_GRID_LOW,
        high=config.MORPH_THRESHOLD_GRID_HIGH,
        steps=config.MORPH_THRESHOLD_GRID_STEPS,
    )
    print(f"Selected threshold={best_thresh:.4f} (validation accuracy={val_acc:.4f})")

    # --- Final evaluation: test set touched exactly once, with the fixed threshold ---
    test_probs, test_true = [], []
    for imgs, lbls in test_ds:
        test_probs.extend(model.predict(imgs, verbose=0).ravel())
        test_true.extend(lbls.numpy())
    utils.evaluate_and_report(test_true, test_probs, best_thresh, label="Stage-3 Morphology Classification")

    model.save(os.path.join(output_dir, "stage3_morphology_model.keras"))
    utils.save_threshold(
        os.path.join(output_dir, "stage3_threshold.json"), best_thresh, source="validation_set"
    )
    print(f"Saved Stage-3 artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--abnormal-dir", required=True)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--cap-per-class", type=int, default=6000)
    args = parser.parse_args()
    cap = args.cap_per_class if args.cap_per_class and args.cap_per_class > 0 else None
    main(args.normal_dir, args.abnormal_dir, args.output_dir, cap)
