"""
Stage-2: RBC-based (microscopic) anemia detection via transfer learning.

Fixes applied relative to the original notebook (MP_finalCode_89_86.ipynb,
"Phase-1" cell):
  1. `EarlyStopping`/`ReduceLROnPlateau` monitored `val_loss`, but the
     original `model.fit()` call passed no validation data, so both
     callbacks were silently inert. This version creates an explicit
     validation split and passes `validation_data=val_ds`.
  2. Dataset size is now made explicit and reproducible via
     `--cap-per-class` (matching the paper's stated "6,000 healthy + 6,000
     anemic" balanced subset) instead of silently using every file found
     on disk.
  3. The final decision threshold is selected on the *validation* set only
     (see utils.select_best_threshold) and applied once to the untouched
     test set for reporting — the test set is never used to pick a
     threshold.

Usage:
    python -m src.stage2_rbc_anemia_detection \
        --healthy-dir /path/to/AneRBC-II/Healthy_individuals/RGB_segmented \
        --anemic-dir  /path/to/AneRBC-II/Anemic_individuals/RGB_segmented \
        --cap-per-class 6000 \
        --output-dir ./artifacts
"""

import argparse
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from . import config, utils


def build_model():
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    for layer in base_model.layers[: -config.RBC_UNFREEZE_LAST_N]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base_model.input, outputs=out), base_model


def main(healthy_dir, anemic_dir, output_dir, cap_per_class):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    paths, labels = utils.load_labeled_filepaths(healthy_dir, anemic_dir, cap_per_class=cap_per_class)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=config.RBC_TEST_SPLIT, stratify=labels, random_state=config.RANDOM_STATE
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=config.RBC_VAL_SPLIT,
        stratify=train_labels, random_state=config.RANDOM_STATE,
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = utils.make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = utils.make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = utils.make_dataset(test_paths, test_labels, shuffle=False)

    model, base_model = build_model()
    model.compile(optimizer=Adam(config.RBC_INITIAL_LR), loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=config.RBC_INITIAL_EPOCHS, callbacks=callbacks)

    # Fine-tuning
    for layer in base_model.layers[-config.RBC_FINETUNE_UNFREEZE_LAST_N :]:
        layer.trainable = True
    model.compile(optimizer=Adam(config.RBC_FINETUNE_LR), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=config.RBC_FINETUNE_EPOCHS, callbacks=callbacks)

    # Threshold selection on validation set ONLY
    val_probs, val_true = [], []
    for imgs, lbls in val_ds:
        val_probs.extend(model.predict(imgs, verbose=0).ravel())
        val_true.extend(lbls.numpy())
    best_thresh, val_acc = utils.select_best_threshold(
        val_true, val_probs, low=0.3, high=0.7, steps=100
    )
    print(f"Selected threshold={best_thresh:.4f} (validation accuracy={val_acc:.4f})")

    # Final, untouched-until-now evaluation on the test set
    test_probs, test_true = [], []
    for imgs, lbls in test_ds:
        test_probs.extend(model.predict(imgs, verbose=0).ravel())
        test_true.extend(lbls.numpy())
    utils.evaluate_and_report(test_true, test_probs, best_thresh, label="Stage-2 RBC Anemia Detection")

    model.save(os.path.join(output_dir, "stage2_rbc_anemia_model.keras"))
    utils.save_threshold(
        os.path.join(output_dir, "stage2_threshold.json"), best_thresh, source="validation_set"
    )
    print(f"Saved Stage-2 artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy-dir", required=True)
    parser.add_argument("--anemic-dir", required=True)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--cap-per-class", type=int, default=6000,
                         help="Set to None-equivalent (0) to use every available file instead.")
    args = parser.parse_args()
    cap = args.cap_per_class if args.cap_per_class and args.cap_per_class > 0 else None
    main(args.healthy_dir, args.anemic_dir, args.output_dir, cap)
