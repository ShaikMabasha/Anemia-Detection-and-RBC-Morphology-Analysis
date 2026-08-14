"""
Stage-3 v2: genuine multi-class RBC morphology classification, trained on
real, independently annotated labels (Chula-RBC-12 dataset) instead of the
binary healthy/anemic proxy used in stage3_morphology_classification.py.

This directly delivers what the paper's Future Work section described as
future work: "identification of specific RBC abnormalities such as
microcytes, macrocytes, target cells, elliptocytes and sickle cells."

Class imbalance handling: the dataset ranges from 6,330 (Normal) down to
307 (Teardrop) samples per class. This script uses class weights (inverse
frequency) rather than naive training, and reports per-class precision/
recall/F1 (macro-averaged) so minority classes aren't hidden by overall
accuracy.

Usage:
    python -m src.stage3_morphology_multiclass \
        --data-dir ./chula_cropped --output-dir ./artifacts
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from . import config, utils


def load_multiclass_dataset(data_dir):
    """data_dir should contain one subfolder per class (as produced by crop_chula_dataset.py).

    Folders starting with "_" (e.g. "_preview", written by
    crop_chula_dataset.py --preview-only) are not real classes and are
    excluded, even if they happen to sit inside the same --output-dir as
    the real class folders.
    """
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_")
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for name in class_names:
        class_dir = os.path.join(data_dir, name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for f in files:
            paths.append(os.path.join(class_dir, f))
            labels.append(class_to_idx[name])
        print(f"{name}: {len(files)} samples")

    return paths, labels, class_names


def build_multiclass_model(num_classes, unfreeze_last_n=60):
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=outputs), base_model


def make_multiclass_dataset(paths, labels, batch_size=config.BATCH_SIZE, shuffle=True):
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 1), seed=config.RANDOM_STATE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(data_dir, output_dir, epochs, finetune_epochs):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    excluded = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("_")
    )
    if excluded:
        print(f"Excluding non-class folder(s) from training: {excluded}")

    paths, labels, class_names = load_multiclass_dataset(data_dir)
    num_classes = len(class_names)
    print(f"\n{num_classes} classes: {class_names}")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=config.RANDOM_STATE
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=config.RANDOM_STATE
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Class weights to counter the severe imbalance (6,330 Normal vs. 307 Teardrop etc.)
    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weight_values)}
    print(f"Class weights: {class_weights}")

    train_ds = make_multiclass_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_multiclass_dataset(val_paths, val_labels, shuffle=False)
    test_ds = make_multiclass_dataset(test_paths, test_labels, shuffle=False)

    model, base_model = build_multiclass_model(num_classes)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4),
    ]
    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weights, callbacks=callbacks,
    )

    # Fine-tuning
    for layer in base_model.layers[-40:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_ds, validation_data=val_ds, epochs=finetune_epochs,
        class_weight=class_weights, callbacks=callbacks,
    )

    # Final evaluation on the untouched test set
    y_true, y_pred = [], []
    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(lbls.numpy())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Stage-3 v2 Multi-class Morphology Classification — Test Set ===")
    print(report)
    print("Confusion matrix:\n", cm)

    model.save(os.path.join(output_dir, "stage3_morphology_multiclass_model.keras"))
    with open(os.path.join(output_dir, "stage3_multiclass_classes.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    with open(os.path.join(output_dir, "stage3_multiclass_report.txt"), "w") as f:
        f.write(report)
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(cm))
    print(f"\nSaved Stage-3 v2 artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Output of crop_chula_dataset.py")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.epochs, args.finetune_epochs)
