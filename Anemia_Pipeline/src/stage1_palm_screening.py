"""
Stage-1: Non-invasive palm-image anemia screening.

EfficientNetB0 (frozen, ImageNet weights) as a feature extractor +
XGBoost as the final classifier, as described in the paper's Stage-1 section.

Usage:
    python -m src.stage1_palm_screening --data-dir /path/to/Palm --output-dir ./artifacts

The `--data-dir` should contain images named so that filenames starting with
"non" (case-insensitive) are non-anemic and everything else is anemic —
matching the Palm-Anemia dataset's own filename convention. If your copy of
the dataset uses a different convention, adapt `label_from_filename`.
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import xgboost as xgb

from . import config, utils


def label_from_filename(filename: str) -> int:
    return 0 if filename.lower().startswith("non") else 1


def load_images_and_labels(dataset_path, image_size=(config.IMG_SIZE, config.IMG_SIZE)):
    images, labels = [], []
    n_anemic = n_non = n_skipped = 0

    for fname in sorted(os.listdir(dataset_path)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = os.path.join(dataset_path, fname)
        try:
            label = label_from_filename(fname)
            img = image.load_img(fpath, target_size=image_size)
            arr = preprocess_input(image.img_to_array(img))
            images.append(arr)
            labels.append(label)
            n_non += label == 0
            n_anemic += label == 1
        except Exception as e:
            print(f"Skipped {fname}: {e}")
            n_skipped += 1

    print(f"Anemic: {n_anemic} | Non-anemic: {n_non} | Skipped: {n_skipped}")
    return np.array(images, dtype="float32"), np.array(labels)


def build_feature_extractor():
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)


def main(data_dir, output_dir):
    utils.set_global_seed()
    os.makedirs(output_dir, exist_ok=True)

    X, y = load_images_and_labels(data_dir)
    print("Data shape:", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.PALM_TEST_SPLIT, random_state=config.RANDOM_STATE, stratify=y
    )

    feature_extractor = build_feature_extractor()
    print("Extracting features...")
    X_train_feat = feature_extractor.predict(X_train, batch_size=32, verbose=1)
    X_test_feat = feature_extractor.predict(X_test, batch_size=32, verbose=1)

    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS)
    xgb_model.fit(X_train_feat, y_train)

    y_pred = xgb_model.predict(X_test_feat)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    feature_extractor.save(os.path.join(output_dir, "stage1_feature_extractor.keras"))
    joblib.dump(xgb_model, os.path.join(output_dir, "stage1_xgb_classifier.pkl"))
    print(f"Saved Stage-1 artifacts to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory of palm images")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
