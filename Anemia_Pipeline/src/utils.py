"""Shared helpers used across all three stages."""

import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from . import config


def set_global_seed(seed=config.RANDOM_STATE):
    """
    Seed Python's random module, NumPy, and TensorFlow/Keras, and ask TF to
    use deterministic ops (including GPU/cuDNN kernels where available).

    Call this once, as the very first thing in main(), in every script that
    trains a model. The data-split seed alone (config.RANDOM_STATE passed to
    train_test_split) is NOT enough for run-to-run reproducibility: Keras
    layer weight initializers and cuDNN's GPU algorithm selection are their
    own separate sources of randomness. This is why two runs of the same
    ablation script with the same seeded split still reported different
    accuracy (96.83% vs 96.36%) before this fix.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"WARNING: could not enable full TF op determinism ({e}). "
              f"Results should still be close across runs but may not be bit-exact.")


def load_labeled_filepaths(class0_dir, class1_dir, cap_per_class=None, seed=config.RANDOM_STATE):
    """
    Build a (paths, labels) pair from two class directories.

    cap_per_class: if given, deterministically subsample each class to this
    many files (sorted first, so the choice is reproducible) rather than
    silently using "however many files happen to be in the folder" — this
    was a source of a paper/code mismatch in the original notebooks.
    """
    rng = np.random.RandomState(seed)

    def list_files(d):
        files = sorted(os.listdir(d))
        if cap_per_class is not None:
            if len(files) < cap_per_class:
                raise ValueError(
                    f"{d} has only {len(files)} files, cannot cap at {cap_per_class}"
                )
            idx = rng.choice(len(files), size=cap_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        return [os.path.join(d, f) for f in files]

    class0_files = list_files(class0_dir)
    class1_files = list_files(class1_dir)

    paths = class0_files + class1_files
    labels = [0] * len(class0_files) + [1] * len(class1_files)
    print(f"Loaded {len(class0_files)} class-0 and {len(class1_files)} class-1 samples "
          f"({len(paths)} total) from:\n  {class0_dir}\n  {class1_dir}")
    return paths, labels


def decode_and_preprocess(path, label, img_size=config.IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = preprocess_input(img)
    return img, label


def make_dataset(paths, labels, batch_size=config.BATCH_SIZE, shuffle=True, seed=config.RANDOM_STATE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: decode_and_preprocess(p, l), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 1), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def select_best_threshold(y_true, y_prob, low, high, steps):
    """
    Sweep a threshold grid and return the value that maximizes accuracy.

    CRITICAL: call this ONLY with validation-set (y_true, y_prob), never with
    test-set values. The test set must remain untouched until the threshold
    is already fixed, or the reported metrics are optimistically biased.
    """
    from sklearn.metrics import accuracy_score

    best_acc, best_thresh = 0.0, 0.5
    for t in np.linspace(low, high, steps):
        preds = (y_prob > t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc, best_thresh = acc, float(t)
    return best_thresh, best_acc


def save_threshold(path, threshold, source="validation_set"):
    with open(path, "w") as f:
        json.dump({"threshold": float(threshold), "selected_on": source}, f, indent=2)


def load_threshold(path):
    with open(path) as f:
        return json.load(f)["threshold"]


def evaluate_and_report(y_true, y_prob, threshold, label="Model"):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print(f"\n=== {label} — evaluated at threshold={threshold:.4f} (test set, untouched until now) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print(report)
    return {"accuracy": acc, "confusion_matrix": cm.tolist(), "report": report, "threshold": threshold}
