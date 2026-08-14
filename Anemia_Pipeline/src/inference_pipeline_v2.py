"""
End-to-end inference v2: palm image -> (if anemic) RBC image -> REAL
multi-class morphology classification (Chula-RBC-12-trained) -> disease
mapping -> diagnostic report.

Difference from inference_pipeline.py: Stage-3 here uses the genuine
multi-class classifier (stage3_morphology_multiclass.py) trained on
independently annotated RBC morphology labels, instead of the binary
abnormal/normal proxy + score-threshold heuristic. The RBC input image is
expected to contain multiple cells (a smear image, like the Chula-RBC-12
inputs) — if you only have a single-cell crop, this still works but the
"summary" will just be that one cell's prediction.

Usage:
    python -m src.inference_pipeline_v2 \
        --artifacts-dir ./artifacts \
        --palm-image path/to/palm.jpg \
        --rbc-image path/to/rbc_smear.jpg \
        --crop-size 64
"""

import argparse
import json
import os

import joblib
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from . import disease_mapping_v2 as dm
from . import utils


def preprocess_for_keras(img_path, img_size=224):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def detect_and_crop_cells(rbc_image_path, crop_size=64):
    """
    Placeholder cell-detection step. The Chula-RBC-12 training data provides
    ground-truth cell centers, but for a NEW, unlabeled smear image you need
    a cell-detection step (e.g., watershed segmentation as used by the
    dataset's own authors, or a lightweight blob/contour detector) to find
    cell centers before cropping and classifying each one.

    This function is intentionally left as a stub — plug in your preferred
    RBC detector here. For now it returns a single crop centered on the
    image (useful only for single-cell input images).
    """
    img = image.load_img(rbc_image_path)
    w, h = img.size
    cx, cy = w // 2, h // 2
    crop = img.crop((cx - crop_size // 2, cy - crop_size // 2, cx + crop_size // 2, cy + crop_size // 2))
    return [crop]


def classify_cells(model, class_names, cell_crops, img_size=224):
    predictions = []
    for crop in cell_crops:
        arr = crop.resize((img_size, img_size))
        arr = image.img_to_array(arr)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        probs = model.predict(arr, verbose=0)[0]
        predictions.append(class_names[int(np.argmax(probs))])
    return predictions


def run(artifacts_dir, palm_image_path, rbc_image_path=None, crop_size=64):
    stage1_extractor = load_model(os.path.join(artifacts_dir, "stage1_feature_extractor.keras"))
    stage1_xgb = joblib.load(os.path.join(artifacts_dir, "stage1_xgb_classifier.pkl"))

    palm_arr = preprocess_for_keras(palm_image_path)
    palm_features = stage1_extractor.predict(palm_arr, verbose=0)
    palm_prob = float(stage1_xgb.predict_proba(palm_features)[0][1])
    palm_anemic = palm_prob > 0.5

    report = {
        "palm_anemia_probability": palm_prob,
        "palm_prediction": "Anemic" if palm_anemic else "Non-Anemic",
    }

    if not palm_anemic or rbc_image_path is None:
        report["summary"] = (
            "No anemia detected at palm-screening stage." if not palm_anemic
            else "Palm screening flagged possible anemia. Provide --rbc-image to continue."
        )
        print(json.dumps(report, indent=2))
        return report

    stage2_model = load_model(os.path.join(artifacts_dir, "stage2_rbc_anemia_model.keras"))
    stage2_threshold = utils.load_threshold(os.path.join(artifacts_dir, "stage2_threshold.json"))
    rbc_arr = preprocess_for_keras(rbc_image_path)
    rbc_prob = float(stage2_model.predict(rbc_arr, verbose=0)[0][0])
    rbc_anemic = rbc_prob > stage2_threshold

    # --- Real multi-class morphology classification ---
    multiclass_model = load_model(os.path.join(artifacts_dir, "stage3_morphology_multiclass_model.keras"))
    with open(os.path.join(artifacts_dir, "stage3_multiclass_classes.json")) as f:
        class_names = json.load(f)

    cell_crops = detect_and_crop_cells(rbc_image_path, crop_size=crop_size)
    cell_predictions = classify_cells(multiclass_model, class_names, cell_crops)
    morphology_summary = dm.summarize_cell_predictions(cell_predictions)
    diseases = dm.map_diseases(morphology_summary.keys())

    report.update({
        "rbc_anemia_probability": rbc_prob,
        "rbc_prediction": "Anemic" if rbc_anemic else "Healthy",
        "rbc_decision_threshold": stage2_threshold,
        "morphology_class_distribution": morphology_summary,
        "possible_disease_indications": diseases,
        "note": "Morphology from a genuine multi-class classifier (Chula-RBC-12-trained), "
                "not a healthy/anemic proxy. Cell detection is a placeholder — see "
                "detect_and_crop_cells() docstring.",
    })
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--palm-image", required=True)
    parser.add_argument("--rbc-image", default=None)
    parser.add_argument("--crop-size", type=int, default=64)
    args = parser.parse_args()
    run(args.artifacts_dir, args.palm_image, args.rbc_image, args.crop_size)
