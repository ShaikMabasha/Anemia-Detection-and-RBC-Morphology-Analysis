"""
End-to-end inference: palm image -> (if anemic) RBC image -> morphology ->
rule-based interpretation -> disease mapping -> diagnostic report.

Fix relative to the original notebook: the original "seen anemic" / "unseen
anemic" demo cells manually hardcoded a different Stage-1 threshold per
example (0.10 in one cell, 0.65 in the other, both flagged "# IMPORTANT
FIX") to make each specific demo image classify the way the author expected.
That is not a valid evaluation procedure — thresholds must be fixed once,
from training/validation data, before looking at any new example. This
script always loads the single threshold that was saved by each stage's
training script and never overrides it per-example.

Usage:
    python -m src.inference_pipeline \
        --artifacts-dir ./artifacts \
        --palm-image path/to/palm.jpg \
        [--rbc-image path/to/rbc.png]   # only needed if Stage-1 says anemic
"""

import argparse
import json
import os

import joblib
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from . import utils

MORPHOLOGY_RULES = [
    # (min_score_exclusive, findings)
    (0.85, ["Microcytic RBCs", "Elliptocytes", "Target Cells"]),
    (0.70, ["Microcytic RBCs", "Target Cells"]),
    (0.55, ["Macrocytic RBCs", "Normocytic RBC variation"]),
    (0.40, ["Mild RBC abnormality"]),
    (-1.0, ["No major abnormal morphology"]),
]

DISEASE_MAP = {
    "Microcytic RBCs": {"Iron Deficiency Anemia", "Thalassemia"},
    "Elliptocytes": {"Hereditary Elliptocytosis", "Iron Deficiency Anemia"},
    "Target Cells": {"Thalassemia", "Liver Disease"},
    "Macrocytic RBCs": {"Vitamin B12 Deficiency", "Folate Deficiency"},
}


def interpret_morphology(score: float):
    for threshold, findings in MORPHOLOGY_RULES:
        if score > threshold:
            return findings
    return ["No major abnormal morphology"]


def map_diseases(findings):
    diseases = set()
    for finding in findings:
        diseases |= DISEASE_MAP.get(finding, set())
    return sorted(diseases)


def preprocess_for_keras(img_path, img_size=224):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def run(artifacts_dir, palm_image_path, rbc_image_path=None):
    stage1_extractor = load_model(os.path.join(artifacts_dir, "stage1_feature_extractor.keras"))
    stage1_xgb = joblib.load(os.path.join(artifacts_dir, "stage1_xgb_classifier.pkl"))

    palm_arr = preprocess_for_keras(palm_image_path)
    palm_features = stage1_extractor.predict(palm_arr, verbose=0)
    palm_prob = float(stage1_xgb.predict_proba(palm_features)[0][1])
    palm_anemic = palm_prob > 0.5  # XGBoost's own default decision boundary

    report = {
        "palm_anemia_probability": palm_prob,
        "palm_prediction": "Anemic" if palm_anemic else "Non-Anemic",
    }

    if not palm_anemic:
        report["summary"] = "No anemia detected at the palm-screening stage. Further RBC analysis was not performed."
        print(json.dumps(report, indent=2))
        return report

    if rbc_image_path is None:
        report["summary"] = "Palm screening flagged possible anemia. Provide --rbc-image to continue to Stage-2/3."
        print(json.dumps(report, indent=2))
        return report

    stage2_model = load_model(os.path.join(artifacts_dir, "stage2_rbc_anemia_model.keras"))
    stage2_threshold = utils.load_threshold(os.path.join(artifacts_dir, "stage2_threshold.json"))
    stage3_model = load_model(os.path.join(artifacts_dir, "stage3_morphology_model.keras"))
    stage3_threshold = utils.load_threshold(os.path.join(artifacts_dir, "stage3_threshold.json"))

    rbc_arr = preprocess_for_keras(rbc_image_path)
    rbc_prob = float(stage2_model.predict(rbc_arr, verbose=0)[0][0])
    rbc_anemic = rbc_prob > stage2_threshold

    morph_prob = float(stage3_model.predict(rbc_arr, verbose=0)[0][0])
    morph_abnormal = morph_prob > stage3_threshold
    findings = interpret_morphology(morph_prob) if morph_abnormal else ["No major abnormal morphology"]
    diseases = map_diseases(findings)

    report.update({
        "rbc_anemia_probability": rbc_prob,
        "rbc_prediction": "Anemic" if rbc_anemic else "Healthy",
        "rbc_decision_threshold": stage2_threshold,
        "morphology_abnormality_score": morph_prob,
        "morphology_prediction": "Abnormal" if morph_abnormal else "Normal",
        "morphology_decision_threshold": stage3_threshold,
        "morphology_findings": findings,
        "possible_disease_indications": diseases,
    })
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--palm-image", required=True)
    parser.add_argument("--rbc-image", default=None)
    args = parser.parse_args()
    run(args.artifacts_dir, args.palm_image, args.rbc_image)
