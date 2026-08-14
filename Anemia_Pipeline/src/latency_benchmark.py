"""
Inference latency benchmark for the three-stage pipeline.

Addresses the "real-time / mobile deployment" claims in the paper's
Discussion section, which currently have no timing numbers behind them.
Reports per-image inference latency (mean, std, p95) for each stage, on
whatever hardware you run this on — report the GPU/CPU used alongside the
numbers, since latency is hardware-dependent.

Usage:
    python -m src.latency_benchmark --artifacts-dir ./artifacts \
        --sample-image path/to/any_test_image.png --n-runs 100
"""

import argparse
import time

import numpy as np
import joblib
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from . import config


def preprocess_for_keras(img_path, img_size=config.IMG_SIZE):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def benchmark(fn, n_runs, warmup=5):
    for _ in range(warmup):  # warm up the GPU/graph before timing
        fn()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)  # ms
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
    }


def main(artifacts_dir, sample_image, n_runs):
    import os

    img_arr = preprocess_for_keras(sample_image)
    results = {}

    # Stage 1: EfficientNetB0 feature extraction + XGBoost
    stage1_extractor = load_model(os.path.join(artifacts_dir, "stage1_feature_extractor.keras"))
    stage1_xgb = joblib.load(os.path.join(artifacts_dir, "stage1_xgb_classifier.pkl"))

    def stage1_fn():
        feats = stage1_extractor.predict(img_arr, verbose=0)
        stage1_xgb.predict_proba(feats)

    results["stage1_palm_screening"] = benchmark(stage1_fn, n_runs)
    print(f"Stage 1: {results['stage1_palm_screening']}")

    # Stage 2: RBC anemia detection
    stage2_path = os.path.join(artifacts_dir, "stage2_rbc_anemia_model.keras")
    if os.path.exists(stage2_path):
        stage2_model = load_model(stage2_path)

        def stage2_fn():
            stage2_model.predict(img_arr, verbose=0)

        results["stage2_rbc_anemia_detection"] = benchmark(stage2_fn, n_runs)
        print(f"Stage 2: {results['stage2_rbc_anemia_detection']}")

    # Stage 3: Morphology classification
    stage3_path = os.path.join(artifacts_dir, "stage3_morphology_model.keras")
    if os.path.exists(stage3_path):
        stage3_model = load_model(stage3_path)

        def stage3_fn():
            stage3_model.predict(img_arr, verbose=0)

        results["stage3_morphology_classification"] = benchmark(stage3_fn, n_runs)
        print(f"Stage 3: {results['stage3_morphology_classification']}")

    # Full pipeline (Stage 1 -> 2 -> 3 sequentially)
    def full_pipeline_fn():
        feats = stage1_extractor.predict(img_arr, verbose=0)
        stage1_xgb.predict_proba(feats)
        if "stage2_rbc_anemia_detection" in results:
            stage2_model.predict(img_arr, verbose=0)
        if "stage3_morphology_classification" in results:
            stage3_model.predict(img_arr, verbose=0)

    results["full_pipeline"] = benchmark(full_pipeline_fn, n_runs)
    print(f"Full pipeline: {results['full_pipeline']}")

    import json
    out_path = os.path.join(artifacts_dir, "latency_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved latency results to {out_path}")
    print("NOTE: report the GPU/CPU model used alongside these numbers in the paper — "
          "latency is hardware-dependent and not comparable across setups without it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--sample-image", required=True)
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()
    main(args.artifacts_dir, args.sample_image, args.n_runs)
