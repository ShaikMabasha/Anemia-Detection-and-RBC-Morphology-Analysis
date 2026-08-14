# Anemia Detection Pipeline — Refactored

This is a cleaned-up, reproducible rewrite of the two original Colab
notebooks (`palm_dataset_efficientnetb0_saved.ipynb` and
`MP_finalCode_89_86.ipynb`), organized to mirror the paper's three stages.
It fixes three concrete bugs found in the original code, and removes the
Colab/Drive-specific plumbing so it can run anywhere.

## What changed and why

| Issue | Where | Fix |
|---|---|---|
| Stage-3 decision threshold was swept against the **test set** itself, then used to report test accuracy — data leakage | `MP_finalCode_89_86.ipynb`, Phase-2 cell | `stage3_morphology_classification.py` selects the threshold on a held-out **validation** split; the test set is touched exactly once, after the threshold is already fixed |
| `EarlyStopping`/`ReduceLROnPlateau` monitored `val_loss` but no validation data was passed to `model.fit()`, so both callbacks were silently inert | `MP_finalCode_89_86.ipynb`, Phase-1 cell | `stage2_rbc_anemia_detection.py` creates an explicit train/val/test split and passes `validation_data=val_ds` |
| `confusion_matrix`/`classification_report` used without being imported (would crash) | `MP_finalCode_89_86.ipynb`, Phase-2 cell | Imports fixed in `utils.py` |
| Demo cells manually hardcoded a different Stage-1 threshold per example (`0.10` vs `0.65`, tagged `# IMPORTANT FIX`) | `MP_finalCode_89_86.ipynb`, "seen/unseen anemic" cells | `inference_pipeline.py` always loads the single threshold saved during training — no per-example overrides |
| Dataset sizes were implicit (`os.listdir(...)` with no cap, or a hardcoded `[:3000]` slice) | both notebooks | `--cap-per-class` argument everywhere, so the exact sample count used is explicit and logged |

## ⚠️ Still needs your confirmation

`stage3_morphology_classification.py` currently builds its normal/abnormal
labels from the *same* Healthy/Anemic folder split as Stage-2, because
that's what the original notebook did. If that's genuinely how the
morphology ground truth was obtained, the paper should say so explicitly
(and note that "morphology" is being approximated by anemic/healthy status).
If there's a separate, independently annotated morphology dataset, point
`--normal-dir`/`--abnormal-dir` at that instead and disregard the note in
`stage3_morphology_classification.py`'s docstring.

## Filling the peer-review gaps

Beyond the three training scripts and inference pipeline, this now includes:

| Gap | Script | What it does |
|---|---|---|
| No cross-validation | `cv_stage1_palm_screening.py` | Stratified k-fold CV for Stage-1, reports mean ± std accuracy/precision/recall/F1 instead of a single split |
| No cross-validation | `cv_deep_stage.py --stage 2` or `--stage 3` | Same idea for the deep-learning stages; `--epochs-per-fold` lets you trade thoroughness for runtime |
| No ablation study | `ablation_stage1.py` | Compares (A) EfficientNetB0+XGBoost [the paper's approach], (B) EfficientNetB0 + simple Dense head, (C) end-to-end CNN with no transfer learning — all on the identical split |
| No significance testing | `statistical_tests.py` | McNemar's test between any two models' predictions on the same test set |
| No latency numbers | `latency_benchmark.py` | Per-image inference timing (mean/std/p95) for each stage and the full pipeline, to support or soften the paper's "real-time"/"mobile deployment" claims |

### Example commands

```bash
# 5-fold CV, Stage 1
python -m src.cv_stage1_palm_screening --data-dir /path/to/Palm --k 5 --output-dir ./artifacts

# 5-fold CV, Stage 2 (reduced epochs per fold to keep runtime sane)
python -m src.cv_deep_stage --stage 2 \
    --class0-dir /path/to/Healthy_individuals/RGB_segmented \
    --class1-dir /path/to/Anemic_individuals/RGB_segmented \
    --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts

# Ablation study, Stage 1
python -m src.ablation_stage1 --data-dir /path/to/Palm --output-dir ./artifacts

# Latency benchmark (train Stage 1/2/3 first so artifacts/ has the saved models)
python -m src.latency_benchmark --artifacts-dir ./artifacts \
    --sample-image /path/to/any_test_image.png --n-runs 100
```

Each script writes its results as JSON into `--output-dir` (e.g., `stage1_cv_results.json`, `stage1_ablation_results.json`, `latency_benchmark_results.json`) so you can pull the numbers straight into the paper's tables.



```bash
pip install -r requirements.txt

# Stage 1
python -m src.stage1_palm_screening --data-dir /path/to/Palm --output-dir ./artifacts

# Stage 2
python -m src.stage2_rbc_anemia_detection \
    --healthy-dir /path/to/AneRBC-II/Healthy_individuals/RGB_segmented \
    --anemic-dir  /path/to/AneRBC-II/Anemic_individuals/RGB_segmented \
    --cap-per-class 6000 --output-dir ./artifacts

# Stage 3
python -m src.stage3_morphology_classification \
    --normal-dir /path/to/normal --abnormal-dir /path/to/abnormal \
    --cap-per-class 6000 --output-dir ./artifacts

# Inference on a new image
python -m src.inference_pipeline --artifacts-dir ./artifacts \
    --palm-image sample_palm.jpg --rbc-image sample_rbc.png
```

Because this environment has no GPU/network access, none of this has been
executed end-to-end here — run it in Colab/locally with the actual datasets
and re-check the reported numbers, especially Stage-3, which is expected to
drop somewhat once the threshold is chosen honestly on validation data
instead of the test set.

## Real multi-class morphology classification (Chula-RBC-12)

This replaces the Stage-3 proxy-label problem entirely, using a genuine,
independently annotated RBC morphology dataset instead of reusing the
healthy/anemic split. Delivers now what the paper's "Future Work" section
described as future work.

Dataset: [Chula-RBC-12](https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset)
— 706 blood smear images, 20,875 individually labeled RBCs, 13 classes
(Normal, Macrocyte, Microcyte, Spherocyte, Target cell, Stomatocyte,
Ovalocyte, Teardrop, Burr cell, Schistocyte, uncategorised, Hypochromia,
Elliptocyte), majority-voted across multiple hematology specialists.

### Workflow

```bash
git clone https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset.git

# 1. Preview crop size before committing to a full run
python -m src.crop_chula_dataset --dataset-dir Chula-RBC-12-Dataset \
    --output-dir ./chula_cropped --crop-size 64 --preview-only
# Open ./chula_cropped/_preview/*.png and confirm whole cells are visible,
# centered, and not mostly background. Adjust --crop-size and re-run
# --preview-only until it looks right.

# 2. Full crop (excludes class 10 "uncategorised" by default — not a real
#    diagnostic category)
python -m src.crop_chula_dataset --dataset-dir Chula-RBC-12-Dataset \
    --output-dir ./chula_cropped --crop-size 64

# 3. Train the multi-class classifier (handles the severe class imbalance
#    — 6,330 Normal vs. 307 Teardrop — via class weighting)
python -m src.stage3_morphology_multiclass \
    --data-dir ./chula_cropped --output-dir ./artifacts

# 4. Run the updated inference pipeline
python -m src.inference_pipeline_v2 \
    --artifacts-dir ./artifacts \
    --palm-image path/to/palm.jpg \
    --rbc-image path/to/rbc_smear.jpg
```

**One honest limitation to know about:** the Chula-RBC-12 labels are cell-center
points, not bounding boxes, and the dataset only provides ground-truth
centers for its own training images. For a genuinely new, unlabeled smear
image at inference time, you need a cell-detection step (e.g., watershed
segmentation, as the dataset's own authors used) to find cell centers
before classifying each one. `inference_pipeline_v2.py`'s
`detect_and_crop_cells()` is currently a **stub** (single center crop only)
— plug in a real detector before relying on this for anything beyond
single-cell input images. This should be stated as a limitation/future work
item if you use this pipeline as-is in the paper.

## Cross-validation for the multi-class morphology model

```bash
python -m src.cv_stage3_multiclass --data-dir ./chula_cropped \
    --k 5 --epochs-per-fold 15 --output-dir ./artifacts
```

Reports macro-averaged precision/recall/F1 per fold plus **per-class F1**,
which matters here specifically because macro-averaging weighs every class
equally — it won't let strong performance on "Normal" (6,330 samples) hide
poor performance on "Teardrop" (307 samples) the way plain accuracy would.
This is what a Q1/Q2 reviewer will expect to see for an imbalanced
multi-class result.
