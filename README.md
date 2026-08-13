# Anemia Detection and RBC Morphology Analysis

Code for a hybrid multi-stage framework for anemia detection and RBC morphology
analysis, combining non-invasive palm-image screening (Stage-1) with microscopic
RBC image analysis (Stage-2), morphology classification, rule-based
interpretation, and disease mapping (Stage-3).

## 1. Repository Structure

```
anemia_pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.sh          # runs the full pipeline end-to-end (Linux/WSL)
├── run_pipeline.bat         # thin Windows launcher that hands off to WSL
└── src/
    ├── config.py            # shared hyperparameters, paths, random seed
    ├── utils.py             # shared helpers incl. set_global_seed()
    │
    ├── stage1_palm_screening.py       # Stage-1: palm images, single split
    ├── cv_stage1_palm_screening.py    # Stage-1: 5-fold cross-validation
    ├── ablation_stage1.py             # Stage-1 ablation + McNemar test
    │
    ├── stage2_rbc_anemia_detection.py # Stage-2: RBC images, single split
    ├── stage3_morphology_classification.py  # Stage-3 binary, single split
    ├── cv_deep_stage.py               # Stage-2/3 5-fold cross-validation
    │
    ├── crop_chula_dataset.py          # crops Chula-RBC-12 for multiclass ext.
    ├── stage3_morphology_multiclass.py # Stage-3 multiclass extension
    ├── cv_stage3_multiclass.py         # multiclass 5-fold cross-validation
    │
    ├── disease_mapping_v2.py          # rule-based morphology -> disease map
    ├── inference_pipeline.py          # end-to-end single-image inference
    ├── inference_pipeline_v2.py       # v2 inference (cell-detection stub)
    ├── latency_benchmark.py           # per-stage / full-pipeline latency
    └── statistical_tests.py           # McNemar significance testing
```

## 2. Prerequisites

- Python 3.10+ (developed and tested on 3.12)
- ~5 GB free disk space (TensorFlow + dependencies)
- Optional: NVIDIA GPU with CUDA/cuDNN for faster training. CPU-only works,
  just slower — all reported results in the paper can be reproduced on CPU.

## 3. Setup

```bash
git clone <this-repo-url>
cd anemia_pipeline
python3 -m venv venv
source venv/bin/activate        # Windows (native): venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Datasets

This repository does **not** include the datasets — they are publicly
available from their original sources and must be downloaded separately:

| Dataset | Used by | Source |
|---|---|---|
| Palm-Anemia | Stage-1 | Kaggle — Gosavi et al. |
| AneRBC-II | Stage-2, Stage-3 (binary) | Kaggle — Shahzad et al. |
| Chula-RBC-12 | Stage-3 (multiclass extension) | GitHub |

Download each and note the local path — you'll pass it via `--data-dir` (or
equivalent) to the scripts below. No specific folder location is required as
long as the path you pass matches the dataset's actual structure.

## 5. Running Individual Stages

Every script can be run standalone via `python -m src.<script_name>`, from
the project root, with the venv activated.

**Stage-1 — Palm-based screening**
```bash
python -m src.stage1_palm_screening --data-dir <path/to/Palm> --output-dir ./artifacts
python -m src.cv_stage1_palm_screening --data-dir <path/to/Palm> --k 5 --output-dir ./artifacts
python -m src.ablation_stage1 --data-dir <path/to/Palm> --output-dir ./artifacts
```

**Stage-2 — RBC-based anemia detection**
```bash
python -m src.stage2_rbc_anemia_detection \
    --healthy-dir <path/to/Healthy> --anemic-dir <path/to/Anemic> \
    --cap-per-class 6000 --output-dir ./artifacts

python -m src.cv_deep_stage --stage 2 \
    --class0-dir <path/to/Healthy> --class1-dir <path/to/Anemic> \
    --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts
```

**Stage-3 — Morphology classification (binary)**
```bash
python -m src.stage3_morphology_classification \
    --normal-dir <path/to/Normal> --abnormal-dir <path/to/Abnormal> \
    --cap-per-class 6000 --output-dir ./artifacts

python -m src.cv_deep_stage --stage 3 \
    --class0-dir <path/to/Normal> --class1-dir <path/to/Abnormal> \
    --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts
```

**Stage-3 extension — Multiclass morphology (Chula-RBC-12)**
```bash
python -m src.crop_chula_dataset --dataset-dir <path/to/Chula-RBC-12-Dataset> \
    --output-dir ./chula_cropped --crop-size 64

python -m src.stage3_morphology_multiclass --data-dir ./chula_cropped --output-dir ./artifacts
python -m src.cv_stage3_multiclass --data-dir ./chula_cropped --k 5 --epochs-per-fold 15 --output-dir ./artifacts
```

**Latency benchmark**
```bash
python -m src.latency_benchmark \
    --artifacts-dir ./artifacts --sample-image <path/to/any_image.png> --n-runs 100
```

## 6. Running the Full Pipeline (Recommended)

`run_pipeline.sh` runs all of the above in the correct order, logs each step
separately, and continues past a failed step rather than stopping (each stage
loads its own data independently, so one failure doesn't block the others).

**Before running:** open `run_pipeline.sh` and edit the path variables near
the top (`PALM_DIR`, `HEALTHY_DIR`/`ANEMIC_DIR`, `NORMAL_DIR`/`ABNORMAL_DIR`,
`CHULA_RAW_DIR`, `SAMPLE_IMAGE`) to point at your local dataset locations.

```bash
bash run_pipeline.sh
```

On Windows, `run_pipeline.bat` hands off to WSL and runs the same script
there (double-click, or run from an existing WSL install).

Expect a full run to take a few hours, dominated by the Stage-2/3
cross-validation (5 folds × several epochs of CNN fine-tuning each) — plan it
as an unattended/overnight run.

## 7. Outputs

- `./artifacts/*.json` — metrics, confusion matrices, and (for the ablation)
  per-sample predictions and McNemar test results, one file per script.
- `./logs/*.log` — one log per pipeline step when run via `run_pipeline.sh`,
  plus a timestamped `pipeline_summary_*.log` with a pass/fail table for the
  whole run.

## 8. Reproducibility

Every training/CV entry point calls `utils.set_global_seed()` first thing,
which seeds Python, NumPy, and TensorFlow/Keras, and enables TensorFlow's
deterministic-operations mode. This makes results consistent across repeated
runs **on the same hardware**. Note that GPU (cuDNN) and CPU execution are
not guaranteed to produce bit-identical results even with the same seed — if
you re-run on different hardware than was used for the paper's reported
numbers, expect results within roughly 1 accuracy point, not bit-exact.

## 9. Citation

If you use this code, please cite:

```bibtex
@article{TODO_citation_key,
  title   = {A Hybrid Multi-Stage Framework for Anemia Detection and RBC Morphology Analysis Using Palm Images and Microscopic Blood Smear Images},
  author  = {Shaik, Mabasha and Gupta, Sumit and Kancharagunta, Kishan Babu},
  journal = {TODO},
  year    = {TODO}
}
```

## 10. License

TODO — add a LICENSE file (MIT / Apache-2.0 are common choices for academic
code) before making the repository public.
