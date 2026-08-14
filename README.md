# Anemia Detection and RBC Morphology Analysis

Code for a hybrid multi-stage framework for anemia detection and RBC morphology
analysis, combining non-invasive palm-image screening (Stage-1) with microscopic
RBC image analysis (Stage-2), morphology classification, rule-based
interpretation, and disease mapping (Stage-3).

## 1\. Repository Structure

```
anemia\\\_pipeline/
├── README.md
├── requirements.txt
└── src/
    ├── config.py            # shared hyperparameters, paths, random seed
    ├── utils.py             # shared helpers incl. set\\\_global\\\_seed()
    │
    ├── stage1\\\_palm\\\_screening.py       # Stage-1: palm images, single split
    ├── cv\\\_stage1\\\_palm\\\_screening.py    # Stage-1: 5-fold cross-validation
    ├── ablation\\\_stage1.py             # Stage-1 ablation + McNemar test
    │
    ├── stage2\\\_rbc\\\_anemia\\\_detection.py # Stage-2: RBC images, single split
    ├── stage3\\\_morphology\\\_classification.py  # Stage-3 binary, single split
    ├── cv\\\_deep\\\_stage.py               # Stage-2/3 5-fold cross-validation
    │
    ├── crop\\\_chula\\\_dataset.py          # crops Chula-RBC-12 for multiclass ext.
    ├── stage3\\\_morphology\\\_multiclass.py # Stage-3 multiclass extension
    ├── cv\\\_stage3\\\_multiclass.py         # multiclass 5-fold cross-validation
    │
    ├── disease\\\_mapping\\\_v2.py          # rule-based morphology -> disease map
    ├── inference\\\_pipeline.py          # end-to-end single-image inference
    ├── inference\\\_pipeline\\\_v2.py       # v2 inference (cell-detection stub)
    ├── latency\\\_benchmark.py           # per-stage / full-pipeline latency
    └── statistical\\\_tests.py           # McNemar significance testing
```

## 2\. Prerequisites

* Python 3.10+ (developed and tested on 3.12)
* \~5 GB free disk space (TensorFlow + dependencies)
* Optional: NVIDIA GPU with CUDA/cuDNN for faster training. CPU-only works,
just slower — all reported results in the paper can be reproduced on CPU.

## 3\. Setup

```bash
git clone <this-repo-url>
cd anemia\\\_pipeline
python3 -m venv venv
source venv/bin/activate        # Windows (native): venv\\\\Scripts\\\\activate
pip install -r requirements.txt
```

## 4\. Datasets

This repository does **not** include the datasets — they are publicly
available from their original sources and must be downloaded separately:

|Dataset|Used by|Source|
|-|-|-|
|Palm-Anemia|Stage-1|Kaggle — Gosavi et al.|
|AneRBC-II|Stage-2, Stage-3 (binary)|Kaggle — Shahzad et al.|
|Chula-RBC-12|Stage-3 (multiclass extension)|GitHub|

Download each and note the local path — you'll pass it via `--data-dir` (or
equivalent) to the scripts below. No specific folder location is required as
long as the path you pass matches the dataset's actual structure.

## 5\. Running Individual Stages

Every script can be run standalone via `python -m src.<script\\\_name>`, from
the project root, with the venv activated.

**Stage-1 — Palm-based screening**

```bash
python -m src.stage1\\\_palm\\\_screening --data-dir <path/to/Palm> --output-dir ./artifacts
python -m src.cv\\\_stage1\\\_palm\\\_screening --data-dir <path/to/Palm> --k 5 --output-dir ./artifacts
python -m src.ablation\\\_stage1 --data-dir <path/to/Palm> --output-dir ./artifacts
```

**Stage-2 — RBC-based anemia detection**

```bash
python -m src.stage2\\\_rbc\\\_anemia\\\_detection \\\\
    --healthy-dir <path/to/Healthy> --anemic-dir <path/to/Anemic> \\\\
    --cap-per-class 6000 --output-dir ./artifacts

python -m src.cv\\\_deep\\\_stage --stage 2 \\\\
    --class0-dir <path/to/Healthy> --class1-dir <path/to/Anemic> \\\\
    --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts
```

**Stage-3 — Morphology classification (binary)**

```bash
python -m src.stage3\\\_morphology\\\_classification \\\\
    --normal-dir <path/to/Normal> --abnormal-dir <path/to/Abnormal> \\\\
    --cap-per-class 6000 --output-dir ./artifacts

python -m src.cv\\\_deep\\\_stage --stage 3 \\\\
    --class0-dir <path/to/Normal> --class1-dir <path/to/Abnormal> \\\\
    --cap-per-class 6000 --k 5 --epochs-per-fold 8 --output-dir ./artifacts
```

**Stage-3 extension — Multiclass morphology (Chula-RBC-12)**

```bash
python -m src.crop\\\_chula\\\_dataset --dataset-dir <path/to/Chula-RBC-12-Dataset> \\\\
    --output-dir ./chula\\\_cropped --crop-size 64

python -m src.stage3\\\_morphology\\\_multiclass --data-dir ./chula\\\_cropped --output-dir ./artifacts
python -m src.cv\\\_stage3\\\_multiclass --data-dir ./chula\\\_cropped --k 5 --epochs-per-fold 15 --output-dir ./artifacts
```

**Latency benchmark**

```bash
python -m src.latency\\\_benchmark \\\\
    --artifacts-dir ./artifacts --sample-image <path/to/any\\\_image.png> --n-runs 100
```

## 6\. Running the Full Pipeline (Recommended)

`run\\\_pipeline.sh` runs all of the above in the correct order, logs each step
separately, and continues past a failed step rather than stopping (each stage
loads its own data independently, so one failure doesn't block the others).

**Before running:** open `run\\\_pipeline.sh` and edit the path variables near
the top (`PALM\\\_DIR`, `HEALTHY\\\_DIR`/`ANEMIC\\\_DIR`, `NORMAL\\\_DIR`/`ABNORMAL\\\_DIR`,
`CHULA\\\_RAW\\\_DIR`, `SAMPLE\\\_IMAGE`) to point at your local dataset locations.

```bash
bash run\\\_pipeline.sh
```

On Windows, `run\\\_pipeline.bat` hands off to WSL and runs the same script
there (double-click, or run from an existing WSL install).

Expect a full run to take a few hours, dominated by the Stage-2/3
cross-validation (5 folds × several epochs of CNN fine-tuning each) — plan it
as an unattended/overnight run.

## 7\. Outputs

* `./artifacts/\\\*.json` — metrics, confusion matrices, and (for the ablation)
per-sample predictions and McNemar test results, one file per script.
* `./logs/\\\*.log` — one log per pipeline step when run via `run\\\_pipeline.sh`,
plus a timestamped `pipeline\\\_summary\\\_\\\*.log` with a pass/fail table for the
whole run.

## 8\. Reproducibility

Every training/CV entry point calls `utils.set\\\_global\\\_seed()` first thing,
which seeds Python, NumPy, and TensorFlow/Keras, and enables TensorFlow's
deterministic-operations mode. This makes results consistent across repeated
runs **on the same hardware**. Note that GPU (cuDNN) and CPU execution are
not guaranteed to produce bit-identical results even with the same seed — if
you re-run on different hardware than was used for the paper's reported
numbers, expect results within roughly 1 accuracy point, not bit-exact.

## 9\. License

TODO — add a LICENSE file (MIT / Apache-2.0 are common choices for academic
code) before making the repository public.

