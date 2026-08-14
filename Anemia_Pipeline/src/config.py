"""
Shared configuration for the anemia detection pipeline.

Centralizing paths/hyperparameters here means every stage script uses the
same values as the paper's Training Configuration table, and the same
values you'd cite when writing up results.
"""

import os

IMG_SIZE = 224
BATCH_SIZE = 16
RANDOM_STATE = 42

# ---- Stage-1: Palm screening ----
PALM_TEST_SPLIT = 0.20
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="binary:logistic",
    tree_method="hist",
    random_state=RANDOM_STATE,
)

# ---- Stage-2: RBC anemia detection ----
RBC_TEST_SPLIT = 0.15          # matches the split that produced the reported 1,800-image test set
RBC_VAL_SPLIT = 0.15           # taken out of the remaining training pool
RBC_INITIAL_EPOCHS = 25
RBC_FINETUNE_EPOCHS = 10
RBC_INITIAL_LR = 1e-4
RBC_FINETUNE_LR = 1e-5
RBC_UNFREEZE_LAST_N = 60       # layers unfrozen for the initial head training
RBC_FINETUNE_UNFREEZE_LAST_N = 40

# ---- Stage-3: Morphology classification ----
MORPH_TRAIN_SPLIT = 0.80
MORPH_VAL_SPLIT = 0.10
MORPH_TEST_SPLIT = 0.10
MORPH_HEAD_EPOCHS = 100
MORPH_FINETUNE_EPOCHS = 6
MORPH_INITIAL_LR = 1e-4
MORPH_FINETUNE_LR = 1e-5
MORPH_UNFREEZE_LAST_N = 60
MORPH_FINETUNE_UNFREEZE_LAST_N = 40
# Threshold search grid — IMPORTANT: this must only ever be swept against the
# validation set, never the test set. See stage3_morphology_classification.py.
MORPH_THRESHOLD_GRID_LOW = 0.30
MORPH_THRESHOLD_GRID_HIGH = 0.70
MORPH_THRESHOLD_GRID_STEPS = 100

# ---- Output locations ----
OUTPUT_DIR = os.environ.get("ANEMIA_PIPELINE_OUTPUT_DIR", "./artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
