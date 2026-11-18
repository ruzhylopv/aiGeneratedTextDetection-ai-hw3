"""Constants, paths and data loading shared by all encoder scripts (same split as the notebooks)."""

import os
from pathlib import Path

import pandas as pd

SEED = 42
LABELS = ["human", "ai", "fusion"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
VAL_FOLD = 0

REPO_DIR = Path(__file__).resolve().parents[2]
SPLITS_CSV = REPO_DIR / "splits" / "train_folds.csv"


def data_dir():
    """Folder with train.csv / test.csv / sample_submission.csv ($HW3_DATA, set by cluster/env.sh)."""
    return Path(os.environ["HW3_DATA"])


def run_dir():
    """Output folder of the current Slurm job ($RUN_DIR, set by cluster/run.sbatch)."""
    return Path(os.environ["RUN_DIR"])


def load_data():
    """Training folds, validation fold (fold 0) and test, split exactly as in the notebooks."""
    train = pd.read_csv(data_dir() / "train.csv").merge(pd.read_csv(SPLITS_CSV), on="id", validate="1:1")
    return {
        "train": train[train["fold"] != VAL_FOLD].reset_index(drop=True),
        "val": train[train["fold"] == VAL_FOLD].reset_index(drop=True),
        "test": pd.read_csv(data_dir() / "test.csv"),
    }
