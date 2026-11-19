"""Constants, paths and data loading shared by all encoder scripts (same split as the notebooks).

Only pandas is needed here, so the laptop notebooks import this module too (to build the split roles).
"""

import os
from pathlib import Path

import pandas as pd

SEED = 42
LABELS = ["human", "ai", "fusion"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
VAL_FOLD = 0

N_TRAIN_SAMPLE = 50_000  # the fixed training sample of the baseline and Branch A
N_STACKER = 30_000       # held-out texts for the curve stacker

REPO_DIR = Path(__file__).resolve().parents[2]
SPLITS_CSV = REPO_DIR / "splits" / "train_folds.csv"
ROLES_CSV = REPO_DIR / "splits" / "curve_roles.csv"  # id, role (detector / stacker)
GATE_CSV = REPO_DIR / "splits" / "gate_texts.csv"    # id, set (val_plot / stacker_slice)


def data_dir():
    """Folder with train.csv / test.csv / sample_submission.csv ($HW3_DATA, set by cluster/env.sh)."""
    return Path(os.environ["HW3_DATA"])


def run_dir():
    """Output folder of the current Slurm job ($RUN_DIR, set by cluster/run.sbatch)."""
    return Path(os.environ["RUN_DIR"])


def models_dir():
    """Folder with the checkpoints that later jobs reuse ($HW3_MODELS, set by cluster/env.sh)."""
    return Path(os.environ["HW3_MODELS"])


def branch_a_dir():
    """The tuned 3-class XLM-R of hw3-encoder-tuning.ipynb, as downloaded by `make -C cluster branch-a`."""
    return models_dir() / "branch_a" / "encoder" / "model"


def load_data(data_path=None):
    """Training folds, validation fold (fold 0) and test, split exactly as in the notebooks."""
    data_path = Path(data_path) if data_path else data_dir()
    train = pd.read_csv(data_path / "train.csv").merge(pd.read_csv(SPLITS_CSV), on="id", validate="1:1")
    return {
        "train": train[train["fold"] != VAL_FOLD].reset_index(drop=True),
        "val": train[train["fold"] == VAL_FOLD].reset_index(drop=True),
        "test": pd.read_csv(data_path / "test.csv"),
    }


def stratified_sample(df, n, seed=SEED):
    """Fixed random sample of n rows with the label × language shares of df (same as the notebooks)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    strata = df["label"] + "|" + df["lang"]
    return (df.groupby(strata, group_keys=False)
              .sample(frac=n / len(df), random_state=seed)
              .reset_index(drop=True))


def prefix_key(texts, n=200):
    """Lower-cased first n characters with whitespace collapsed: the human–fusion twin key of the EDA."""
    return texts.str[: 2 * n].str.lower().str.replace(r"\s+", " ", regex=True).str.strip().str[:n]


def sample_with_shares(pool, n, shares, seed=SEED):
    """About n rows of pool, drawn per label × language stratum so the strata follow `shares`."""
    strata = pool["label"] + "|" + pool["lang"]
    counts = (shares * n).round().astype(int)
    parts = [pool[strata == stratum].sample(count, random_state=seed) for stratum, count in counts.items()]
    return pd.concat(parts).sort_index()


def build_roles(train_fold):
    """Detector and stacker texts of the curve pipeline, as an (id, role) table.

    detector: human and ai texts of the fixed 50k sample (Branch A was trained on the whole sample).
    stacker: texts outside the 50k sample whose 200-character prefix matches no sample text (twin guard),
    drawn with the label × language shares of the training folds.
    """
    sample = stratified_sample(train_fold, N_TRAIN_SAMPLE)
    detector = sample[sample["label"].isin(["human", "ai"])]
    seen_prefixes = set(prefix_key(sample["text"]))
    unseen = ~train_fold["id"].isin(sample["id"]) & ~prefix_key(train_fold["text"]).isin(seen_prefixes)
    shares = (train_fold["label"] + "|" + train_fold["lang"]).value_counts(normalize=True)
    stacker = sample_with_shares(train_fold[unseen], N_STACKER, shares)
    return pd.concat([detector[["id"]].assign(role="detector"), stacker[["id"]].assign(role="stacker")],
                     ignore_index=True)


def load_sets(data_path=None):
    """Texts of every set of the curve pipeline: detector, stacker, val and test."""
    data = load_data(data_path)
    roles = pd.read_csv(ROLES_CSV)
    sets = {role: data["train"][data["train"]["id"].isin(roles.loc[roles["role"] == role, "id"])]
            .reset_index(drop=True) for role in ("detector", "stacker")}
    return {**sets, "val": data["val"], "test": data["test"]}


def load_gate():
    """Texts of the gate check, with their `set` column (val_plot / stacker_slice)."""
    data = load_data()
    gate = pd.read_csv(GATE_CSV)
    return pd.concat([data["train"], data["val"]]).merge(gate, on="id", validate="1:1").reset_index(drop=True)
