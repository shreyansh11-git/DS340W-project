"""
Loader for the Train / Test / Validation splits.

The validation set is a HELD-OUT set. This module refuses to hand it over unless
you explicitly pass allow_holdout=True, so it cannot be loaded by accident during
development.

    from src.data import load_split
    tr = load_split("train")                 # fine
    te = load_split("test")                  # fine
    va = load_split("validation")            # raises RuntimeError
    va = load_split("validation", allow_holdout=True)   # deliberate, final run only
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
DIRS = {"train": "train", "test": "test", "validation": "HOLDOUT_validation"}


def manifest():
    with open(os.path.join(SPLITS, "manifest.json")) as fh:
        return json.load(fh)


def load_split(split, arm="daily", allow_holdout=False):
    """split: train|test|validation.  arm: daily|first15."""
    if split not in DIRS:
        raise ValueError(f"split must be one of {list(DIRS)}, got {split!r}")
    if split == "validation" and not allow_holdout:
        raise RuntimeError(
            "The validation set is held out.\n"
            "Do not load it while developing, tuning, or debugging - every look at it\n"
            "costs you the ability to claim the final number is honest.\n"
            "When the pipeline is finished and Train/Test results are final, call:\n"
            "    load_split('validation', allow_holdout=True)"
        )
    path = os.path.join(SPLITS, DIRS[split], f"TSLA_{arm}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}\nRun: python src/make_splits.py")
    return pd.read_csv(path, parse_dates=["date"])
