"""
Replication of Khan et al. (2023), Tables 8 and 9: test accuracy of nine classifiers on TSLA,
for the 1-day strategy (daily bars) and the proposed 15-minute strategy (first 15-minute bar).

Every choice that differs between the paper's pipeline and ours is a field of `Config`, so a
run is defined by one line and any block can be swapped later without rewriting the script.

    PUBLISHED   the pipeline as the authors ran it: their split date, raw (unadjusted) prices,
                Min-Max scaling fitted on all data before splitting, default k-fold grid search,
                and the "Previous" features shifted to the NEXT day as in their published code.
    PAPER_TEXT  identical, except the "Previous" features use the PREVIOUS day, as the paper's
                Table 2 says. The gap between the two is the effect of the lookahead.

Note: the paper's test window (2020-11-16 to 2021-12-31) overlaps the old held-out split.
These runs only replicate the authors' published setup; the final untouched test set for
our own methods is TSLA data from 2022 onward.

Usage:  python experiments/replicate_khan.py
Writes: results/replicate_khan.csv, results/replicate_khan.md
"""
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.khan_features import FEATURES, build  # noqa: E402
from src.khan_models import MODELS, fit_predict  # noqa: E402

DATA = {"daily": "TSLA_daily_2016_2021.csv", "first15": "TSLA_first15min_2016_2021.csv"}

# Tables 8 (1-day) and 9 (15-min) of Khan et al. (2023), test accuracy in %
PAPER = {
    "daily": {"Logistic Regression": 85.51, "XGBoost": 84.80, "Random Forest": 84.45, "ANN": 84.45,
              "AdaBoost": 83.74, "Decision Tree": 83.01, "SVM": 82.68, "KNN": 79.15, "Naive Bayes": 73.49},
    "first15": {"Random Forest": 91.27, "AdaBoost": 90.93, "XGBoost": 90.93, "Logistic Regression": 90.60,
                "ANN": 89.93, "SVM": 88.59, "Decision Tree": 88.10, "Naive Bayes": 81.54, "KNN": 80.53},
}


@dataclass(frozen=True)
class Config:
    name: str
    prev_mode: str = "lookahead"     # lagged | same_day | lookahead   (src/khan_features.py)
    split_date: str = "2020-11-16"   # first test day (paper Table 5)
    price: str = ""                  # "" = raw Alpaca prices, "adj_" = split-adjusted
    scale: str = "all_data"          # all_data (paper, before split) | train_only
    cv: str = "kfold"                # kfold (paper) | timeseries         (src/khan_models.py)


PUBLISHED = Config("published")
PAPER_TEXT = replace(PUBLISHED, name="paper_text", prev_mode="lagged")
RUNS = [PUBLISHED, PAPER_TEXT]


def load(arm):
    return pd.read_csv(os.path.join(ROOT, "data", DATA[arm]), parse_dates=["date"])


def minmax(tr, te, all_rows, how):
    """Min-Max scaling (paper Eq. 1). how='all_data' fits on every row, as in the paper."""
    ref = all_rows if how == "all_data" else tr
    lo, hi = ref.min(), ref.max()
    span = (hi - lo).replace(0, 1)
    return (tr - lo) / span, (te - lo) / span


def run(cfg, arm):
    feat = build(load(arm), prev_mode=cfg.prev_mode, price=cfg.price)
    cut = pd.Timestamp(cfg.split_date)
    tr, te = feat[feat["date"] < cut], feat[feat["date"] >= cut]
    Xtr, Xte = minmax(tr[FEATURES], te[FEATURES], feat[FEATURES], cfg.scale)
    ytr, yte = tr["target"].astype(int), te["target"].astype(int)
    rows = []
    for name in MODELS:
        pred = fit_predict(name, Xtr, ytr, Xte, cv=cfg.cv)
        rows.append(dict(run=cfg.name, arm=arm, model=name, accuracy=100 * accuracy_score(yte, pred),
                         paper=PAPER[arm].get(name, np.nan), n_test=len(yte)))
        print(f"{cfg.name:11s} {arm:8s} {name:20s} {rows[-1]['accuracy']:6.2f}  (paper {rows[-1]['paper']})")
    rows.append(dict(run=cfg.name, arm=arm, model="Always-up baseline", accuracy=100 * yte.mean(),
                     paper=np.nan, n_test=len(yte)))
    return rows


def write_table(res, out):
    """Paper vs replication table, one column per (strategy, run)."""
    wide = res.pivot_table(index="model", columns=["arm", "run"], values="accuracy")
    for arm in DATA:
        wide[(arm, "paper")] = wide.index.map(PAPER[arm])
    wide = wide[[(a, r) for a in DATA for r in ("paper", PUBLISHED.name, PAPER_TEXT.name)]]
    models = [m for m in wide.index if m != "Always-up baseline"]
    wide.loc["Average (9 models)"] = wide.loc[models].mean()
    wide = wide.reindex(list(PAPER["daily"]) + ["Average (9 models)", "Always-up baseline"])
    label = {"daily": "1-day", "first15": "15-min"}
    wide.columns = [f"{label[a]} {r}" for a, r in wide.columns]
    with open(out + ".md", "w") as fh:
        fh.write("# Replication of Khan et al. (2023), Tables 8 and 9\n\n")
        fh.write("TSLA test accuracy (%), test period 2020-11-16 to 2021-12-31 (paper Table 5).\n")
        fh.write("paper = reported in Tables 8 and 9; published = authors' pipeline including next-day "
                 "'Previous' features; paper_text = same, with previous-day features as the paper states.\n\n")
        fh.write(wide.round(2).to_markdown() + "\n")
    return wide


def main():
    res = pd.DataFrame([r for cfg in RUNS for arm in DATA for r in run(cfg, arm)])
    out = os.path.join(ROOT, "results", "replicate_khan")
    res.to_csv(out + ".csv", index=False)
    print(write_table(res, out).round(2).to_string())


if __name__ == "__main__":
    main()
