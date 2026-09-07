# Phase 1 — Executing the Parent Paper's Published Code

**Parent paper:** Khan, A.H. et al. (2023), *"A performance comparison of machine learning
models for stock market prediction with novel investment strategy,"* PLOS ONE 18(9): e0286362.
https://doi.org/10.1371/journal.pone.0286362

**Code source:** the authors' own repository, cited in their Supporting Information (S1 File):
https://github.com/AzazHassankhan/Machine-Learning-based-Trading-Techniques

## Files

| File | What it is |
|---|---|
| `ORIGINAL_UNMODIFIED.ipynb` | Pristine download from the authors' repo, untouched |
| `Machine_Learning_based_Backtester.ipynb` | The runnable version — 2 cells changed, see `PATCH_NOTES.md` |
| `PATCH_NOTES.md` | Exactly what changed and why |
| `requirements.txt` | Pinned environment |
| `TSLA_daily_2016_2021.csv` | Data, from the same Alpaca source the paper used |
| `TSLA_first15min_2016_2021.csv` | The paper's 15-minute arm |

## Setup

```bash
conda create -n ds340w python=3.11 -y
conda activate ds340w
pip install -r requirements.txt
```

Pandas is pinned to 1.5.3 deliberately — the original code uses integer indexing that was
positional in pandas 1.x and is label-based in pandas 2+. Pinning lets the authors' code
run as written instead of being edited.

## Run

```bash
jupyter notebook Machine_Learning_based_Backtester.ipynb
```

Then Kernel -> Restart & Run All.

## Verified result

Executes end-to-end, 14/14 cells, zero errors:

```
accuracy                 0.8233
test samples             283
weighted F1              0.82
trades (buy / sell)      70 / 72
profitable / stopped     117 / 24
reported profitability   0.8298
```

## Important caveat

This notebook is **not** a full implementation of the paper. Audited against the published
method, it:

- runs **1 of the 9 models** the paper compares (the other five are commented out; XGBoost,
  ANN and Naive Bayes appear nowhere)
- does **not** implement the paper's central contribution, the first-15-minutes-after-open filter
- was last committed **19 Feb 2022**, eighteen months before the paper published
- contains **lookahead bias**: three features use `.shift(-1)`, pulling next-period values into
  the current row, while the paper's own Table 2 specifies these as *"Previous"* (lagged).
  The reported accuracy is therefore inflated.

These findings are documented as the basis for the Phase 2 extension.
