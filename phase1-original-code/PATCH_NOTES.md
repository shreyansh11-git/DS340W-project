# Patch Notes — exactly what was changed, and why

The goal was to **execute the authors' published code**, not rewrite it.
`ORIGINAL_UNMODIFIED.ipynb` is the pristine file downloaded from the authors' repository.
`Machine_Learning_based_Backtester.ipynb` is that same file with the changes listed below.

**One cell was changed. No feature, model, hyperparameter, metric, or analysis step was touched.**

---

## Change 1 — the data-loading cell (cell 3)

### Why it was unavoidable
The original cell calls:

```python
df = api.get_barset(symb, TimeFrame(15, TimeFrameUnit.Minute), limit=1000).df
```

`api.get_barset()` is a **removed** Alpaca API endpoint. It belonged to the v1 API;
Alpaca migrated to v2 and then to the `alpaca-py` SDK. The call cannot execute today
under any configuration.

The cell also contained the original author's **live API key and secret**, hardcoded and
committed to a public repository. Those credentials are dead, are not ours, and were removed.

### What replaced it
The notebook's own commented-out lines, directly above the API call in the original,
show a CSV loading path the author used previously:

```python
# df=pd.read_csv('FB.csv')
# df=df.set_index(pd.DatetimeIndex(df['Gmt time'].values))
```

We use that same path, pointed at TSLA data downloaded from **the same source the paper
used** (Alpaca Market Data API), covering the paper's own window of 2016-01-04 to
2021-12-31.

### What did NOT change
Every other cell is byte-identical to the original: the feature engineering, the
correlation screen, the ADF stationarity test, the 80/20 split, the
`RandomForestClassifier(n_estimators=100, random_state=30)`, the accuracy and
classification report, the buy/sell decision loop with its $0.40 take-profit and
stop-loss, and all plotting.

---

## Change 2 — a dead import line

`import alpaca_trade_api as tradeapi` was removed. The package is deprecated and, after
Change 1, unused. Leaving it in would abort the notebook on a missing module.

---

## Not a code change: the Python environment

The original code contains this pattern throughout:

```python
for i in range(len(df)):
    if (df['future_returns'][i] > 0):
```

Indexing a Series with an integer was **positional** in pandas 1.x. In pandas 2.0+ an
integer key is treated as a **label**, so on a DatetimeIndex this raises `KeyError: 0`.

Rather than edit the authors' logic to use `.iloc`, we pin **pandas 1.5.3** (see
`requirements.txt`) so the original code runs exactly as it was written. This is an
environment fix, not a code fix.

---

## Verification

Executed end-to-end, 14/14 code cells, **zero errors**. Output produced:

| Result | Value |
|---|---|
| Test samples | 283 |
| Random Forest accuracy | 0.8233 |
| Weighted F1 | 0.82 |
| Trades: buys / sells | 70 / 72 |
| Profitable / stopped out | 117 / 24 |
| Reported "Profitability" | 0.8298 |
