# DS-340W Term Project — Reproducing Khan et al. (2023)

Reproduction and extension of a peer-reviewed machine-learning study on US equity
price-direction prediction.

**Parent paper:** Khan, A.H.; Shah, A.; Ali, A.; Shahid, R.; Zahid, Z.U.; Sharif, M.U.;
Jan, T.; Zafar, M.H. *"A performance comparison of machine learning models for stock
market prediction with novel investment strategy."*
**PLOS ONE** 18(9): e0286362 (21 Sept 2023) — peer-reviewed, open access.
[DOI 10.1371/journal.pone.0286362](https://doi.org/10.1371/journal.pone.0286362)

The paper compares nine ML classifiers predicting Tesla's next-day price direction under
two designs: a conventional end-of-day time frame, and a proposed strategy using only the
**first 15 minutes after the market open**. It evaluates both with classification metrics
and a portfolio simulation.

---

## Data

**Source:** [Alpaca Market Data API](https://docs.alpaca.markets/us/reference/stockbars) —
the same provider the paper used (its Methodology p. 4, Fig 1, Fig 2, Tables 5 and 6).
Free tier, SIP consolidated feed.

**Coverage:** TSLA, 2016-01-04 → 2021-12-31, **1,511 trading days**.

Two arms sharing an identical trading calendar:

| File | Contents |
|---|---|
| `data/TSLA_daily_2016_2021.csv` | Daily bars — the paper's "1-day time frame" |
| `data/TSLA_first15min_2016_2021.csv` | First 15-min bar after each open — the paper's contribution |

Both carry raw OHLCV **and** `adj_*` split-adjusted columns. See [DATA.md](DATA.md) for full
source documentation.

**Validation against external ground truth:** split-adjusted annual returns reproduce
published figures to two decimal places for every computable year (2017 45.70%, 2018 6.89%,
2019 25.70%, 2020 743.44%, 2021 49.76%). Trading-day count reconciles: 1,565 business days
minus 54 market holidays.

### Why adjusted prices are included
Tesla's 5-for-1 split (2020-08-31) falls inside the test window. In raw prices it reads as a
−77.5% single-day return that never happened, leaving the paper's 50-period SMA feature
sitting ~2.9x above price for 50 sessions. Adjusted, that ratio is 0.71x. Raw columns are
preserved so the paper's exact setup remains reproducible.

---

## Splits

Chronological — **never shuffled**. On a time series a random split leaks future prices
into training.

| Split | Rows | % | Range |
|---|---|---|---|
| Train | 1,057 | 70.0% | 2016-01-04 → 2020-03-16 |
| Test | 302 | 20.0% | 2020-03-17 → 2021-05-26 |
| **Validation (held out)** | 152 | 10.1% | 2021-05-27 → 2021-12-31 |

Validation is the most recent slice, mimicking real deployment. It is **held out** and
enforced in code: `src/data.py` raises `RuntimeError` unless `allow_holdout=True` is passed
explicitly.

Verified: the three splits reassemble byte-identical to the source, with zero date overlap.

---

## Setup

```bash
pip install alpaca-py pandas

# Alpaca keys (free: https://alpaca.markets/) go in a .env file — never committed
cat > .env <<'ENV'
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
ENV

python src/fetch_data.py      # download both arms (auto-detects the SIP feed)
python src/adjust_prices.py   # add split-adjusted columns + verify
python src/make_splits.py     # build train / test / validation
```

```python
from src.data import load_split
train = load_split("train")                  # ok
test  = load_split("test", arm="first15")    # ok
val   = load_split("validation")             # raises — held out by design
```

---

## Layout

```
DATA.md                   dataset documentation and provenance
phase1-original-code/     the authors' published notebook, made runnable (see its README)
src/fetch_data.py         Alpaca download, both arms
src/adjust_prices.py      split adjustment with verification
src/make_splits.py        chronological splitting
src/data.py               loader with held-out guard
data/                     datasets and splits/
results/                  tables and figures (generated)
```

## Status
- [x] Parent paper selected
- [x] Data source identified, access verified, datasets downloaded
- [x] Train / test / validation splits
- [x] Phase 1a — executed the authors' published code (see phase1-original-code/)
- [ ] Phase 1b — full reproduction of the paper's method
- [ ] Phase 2 — extension

## Note on tooling
Portions of this repository were developed with AI assistance (Claude), which is permitted
for this course. Commits carrying a `Co-Authored-By: Claude` trailer indicate that.
