# DS-340W Term Project — Parent Paper Implementation

**Parent paper:** Khan et al. (2023), "A performance comparison of machine learning models for
stock market prediction with novel investment strategy," *PLOS ONE* 18(9): e0286362.
https://doi.org/10.1371/journal.pone.0286362

**Topic:** US equity markets — machine-learning prediction of Tesla (TSLA) price direction,
comparing a conventional end-of-day strategy against a proposed first-15-minutes-after-open strategy.

## Data
See [DATA.md](DATA.md) for the full data-source documentation, links, and access instructions.
Source: Alpaca Market Data API (free tier), TSLA, 2016-01-01 to 2021-12-31.

## Layout
    src/fetch_data.py   data acquisition (both arms)
    data/               downloaded CSVs
    notebooks/          exploration
    results/            tables and figures
    docs/               paper PDF and notes

## Status
- [x] Parent paper selected
- [x] Data source identified and access verified
- [ ] Phase 1 — reproduce the paper
- [ ] Phase 2 — extension

## Data splits
Chronological (never shuffled - this is a time series). Validation is the most
recent slice and is held out.

| Split | Rows | % | Range |
|---|---|---|---|
| Train | 1057 | 70.0% | 2016-01-04 to 2020-03-16 |
| Test | 302 | 20.0% | 2020-03-17 to 2021-05-26 |
| Validation (held out) | 152 | 10.1% | 2021-05-27 to 2021-12-31 |

Rebuild with `python src/make_splits.py`. Load with `src/data.py`, which refuses to
serve the validation set unless `allow_holdout=True` is passed explicitly.

KNOWN ISSUE: Tesla's 5-for-1 split (2020-08-31) falls inside the test window, and
prices are raw/unadjusted. See the split-adjustment note before modelling.
