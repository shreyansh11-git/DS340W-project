DS-340W - TSLA Dataset Splits (Train / Test / Validation)
=========================================================

PARENT PAPER
  Khan, A.H. et al. (2023). "A performance comparison of machine learning models
  for stock market prediction with novel investment strategy."
  PLOS ONE 18(9): e0286362.  https://doi.org/10.1371/journal.pone.0286362

DATA SOURCE
  Alpaca Market Data API (SIP consolidated feed) - the same provider the paper
  used (its Methodology p.4, Fig 1, Fig 2, Tables 5 and 6).
  https://docs.alpaca.markets/us/reference/stockbars

  Instrument : Tesla Inc. (TSLA)
  Coverage   : 2016-01-04 to 2021-12-31, 1,511 trading days
  Prices     : RAW / unadjusted, matching the paper's Fig 2 convention
  Columns    : date, open, high, low, close, volume

TWO DATA ARMS (identical trading calendar)
  TSLA_daily_*.csv      Daily bars - the paper's "1-day time frame"
  TSLA_first15min_*.csv First 15-minute bar after each market open -
                        the paper's proposed strategy

SPLIT DESIGN
  Chronological, never shuffled. This is a time series: a random split would
  place future prices in the training set and leak information backwards.
  The validation set is the MOST RECENT slice, mimicking real deployment.

  train        1057 rows   70.0%   2016-01-04 .. 2020-03-16
  test          302 rows   20.0%   2020-03-17 .. 2021-05-26
  validation    152 rows   10.1%   2021-05-27 .. 2021-12-31
  TOTAL        1511 rows  100.0%

  The validation set is HELD OUT and has not been inspected. It will not be
  touched until the pipeline is complete and Train/Test results are final.

VERIFICATION
  - The three splits reassemble byte-identical to the source files
  - Zero date overlap between any pair of splits; no gaps; no nulls
  - Trading-day count reconciles: 1,565 business days - 54 market holidays
  - Split-adjusted annual returns reproduce published figures to two decimals:
      2017 45.70% | 2018 6.89% | 2019 25.70% | 2020 743.44% | 2021 49.76%

KNOWN ISSUE (documented, not corrected at this stage)
  Tesla's 5-for-1 stock split (2020-08-31) falls inside the test window. In raw
  prices this appears as a -77.5% single-day move that did not occur. Raw data
  is retained deliberately for the reproduction phase, to match the parent
  paper's own convention; split-adjusted prices will be introduced as a
  documented improvement in the extension phase.
