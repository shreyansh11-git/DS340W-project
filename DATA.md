# Dataset Documentation — Parent Paper

**Parent paper:** Khan, A.H.; Shah, A.; Ali, A.; Shahid, R.; Zahid, Z.U.; Sharif, M.U.; Jan, T.;
Zafar, M.H. "A performance comparison of machine learning models for stock market prediction with
novel investment strategy." *PLOS ONE* 18(9): e0286362, 21 September 2023.
DOI: https://doi.org/10.1371/journal.pone.0286362

---

## 1. Where the paper says its data comes from

The paper identifies its data source in four separate places:

| Location | What it states |
|---|---|
| Methodology, p. 4 | "The first step is to import the stock market data from **Alpaca broker**" |
| Fig 1 (flow chart) | First node of the pipeline is labelled **"Alpaca Broker"** |
| Fig 2 caption, p. 8 | "Imported stock prices of Tesla Inc. from **Alpaca broker**" |
| Table 5, p. 7 | Exact ranges: total 2016-01-01 to 2021-12-31; train 2016-01-01 to 2020-11-15; test 2020-11-16 to 2021-12-31 |
| Table 6, p. 8 | Data Sources: "Historical stock market data from **Alpaca Broker**"; Type: "Stock market historical prices from Jan 2016 to Dec 2021" |

**Instrument:** Tesla Inc. (**TSLA**), a single US-listed equity.
**Two data arms:**
1. **1-day time frame** — daily bars (the traditional baseline).
2. **15-min strategy** — the *first 15-minute bar after the market open* on each trading day.
   This is the paper's central contribution.

---

## 2. Primary dataset link (the source the paper used)

**Alpaca Market Data API — Historical Bars**
- API docs: https://docs.alpaca.markets/us/reference/stockbars
- Plan comparison / coverage: https://docs.alpaca.markets/us/docs/about-market-data-api
- Free account signup: https://alpaca.markets/
- Python SDK: https://alpaca.markets/sdks/python/market_data.html

**Access verified (Sept 2026):**
- Historical coverage: **since 2016** — covers the paper's full window
- Cost: **free** on the Basic plan; no credit card required
- Rate limit: 200 requests/min (Basic)
- Authentication: free API key + secret from the Alpaca dashboard

**Documented caveat.** On the free Basic plan, US equity data is served from the **IEX feed**
rather than the full SIP consolidated tape. IEX carries roughly 2–3% of US equity volume, so
intraday bars will differ somewhat from the consolidated prices the authors likely had in 2022.
Daily bars are close to identical. This is a deviation to document in the report, not a blocker —
and comparing IEX vs consolidated is itself a defensible robustness check.

---

## 3. Secondary / cross-check source

For the **daily** arm only, the same series is freely available without any API key:
- Yahoo Finance (TSLA): https://finance.yahoo.com/quote/TSLA/history
- Python: `yfinance.download("TSLA", start="2016-01-01", end="2021-12-31")`
- Stooq (direct CSV): https://stooq.com/q/d/l/?s=tsla.us&d1=20160101&d2=20211231&i=d

**Not viable for the 15-minute arm:** Yahoo Finance caps intraday history at roughly 60 days,
so it cannot reach back to 2016. Alpaca is required for arm 2.

---

## 4. How to obtain the data first-hand

    pip install alpaca-py pandas
    export APCA_API_KEY_ID="your key"
    export APCA_API_SECRET_KEY="your secret"
    python src/fetch_data.py

Produces:
- `data/TSLA_daily_2016_2021.csv`      — arm 1
- `data/TSLA_first15min_2016_2021.csv` — arm 2

Both are small (roughly 1,500 rows each, well under 1 MB), so they can be committed to the
repository and uploaded directly with the assignment.

The script prints a verification block comparing row counts against the paper's Table 5 ranges
and the ~283 test observations implied by its Table 10 confusion matrices.

---

## 5. Size

| File | Rows (expected) | Approx. size |
|---|---|---|
| TSLA_daily_2016_2021.csv | ~1,510 trading days | ~70 KB |
| TSLA_first15min_2016_2021.csv | ~1,510 trading days | ~70 KB |

Small enough to upload directly as files.
