"""
Data acquisition for the DS-340W parent paper.

Khan, A.H. et al. (2023) "A performance comparison of machine learning models for
stock market prediction with novel investment strategy." PLOS ONE 18(9): e0286362.
https://doi.org/10.1371/journal.pone.0286362

SOURCE (paper, Table 5 / Table 6 / Fig 2): Alpaca Broker historical market data,
Tesla Inc. (TSLA), 1 Jan 2016 - 31 Dec 2021.

This script reproduces BOTH data arms of the paper:
  Arm 1 "1-day time frame"  -> daily bars
  Arm 2 "15-min strategy"   -> the FIRST 15-minute bar after the open, each trading day

Setup
-----
    pip install alpaca-py pandas
    export APCA_API_KEY_ID="..."       # free key: https://alpaca.markets/
    export APCA_API_SECRET_KEY="..."
    python src/fetch_data.py
"""
import os
import sys
import pandas as pd

START = "2016-01-01"
END   = "2021-12-31"
SYMBOL = "TSLA"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def client():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError:
        sys.exit("pip install alpaca-py")
    k = os.environ.get("APCA_API_KEY_ID")
    s = os.environ.get("APCA_API_SECRET_KEY")
    if not k or not s:
        sys.exit(
            "Set your Alpaca keys first:\n"
            "  1. Sign up free at https://alpaca.markets/\n"
            "  2. Dashboard -> API Keys -> Generate\n"
            '  3. export APCA_API_KEY_ID="..."\n'
            '     export APCA_API_SECRET_KEY="..."'
        )
    return StockHistoricalDataClient(k, s)


def fetch(tf, label):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tfmap = {
        "day": TimeFrame.Day,
        "15min": TimeFrame(15, TimeFrameUnit.Minute),
    }
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=tfmap[tf],
        start=pd.Timestamp(START, tz="America/New_York"),
        end=pd.Timestamp(END + " 23:59", tz="America/New_York"),
        feed="iex",          # free/Basic plan = IEX feed (see README caveat)
        adjustment="raw",
    )
    print(f"Requesting {label} bars for {SYMBOL} {START}..{END} ...")
    bars = client().get_stock_bars(req).df
    if bars.empty:
        sys.exit(f"No {label} data returned. Check keys and plan.")
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["et"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars["date"] = bars["et"].dt.date
    return bars


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- Arm 1: daily bars (paper's "1-day time frame")
    daily = fetch("day", "daily")
    daily_out = daily[["date", "open", "high", "low", "close", "volume"]].copy()
    p1 = os.path.join(OUT, "TSLA_daily_2016_2021.csv")
    daily_out.to_csv(p1, index=False)
    print(f"  -> {len(daily_out):>6} rows  {p1}")

    # ---- Arm 2: first 15-minute bar of each trading day (the paper's contribution)
    m15 = fetch("15min", "15-minute")
    m15 = m15[m15["et"].dt.time >= pd.Timestamp("09:30").time()]
    first = m15.sort_values("et").groupby("date", as_index=False).first()
    first_out = first[["date", "open", "high", "low", "close", "volume"]].copy()
    p2 = os.path.join(OUT, "TSLA_first15min_2016_2021.csv")
    first_out.to_csv(p2, index=False)
    print(f"  -> {len(first_out):>6} rows  {p2}")

    # ---- verification against the paper
    print("\n--- Verification vs the paper ---")
    for name, df in [("daily", daily_out), ("first-15min", first_out)]:
        d = pd.to_datetime(df["date"])
        test = df[(d >= "2020-11-16") & (d <= "2021-12-31")]
        print(f"{name:>12}: {len(df):>5} total rows | "
              f"{len(test):>4} in the paper's test window (Table 10 implies ~283)")
    print("\nPaper Table 5: total 2016-01-01..2021-12-31, "
          "train ..2020-11-15, test 2020-11-16..2021-12-31")


if __name__ == "__main__":
    main()
