"""
Intraday data for parent paper 3:
Li, Sakkas & Urquhart (2022), "Intraday time series momentum: Global evidence and
links to market characteristics", Journal of Financial Markets.

Downloads 30-minute bars from the Alpaca Market Data API (same provider and .env keys
as src/fetch_data.py) and saves only regular-session bars (09:30-16:00 New York time).

    python src/fetch_intraday.py                 # SPY, 2016-01-01 .. 2025-12-31
    python src/fetch_intraday.py TSLA 2016-01-01 2021-12-31

Output: data/intraday/<SYMBOL>_30min_<start>_<end>.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fetch_data import client, preflight  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "intraday")


def fetch_30min(symbol, start, end, feed="sip"):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    cli = client()
    req = StockBarsRequest(symbol_or_symbols=symbol,
                           timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                           start=pd.Timestamp(start, tz="America/New_York"),
                           end=pd.Timestamp(end + " 23:59", tz="America/New_York"),
                           adjustment="all", feed=feed)
    bars = cli.get_stock_bars(req).df.reset_index()
    bars["et"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
    t = bars["et"].dt.time
    bars = bars[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    bars["date"] = bars["et"].dt.date
    bars["time"] = bars["et"].dt.strftime("%H:%M")
    return bars[["date", "time", "open", "high", "low", "close", "volume"]].sort_values(["date", "time"])


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = sys.argv[2] if len(sys.argv) > 2 else "2016-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-12-31"
    preflight()
    os.makedirs(OUT, exist_ok=True)
    df = fetch_30min(symbol, start, end)
    path = os.path.join(OUT, f"{symbol}_30min_{start[:4]}_{end[:4]}.csv")
    df.to_csv(path, index=False)
    print(f"{len(df)} bars, {df['date'].nunique()} trading days -> {path}")


if __name__ == "__main__":
    main()
