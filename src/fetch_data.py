"""
Data acquisition for the DS-340W parent paper.

Khan, A.H. et al. (2023) "A performance comparison of machine learning models for
stock market prediction with novel investment strategy." PLOS ONE 18(9): e0286362.
https://doi.org/10.1371/journal.pone.0286362

SOURCE (paper, Table 5 / Table 6 / Fig 2): Alpaca Broker historical market data,
Tesla Inc. (TSLA), 1 Jan 2016 - 31 Dec 2021.

Arm 1 "1-day time frame"  -> daily bars
Arm 2 "15-min strategy"   -> the FIRST 15-minute bar after the open, each trading day

Setup: put your keys in a .env file in this folder, then run
    python src/fetch_data.py
"""
import os
import sys
import pandas as pd

START = "2016-01-01"
END = "2021-12-31"
SYMBOL = "TSLA"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Feeds to probe, best first. SIP is the full consolidated tape; IEX is a single
# venue with much shorter history on the free plan.
FEEDS = ["sip", "iex"]


def load_dotenv():
    """Read key=value pairs from a .env file in the project root, if present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def preflight():
    try:
        import alpaca  # noqa: F401
    except ImportError:
        sys.exit("Missing dependency. Run:\n\n    pip install alpaca-py pandas\n")
    load_dotenv()
    if not os.environ.get("APCA_API_KEY_ID") or not os.environ.get("APCA_API_SECRET_KEY"):
        sys.exit(
            "No Alpaca keys found.\n\n"
            "  Create a file called .env in this folder containing:\n\n"
            "       APCA_API_KEY_ID=your_key_here\n"
            "       APCA_API_SECRET_KEY=your_secret_here\n"
        )
    print("Setup OK: alpaca-py installed, keys found.\n")


def client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"],
                                     os.environ["APCA_API_SECRET_KEY"])


def _request(cli, timeframe, start, end, feed):
    from alpaca.data.requests import StockBarsRequest
    kw = dict(symbol_or_symbols=SYMBOL, timeframe=timeframe,
              start=start, end=end, adjustment="raw")
    if feed:
        kw["feed"] = feed
    return cli.get_stock_bars(StockBarsRequest(**kw)).df


def probe_feed(cli):
    """Find which data feed actually reaches the paper's 2016 start date."""
    from alpaca.data.timeframe import TimeFrame
    print("Probing which feed reaches 2016 (the paper's start date)...")
    for feed in FEEDS:
        try:
            df = _request(cli, TimeFrame.Day,
                          pd.Timestamp("2016-01-04", tz="America/New_York"),
                          pd.Timestamp("2016-03-01", tz="America/New_York"), feed)
            n = len(df)
            if n:
                d = df.reset_index()["timestamp"]
                print(f"  feed={feed:<4} {n:>4} rows  {d.min().date()} .. {d.max().date()}   USABLE")
                return feed
            print(f"  feed={feed:<4}    0 rows  (no 2016 history on this feed)")
        except Exception as e:
            print(f"  feed={feed:<4} error: {type(e).__name__}: {str(e)[:90]}")
    print("\n  No feed returned 2016 data. Falling back to whatever IEX provides.")
    return "iex"


def fetch(cli, tf, label, feed):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    tfmap = {"day": TimeFrame.Day, "15min": TimeFrame(15, TimeFrameUnit.Minute)}
    print(f"Requesting {label} bars for {SYMBOL} {START}..{END} (feed={feed}) ...")
    bars = _request(cli, tfmap[tf],
                    pd.Timestamp(START, tz="America/New_York"),
                    pd.Timestamp(END + " 23:59", tz="America/New_York"), feed)
    if bars.empty:
        sys.exit(f"No {label} data returned.")
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["et"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars["date"] = bars["et"].dt.date
    return bars


def main():
    preflight()
    os.makedirs(OUT, exist_ok=True)
    cli = client()
    feed = probe_feed(cli)
    print()

    daily = fetch(cli, "day", "daily", feed)
    d_out = daily[["date", "open", "high", "low", "close", "volume"]].copy()
    p1 = os.path.join(OUT, "TSLA_daily_2016_2021.csv")
    d_out.to_csv(p1, index=False)
    print(f"  -> {len(d_out):>6} rows  {p1}")

    m15 = fetch(cli, "15min", "15-minute", feed)
    m15 = m15[m15["et"].dt.time >= pd.Timestamp("09:30").time()]
    first = m15.sort_values("et").groupby("date", as_index=False).first()
    f_out = first[["date", "open", "high", "low", "close", "volume"]].copy()
    p2 = os.path.join(OUT, "TSLA_first15min_2016_2021.csv")
    f_out.to_csv(p2, index=False)
    print(f"  -> {len(f_out):>6} rows  {p2}")

    print("\n--- Verification vs the paper ---")
    for name, df in [("daily", d_out), ("first-15min", f_out)]:
        d = pd.to_datetime(df["date"])
        test = df[(d >= "2020-11-16") & (d <= "2021-12-31")]
        train = df[d <= "2020-11-15"]
        print(f"{name:>12}: {len(df):>5} total | {len(train):>5} train | {len(test):>4} test "
              f"| starts {d.min().date()}")
    print("\nPaper Table 5 expects roughly: 1510 total | 1227 train | 283 test | starts 2016-01-04")


if __name__ == "__main__":
    main()
