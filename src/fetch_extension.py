"""
Download TSLA for 2016-2025 so the final, untouched test set (2022-2025) can be used.

Same source and settings as src/fetch_data.py (Alpaca, raw prices, daily bars plus the first
15-minute bar of each trading day). The full 2016-2025 range is pulled in one go so indicator
warm-up periods run continuously across the 2021/2022 boundary.

Split-adjusted columns (adj_*) are added for both Tesla splits:
    2020-08-31  5-for-1
    2022-08-25  3-for-1
Older prices are divided by the split ratio (volume multiplied), so the latest prices are unchanged.

Run on your own computer (needs the Alpaca keys in .env):
    python src/fetch_extension.py
Writes:
    data/TSLA_daily_2016_2025.csv
    data/TSLA_first15min_2016_2025.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_data as F  # noqa: E402

F.START, F.END = "2016-01-01", "2025-12-31"
SPLITS = [("2020-08-31", 5.0), ("2022-08-25", 3.0)]
PRICES = ["open", "high", "low", "close"]
# published split-adjusted calendar-year returns, used only as a printed cross-check
PUBLISHED = {2022: -65.0, 2023: 101.7, 2024: 62.5}


def adjust(df):
    df = df.copy()
    for c in PRICES:
        df["adj_" + c] = df[c].astype(float)
    df["adj_volume"] = df["volume"].astype(float)
    for eff, ratio in SPLITS:
        before = df["date"] < pd.Timestamp(eff)
        for c in PRICES:
            df.loc[before, "adj_" + c] /= ratio
        df.loc[before, "adj_volume"] *= ratio
    return df


def check(name, df):
    old = os.path.join(F.OUT, f"TSLA_{name}_2016_2021.csv")
    if os.path.exists(old):
        o = pd.read_csv(old, parse_dates=["date"]).set_index("date")["close"]
        n = df.set_index("date")["close"].reindex(o.index)
        print(f"   overlap with existing 2016-2021 file: {n.notna().sum()} of {len(o)} days, "
              f"max raw close difference {(n - o).abs().max():.4f}")
    r = df["adj_close"].pct_change()
    for eff, _ in SPLITS:
        i = df.index[df["date"] == pd.Timestamp(eff)]
        if len(i):
            print(f"   adjusted return on split date {eff}: {r.loc[i[0]] * 100:+.2f}%")
    if name == "daily":
        ye = df.groupby(df["date"].dt.year)["adj_close"].last()
        for y, p in PUBLISHED.items():
            if y in ye.index and y - 1 in ye.index:
                print(f"   {y} return {(ye[y] / ye[y - 1] - 1) * 100:+7.1f}%   (published about {p:+.1f}%)")
    yrs = df["date"].dt.year.value_counts().sort_index()
    print("   trading days per year: " + ", ".join(f"{y}: {c}" for y, c in yrs.items()))


def main():
    F.preflight()
    cli = F.client()
    feed = F.probe_feed(cli)
    print()
    daily = F.fetch(cli, "day", "daily", feed)
    d = daily[["date"] + PRICES + ["volume"]].copy()
    m15 = F.fetch(cli, "15min", "15-minute", feed)
    m15 = m15[m15["et"].dt.time >= pd.Timestamp("09:30").time()]
    first = m15.sort_values("et").groupby("date", as_index=False).first()
    f = first[["date"] + PRICES + ["volume"]].copy()
    for name, df in (("daily", d), ("first15min", f)):
        df["date"] = pd.to_datetime(df["date"])
        df = adjust(df.sort_values("date").reset_index(drop=True))
        path = os.path.join(F.OUT, f"TSLA_{name}_2016_2025.csv")
        df.to_csv(path, index=False)
        print(f"\n{name}: {len(df)} rows -> {path}")
        check(name, df)


if __name__ == "__main__":
    main()
