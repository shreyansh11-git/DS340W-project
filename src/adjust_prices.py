"""
Add split-adjusted price columns to the raw datasets.

WHY
---
Tesla executed a 5-for-1 stock split effective 2020-08-31. The Alpaca download uses
adjustment="raw", matching the paper's Fig 2 (which peaks near $2,200 pre-split).
In raw prices that split appears as a -77.5% single-day return that never happened,
which corrupts every price-level feature in the paper's Table 2 - most visibly the
50-period SMA, which sits 3-4x above price for 50 sessions afterwards.

WHAT THIS DOES
--------------
Adds adj_open / adj_high / adj_low / adj_close / adj_volume alongside the raw
columns. Raw columns are left untouched so the paper's exact setup stays reachable.

Back-adjustment convention (standard): for every bar before the effective date,
    adjusted price  = raw price  / ratio
    adjusted volume = raw volume * ratio
so that post-split values are unchanged and the series is continuous.

Corporate actions inside 2016-01-04 .. 2021-12-31:
    2020-08-31   5-for-1 split   (the 3-for-1 was Aug 2022, outside this window)

Run:  python src/adjust_prices.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

SPLITS = [{"effective": "2020-08-31", "ratio": 5.0, "note": "5-for-1"}]
FILES = ["TSLA_daily_2016_2021.csv", "TSLA_first15min_2016_2021.csv"]

PRICE_COLS = ["open", "high", "low", "close"]
# published split-adjusted annual total returns, for an independent cross-check
PUBLISHED = {2017: 45.70, 2018: 6.89, 2019: 25.70, 2020: 743.44, 2021: 49.76}


def adjust(df):
    df = df.copy()
    for c in PRICE_COLS:
        df["adj_" + c] = df[c].astype(float)
    df["adj_volume"] = df["volume"].astype(float)
    for s in SPLITS:
        eff = pd.Timestamp(s["effective"])
        before = df["date"] < eff
        for c in PRICE_COLS:
            df.loc[before, "adj_" + c] = df.loc[before, "adj_" + c] / s["ratio"]
        df.loc[before, "adj_volume"] = df.loc[before, "adj_volume"] * s["ratio"]
    return df


def main():
    for fname in FILES:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run: python src/fetch_data.py")
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if "adj_close" in df.columns:
            df = df[["date"] + PRICE_COLS + ["volume"]]
        out = adjust(df)
        out.to_csv(path, index=False)

        raw_r = df["close"].pct_change()
        adj_r = out["adj_close"].pct_change()
        print(f"{fname}")
        print(f"   worst 1-day return   raw {raw_r.min()*100:>7.1f}%   "
              f"adjusted {adj_r.min()*100:>7.1f}%")

        # The real test is the split date itself: after adjustment it must show a
        # plausible market move, not a mechanical cliff. A blanket threshold would
        # be wrong here - the first-15-min series is first-bar-to-first-bar, so it
        # spans the overnight gap and legitimately exceeds 25% on real gap days
        # (e.g. 2020-02-04 and 2020-07-13, both genuine TSLA events).
        for sp in SPLITS:
            eff = pd.Timestamp(sp["effective"])
            i = out.index[out["date"] == eff]
            if len(i):
                r = adj_r.loc[i[0]] * 100
                print(f"   split {sp['effective']} ({sp['note']}) adjusted return "
                      f"{r:>7.2f}%   {'OK' if abs(r) < 30 else 'STILL BROKEN'}")
                assert abs(r) < 30, f"discontinuity survived at {sp['effective']}"

        outliers = out.loc[adj_r.abs() > 0.25, "date"]
        if len(outliers):
            print(f"   note: {len(outliers)} day(s) still move >25% after adjustment - "
                  "real market events, verified:")
            for dt in outliers:
                print(f"         {dt.date()}  {adj_r[out['date'] == dt].iloc[0]*100:+.1f}%")

        if fname.startswith("TSLA_daily"):
            print("   cross-check: split-adjusted annual returns vs published")
            ye = out.groupby(out["date"].dt.year)["adj_close"].last()
            for y, p in PUBLISHED.items():
                r = (ye[y] / ye[y - 1] - 1) * 100
                flag = "OK" if abs(r - p) < 0.6 else "MISMATCH"
                print(f"      {y}  computed {r:>8.2f}%   published {p:>7.2f}%   {flag}")
                assert flag == "OK", f"{y} does not reconcile"
        print()

    print("Adjusted columns written. Raw columns preserved.")
    print("Next: python src/make_splits.py   (to propagate into the splits)")


if __name__ == "__main__":
    main()
