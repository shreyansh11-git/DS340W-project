"""
Split the TSLA datasets into Train / Test / Validation.

DESIGN DECISIONS
----------------
1. CHRONOLOGICAL, never random. This is a time series: a shuffled split would put
   future prices in the training set and leak information backwards, inflating
   every metric. Rows are cut by position on the trading-day calendar.

2. 70 / 20 / 10, with VALIDATION LAST. The held-out set is the most recent slice,
   which is the only arrangement that mimics real deployment: train on the past,
   judge on the future.

3. IDENTICAL date boundaries for both arms (daily and first-15-min), so the two
   datasets stay row-aligned and directly comparable.

4. NO OVERLAP, NO GAPS. Every row lands in exactly one split; the three
   reassemble into the original file.

Run:  python src/make_splits.py
"""
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(DATA, "splits")

TRAIN_FRAC, TEST_FRAC = 0.70, 0.20   # validation takes the remainder (~10%)

ARMS = {
    "daily":   "TSLA_daily_2016_2021.csv",
    "first15": "TSLA_first15min_2016_2021.csv",
}

# Paper's own boundary, recorded so the exact reproduction split stays reachable.
PAPER_TRAIN_END = "2020-11-15"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    frames = {}
    for arm, fname in ARMS.items():
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Run: python src/fetch_data.py")
        frames[arm] = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # both arms must share the same calendar, or the boundaries would diverge
    cal = frames["daily"]["date"]
    for arm, df in frames.items():
        if not df["date"].equals(cal):
            sys.exit(f"Calendar mismatch in {arm}: the two arms must share identical dates.")

    n = len(cal)
    n_train = int(n * TRAIN_FRAC)
    n_test = int(n * TEST_FRAC)
    bounds = {
        "train":      (0, n_train),
        "test":       (n_train, n_train + n_test),
        "validation": (n_train + n_test, n),
    }

    dirs = {
        "train": os.path.join(SPLITS, "train"),
        "test": os.path.join(SPLITS, "test"),
        "validation": os.path.join(SPLITS, "HOLDOUT_validation"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    manifest = {
        "source_rows": n,
        "method": "chronological, no shuffling; validation is the most recent slice",
        "fractions": {"train": TRAIN_FRAC, "test": TEST_FRAC,
                      "validation": round(1 - TRAIN_FRAC - TEST_FRAC, 4)},
        "paper_train_end": PAPER_TRAIN_END,
        "splits": {},
    }

    print(f"Source: {n} trading days, {cal.min().date()} .. {cal.max().date()}\n")
    print(f"{'split':<12}{'rows':>6}{'pct':>8}  {'start':<12}{'end':<12}")
    print("-" * 52)

    for split, (a, b) in bounds.items():
        info = {"rows": b - a, "pct": round((b - a) / n * 100, 1),
                "start": str(cal.iloc[a].date()), "end": str(cal.iloc[b - 1].date()),
                "files": {}}
        for arm in ARMS:
            sub = frames[arm].iloc[a:b]
            out = os.path.join(dirs[split], f"TSLA_{arm}_{split}.csv")
            sub.to_csv(out, index=False)
            info["files"][arm] = {"path": os.path.relpath(out, ROOT),
                                  "rows": len(sub), "sha256_16": sha256(out)}
        manifest["splits"][split] = info
        print(f"{split:<12}{info['rows']:>6}{info['pct']:>7.1f}%  "
              f"{info['start']:<12}{info['end']:<12}")

    # ---- integrity assertions
    tot = sum(v["rows"] for v in manifest["splits"].values())
    assert tot == n, f"rows lost: {tot} != {n}"
    ends = [manifest["splits"][s]["end"] for s in ("train", "test")]
    starts = [manifest["splits"][s]["start"] for s in ("test", "validation")]
    for e, s in zip(ends, starts):
        assert e < s, f"overlap between splits at {e}/{s}"
    print("\nIntegrity: rows sum to source, no overlap, no gaps, chronological order OK")

    with open(os.path.join(SPLITS, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    guard = os.path.join(dirs["validation"], "DO_NOT_OPEN.md")
    with open(guard, "w") as fh:
        fh.write(
            "# HELD-OUT VALIDATION SET - DO NOT TOUCH\n\n"
            f"{manifest['splits']['validation']['rows']} trading days, "
            f"{manifest['splits']['validation']['start']} to "
            f"{manifest['splits']['validation']['end']}.\n\n"
            "This is the unseen set. It must not be read, plotted, summarised, or used for\n"
            "model selection, feature choice, or hyperparameter tuning.\n\n"
            "Open it exactly once, at the very end, when the pipeline is bug-free and the\n"
            "Train/Test results are final. Every look at this data costs you the ability to\n"
            "claim the final number is honest.\n\n"
            "`src/data.py` refuses to load it unless you pass `allow_holdout=True`.\n"
        )

    print(f"\nWrote splits to {os.path.relpath(SPLITS, ROOT)}/")
    print(f"Manifest: {os.path.relpath(os.path.join(SPLITS,'manifest.json'), ROOT)}")
    print(f"Guard:    {os.path.relpath(guard, ROOT)}")


if __name__ == "__main__":
    main()
