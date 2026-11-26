"""
Final evaluation with significance tests.

Runs every setup from experiments/novel_techniques.py (S0 to S7) on two periods:

  development   the paper's test period, 2020-11-16 to 2021-12-31 (used while building the
                techniques; repeated here only to add significance tests to Table II)
  final         2022-01-03 to 2025-12-31, never used before. Static setups are trained once on
                2016-2021; rolling setups keep refitting on past data only. Run this ONCE.

For every model it reports:
  accuracy with a 95% Wilson interval, and an exact McNemar test against always predicting up
  Sharpe ratio after costs with a 95% block-bootstrap interval, and the interval of its
  difference from buy and hold
For every setup it adds a Diebold-Mariano test of its average daily classification loss
against S2 (the leakage-fixed pipeline).

Usage:
  python src/fetch_extension.py          # once, downloads 2016-2025 data
  python experiments/final_test.py       # writes results/final_test.md and .csv
  python experiments/final_test.py --dry-run
        checks the code on 2021 only, using the old 2016-2021 files; prints, writes nothing
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import novel_techniques as NT  # noqa: E402
from src.backtest import COST, metrics, simulate  # noqa: E402
from src.stats_tests import diebold_mariano, mcnemar, sharpe_bootstrap, wilson_ci  # noqa: E402

PERIODS = {
    "development": ("TSLA_daily_2016_2021.csv", "TSLA_first15min_2016_2021.csv", "2020-11-16"),
    "final": ("TSLA_daily_2016_2025.csv", "TSLA_first15min_2016_2025.csv", "2022-01-01"),
}
DRY_RUN = ("TSLA_daily_2016_2021.csv", "TSLA_first15min_2016_2021.csv", "2021-01-01")
BASE = "S2"


def use(period):
    daily, first15, split = period
    NT.DAILY, NT.FIRST15, NT.SPLIT = NT.load(daily), NT.load(first15), pd.Timestamp(split)


def model_row(key, label, model, y, pred, net, bench):
    k, n = int(np.sum(y == pred)), len(y)
    lo, hi = wilson_ci(k, n)
    _, _, p = mcnemar(y, pred, np.ones_like(y))
    b = sharpe_bootstrap(net, bench)
    return dict(setup=key, technique=label, model=model, n=n, accuracy=100 * k / n,
                acc_lo=100 * lo, acc_hi=100 * hi, p_vs_always_up=p, days_long=100 * np.mean(pred),
                **metrics(net), **b)


def run_models():
    rows, losses, test = [], {}, None
    for s in NT.SETUPS:
        feat, cols = NT.features(s)
        per_model = []
        for name in s.models:
            test, pred = NT.predict(s, name, feat, cols)
            y = test["target"].astype(int).values
            net, _ = simulate(pred, test["ret_next"].values)
            bench, _ = simulate(np.ones(len(test)), test["ret_next"].values)
            rows.append(model_row(s.key, s.label, name, y, pred, net, bench))
            per_model.append(pd.Series((pred != y).astype(float), index=test["date"].values))
            print(f"  {s.key:4s} {name:20s} acc {rows[-1]['accuracy']:6.2f}  sharpe {rows[-1]['sharpe']:6.2f}", flush=True)
        losses[s.key] = pd.concat(per_model, axis=1).mean(axis=1)
    return rows, losses, test


def run_rule():
    d = NT.DAILY.set_index("date")["adj_close"]
    f = NT.FIRST15.set_index("date")["adj_close"]
    df = pd.DataFrame({"open_ret": f / d.shift(1) - 1, "rest": d / f - 1, "day": d / d.shift(1) - 1}).dropna()
    df = df[df.index >= NT.SPLIT]
    y = (df["rest"] > 0).astype(int).values
    pred = (df["open_ret"] > 0).astype(int).values
    bench, _ = simulate(np.ones(len(df)), df["day"].values)
    gross = pred * df["rest"].values
    return [model_row(key, f"Opening-interval rule (Li), {kind}", "Rule", y, pred, r, bench)
            for key, kind, r in (("S7a", "before costs", gross), ("S7b", "after costs", gross - 2 * COST * pred))]


def summarise(res, losses, test):
    bench = simulate(np.ones(len(test)), test["ret_next"].values)[0]
    bh = {**metrics(bench), **sharpe_bootstrap(bench)}
    always_up = 100 * test["target"].mean()
    out = []
    for key, g in res.groupby("setup", sort=False):
        best = g.loc[g["accuracy"].idxmax()]
        sig = g["p_vs_always_up"] < 0.05
        dm, p_dm = (np.nan, np.nan)
        if key in losses and key != BASE:
            a, b = losses[key].align(losses[BASE], join="inner")
            dm, p_dm = diebold_mariano(a.values, b.values)
        out.append({
            "Setup": key, "Technique": g["technique"].iloc[0], "Models": len(g),
            "Avg accuracy %": g["accuracy"].mean(), "Best accuracy %": best["accuracy"], "Best model": best["model"],
            "Sig. above always-up": f"{int((sig & (g['accuracy'] > always_up)).sum())} of {len(g)}",
            "Sig. below always-up": f"{int((sig & (g['accuracy'] < always_up)).sum())} of {len(g)}",
            "Days long %": g["days_long"].mean(), "Avg Sharpe (net)": g["sharpe"].mean(),
            "Sharpe sig. above B&H": f"{int((g['diff_lo'] > 0).sum())} of {len(g)}",
            "Sharpe sig. below B&H": f"{int((g['diff_hi'] < 0).sum())} of {len(g)}",
            f"DM vs {BASE} (p)": "" if np.isnan(p_dm) else f"{dm:+.2f} ({p_dm:.3f})",
        })
    return pd.DataFrame(out), bh, always_up


def evaluate(period):
    use(period)
    rows, losses, test = run_models()
    res = pd.DataFrame(rows + run_rule())
    table, bh, always_up = summarise(res, losses, test)
    return res, table, bh, always_up, test


def section(fh, title, res, table, bh, always_up, test):
    d0, d1 = test["date"].min().date(), test["date"].max().date()
    fh.write(f"## {title}: {d0} to {d1} ({len(test)} trading days)\n\n")
    fh.write(f"Always-up baseline accuracy: {always_up:.2f}%. Buy and hold: return {bh['total_return_pct']:.1f}%, "
             f"Sharpe {bh['sharpe']:.2f} (95% interval {bh['sharpe_lo']:.2f} to {bh['sharpe_hi']:.2f}), "
             f"max drawdown {bh['max_drawdown_pct']:.1f}%.\n\n")
    fh.write(table.round(2).to_markdown(index=False) + "\n\n")
    cols = ["setup", "model", "accuracy", "acc_lo", "acc_hi", "p_vs_always_up", "days_long",
            "sharpe", "sharpe_lo", "sharpe_hi", "diff_lo", "diff_hi", "total_return_pct", "max_drawdown_pct"]
    fh.write("### By model\n\n" + res[cols].round(3).to_markdown(index=False) + "\n\n")


def main():
    if "--dry-run" in sys.argv:
        print("Dry run on 2021 (old files). Nothing is written.")
        _, table, bh, always_up, _ = evaluate(DRY_RUN)
        print(table.round(2).to_string(index=False))
        print(f"always-up {always_up:.2f}%  buy-and-hold Sharpe {bh['sharpe']:.2f}")
        return
    missing = [f for f in PERIODS["final"][:2] if not os.path.exists(os.path.join(ROOT, "data", f))]
    if missing:
        sys.exit("Missing " + ", ".join(missing) + ". Run first:  python src/fetch_extension.py")
    out = os.path.join(ROOT, "results", "final_test")
    results = {}
    for name, period in PERIODS.items():
        print(f"\n=== {name} ===")
        results[name] = evaluate(period)
    pd.concat([r[0].assign(period=n) for n, r in results.items()]).to_csv(out + ".csv", index=False)
    with open(out + ".md", "w") as fh:
        fh.write("# Final evaluation with significance tests\n\n"
                 "Sig. = p < 0.05, exact McNemar test against always predicting up. Sharpe intervals: 95% circular "
                 "block bootstrap (20-day blocks, 2,000 resamples); 'sig. above/below B&H' means the interval of the "
                 f"Sharpe difference from buy and hold excludes zero. DM: Diebold-Mariano statistic on average daily "
                 f"classification loss against {BASE} (positive = more errors than {BASE}). Trading: USD 10,000, "
                 f"{COST:.1%} commission per trade, long or cash. S7 accuracy is for the rest-of-day direction.\n\n")
        section(fh, "Final test (unseen)", *results["final"])
        section(fh, "Development period (paper's test window)", *results["development"])
    print("\n" + results["final"][1].round(2).to_string(index=False))
    print(f"\nWrote {out}.md and {out}.csv")


if __name__ == "__main__":
    main()
