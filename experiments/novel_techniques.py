"""
Novel approach: change one block of the Khan et al. (2023) pipeline at a time and compare.

Each row of the comparison is a `Setup`. Setups build on each other, so the effect of each
technique can be read off as the change from the row above:

  S0  Paper as published      authors' pipeline, next-day "Previous" features (lookahead)
  S1  Paper as written        same, but previous-day features as the paper's Table 2 states
  S2  + leakage fixes         split-adjusted prices, scaling fitted on training data only,
                              time-ordered cross-validation
  S3  + scale-free features   price-level features as ratios (src/novel_features.py)
  S4  + opening return        Li et al. (2022) opening-interval return as a 10th feature
  S5  + rolling retraining    Kelly et al. (2024): refit every 21 days on the last 504 days
  S6  Ridge on random features  Kelly et al. (2024) model as a 10th model (src/rff_ridge.py)
  S5b Rolling retraining only  S5's retraining on S2's features, to isolate its effect
  S7  Opening-interval rule   Li et al. (2022) timing rule applied to TSLA, no machine learning
                              (S7a before costs, S7b after costs). Its accuracy is for the
                              rest-of-day direction, a different target from the other rows.

Evaluation for every row: test accuracy against an always-up baseline, and a USD 10,000 trading
simulation with a 0.1% commission against buy and hold (src/backtest.py). Test period is the
paper's: 2020-11-16 to 2021-12-31.

Usage:  python experiments/novel_techniques.py
Writes: results/novel_techniques.csv, results/novel_techniques.md
"""
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.backtest import COST, metrics, simulate  # noqa: E402
from src.khan_features import FEATURES, build  # noqa: E402
from src.khan_models import MODELS, fit_predict  # noqa: E402
from src.novel_features import add_next_return, opening_return, scale_free  # noqa: E402
from src.rff_ridge import RFFRidge  # noqa: E402

SPLIT = pd.Timestamp("2020-11-16")
CLASSIFIERS = tuple(MODELS)


@dataclass(frozen=True)
class Setup:
    key: str
    label: str
    prev_mode: str = "lagged"      # lagged | lookahead
    price: str = "adj_"            # "" raw | "adj_" split-adjusted
    scale: str = "train_only"      # all_data | train_only
    cv: str = "timeseries"         # kfold | timeseries
    scale_free: bool = False
    opening_return: bool = False
    window: int = 0                # 0 = train once; otherwise rolling window length in days
    refit: int = 21                # days between refits when rolling
    models: tuple = CLASSIFIERS


S0 = Setup("S0", "Paper as published (lookahead)", prev_mode="lookahead", price="", scale="all_data", cv="kfold")
S1 = replace(S0, key="S1", label="Paper as written (honest)", prev_mode="lagged")
S2 = Setup("S2", "+ leakage fixes")
S3 = replace(S2, key="S3", label="+ scale-free features", scale_free=True)
S4 = replace(S3, key="S4", label="+ opening return (Li)", opening_return=True)
S5 = replace(S4, key="S5", label="+ rolling retraining (Kelly)", window=504)
S6 = replace(S4, key="S6", label="Ridge on random features (Kelly)", window=252, models=("RFF Ridge",))
S5b = replace(S2, key="S5b", label="Rolling retraining only (on S2)", window=504)
SETUPS = [S0, S1, S2, S3, S4, S5, S5b, S6]


def load(name):
    return pd.read_csv(os.path.join(ROOT, "data", name), parse_dates=["date"])


DAILY = load("TSLA_daily_2016_2021.csv")
FIRST15 = load("TSLA_first15min_2016_2021.csv")


def features(s):
    feat = build(DAILY, prev_mode=s.prev_mode, price=s.price)
    cols = list(FEATURES)
    if s.scale_free:
        feat = scale_free(feat, DAILY, s.price)
    if s.opening_return:
        feat = opening_return(feat, DAILY, FIRST15, s.price)
        cols.append("open_ret")
    return add_next_return(feat, DAILY, "adj_").dropna().reset_index(drop=True), cols


def minmax(tr, te, ref):
    lo, hi = ref.min(), ref.max()
    span = (hi - lo).replace(0, 1)
    return (tr - lo) / span, (te - lo) / span


def predict_block(s, name, train, test, cols, all_rows):
    ref = all_rows[cols] if s.scale == "all_data" else train[cols]
    Xtr, Xte = minmax(train[cols], test[cols], ref)
    if name == "RFF Ridge":
        f = RFFRidge().fit(Xtr.values, train["ret_next"].values).predict(Xte.values)
        return (f > 0).astype(int)
    return fit_predict(name, Xtr, train["target"].astype(int), Xte, cv=s.cv)


def predict(s, name, feat, cols):
    train, test = feat[feat["date"] < SPLIT], feat[feat["date"] >= SPLIT]
    if not s.window:
        return test, predict_block(s, name, train, test, cols, feat)
    preds = []
    for start in range(0, len(test), s.refit):
        block = test.iloc[start:start + s.refit]
        past = feat[feat["date"] < block["date"].iloc[0]].iloc[-s.window:]
        preds.append(predict_block(s, name, past, block, cols, feat))
    return test, np.concatenate(preds)


def evaluate(s):
    feat, cols = features(s)
    rows = []
    for name in s.models:
        test, pred = predict(s, name, feat, cols)
        net, n_trades = simulate(pred, test["ret_next"].values)
        rows.append(dict(setup=s.key, technique=s.label, model=name,
                         accuracy=100 * accuracy_score(test["target"].astype(int), pred),
                         up_share=100 * np.mean(pred), trades=int(n_trades),
                         n_test=len(test), **metrics(net)))
        print(f"{s.key} {name:20s} acc {rows[-1]['accuracy']:6.2f}  sharpe {rows[-1]['sharpe']:6.2f}", flush=True)
    return rows, test


def opening_rule():
    """S7: Li et al. (2022) timing rule on TSLA. If the return from yesterday's close to the end of
    the first 15 minutes is positive, hold TSLA from then until today's close; otherwise stay out.
    Each trading day is a round trip, so the commission is charged twice per active day."""
    d = DAILY.set_index("date")["adj_close"]
    f = FIRST15.set_index("date")["adj_close"]
    df = pd.DataFrame({"open_ret": f / d.shift(1) - 1, "rest": d / f - 1}).dropna()
    df = df[(df.index >= SPLIT)]
    pos = (df["open_ret"] > 0).astype(float).values
    gross = pos * df["rest"].values
    net = gross - 2 * COST * pos
    hit = accuracy_score((df["rest"] > 0).astype(int), (df["open_ret"] > 0).astype(int))
    out = []
    for key, kind, r in (("S7a", "before costs", gross), ("S7b", "after costs", net)):
        out.append(dict(setup=key, technique=f"Opening-interval rule (Li), {kind}", model="Rule",
                        accuracy=100 * hit, up_share=100 * pos.mean(), trades=int(2 * pos.sum()),
                        n_test=len(df), **metrics(r)))
    return out


def summary(res, test):
    bh = metrics(simulate(np.ones(len(test)), test["ret_next"].values)[0])
    always_up = 100 * test["target"].mean()
    rows = []
    for key, g in res.groupby("setup", sort=False):
        best = g.loc[g["accuracy"].idxmax()]
        rows.append({
            "Setup": key, "Technique": g["technique"].iloc[0], "Models": len(g),
            "Avg accuracy %": g["accuracy"].mean(), "Best accuracy %": best["accuracy"],
            "Days long %": g["up_share"].mean(),
            "Best model": best["model"], "Beat always-up": f"{(g['accuracy'] > always_up).sum()} of {len(g)}",
            "Avg Sharpe (net)": g["sharpe"].mean(), "Best Sharpe (net)": g["sharpe"].max(),
            "Beat buy and hold": f"{(g['sharpe'] > bh['sharpe']).sum()} of {len(g)}",
        })
    rows.append({"Setup": "", "Technique": "Always-up baseline / buy and hold", "Models": "",
                 "Avg accuracy %": always_up, "Best accuracy %": np.nan, "Days long %": 100.0, "Best model": "",
                 "Beat always-up": "", "Avg Sharpe (net)": bh["sharpe"], "Best Sharpe (net)": np.nan,
                 "Beat buy and hold": ""})
    return pd.DataFrame(rows), bh, always_up


def main():
    all_rows, test = [], None
    for s in SETUPS:
        rows, test = evaluate(s)
        all_rows += rows
    all_rows += opening_rule()
    res = pd.DataFrame(all_rows)
    out = os.path.join(ROOT, "results", "novel_techniques")
    res.to_csv(out + ".csv", index=False)
    table, bh, always_up = summary(res, test)
    detail = res.pivot_table(index="model", columns="setup", values="accuracy", sort=False)
    with open(out + ".md", "w") as fh:
        fh.write("# Novel techniques: comparison table\n\n")
        fh.write(f"TSLA, test period 2020-11-16 to 2021-12-31 ({len(test)} trading days). "
                 f"Trading simulation: USD 10,000, {COST:.1%} commission per trade, long or cash.\n")
        fh.write(f"Buy and hold over the same period: Sharpe {bh['sharpe']:.2f}, "
                 f"return {bh['total_return_pct']:.1f}%, max drawdown {bh['max_drawdown_pct']:.1f}%.\n\n")
        fh.write(table.round(2).to_markdown(index=False) + "\n\n")
        fh.write("## Test accuracy by model (%)\n\n" + detail.round(2).to_markdown() + "\n\n")
        cols = ["setup", "model", "accuracy", "sharpe", "total_return_pct", "max_drawdown_pct", "trades"]
        fh.write("## Trading results by model\n\n" + res[cols].round(2).to_markdown(index=False) + "\n")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
