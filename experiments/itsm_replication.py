"""
Replication of Li, Sakkas & Urquhart (2022), Journal of Financial Markets, for one
US instrument (SPY by default) with Alpaca 30-minute bars.

For each trading day t:
    r_first = price(10:00, t) / close(t-1) - 1     # includes the overnight move
    r_last  = close(t)        / price(close-30min, t) - 1

1. In-sample predictive regression  r_last = a + b * r_first + e   (Newey-West t-stat)
   also split by calendar year.
2. Out-of-sample: expanding window, first 5 years as the initial estimation period,
   compared with the historical mean of r_last. Reports Campbell-Thompson R2_OOS and
   the Clark-West (2007) MSPE-adjusted t-statistic (Newey-West).
3. Market-timing rule (paper's Eq. 6): long the last half hour if r_first > 0,
   short otherwise; flat overnight. Annualized mean, SD and Sharpe, compared with
   buy-and-hold, gross and after an assumed round-trip cost.

Usage:
    python src/fetch_intraday.py SPY 2016-01-01 2025-12-31
    python experiments/itsm_replication.py data/intraday/SPY_30min_2016_2025.csv
Writes results/itsm_<name>.md and results/itsm_<name>_daily.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_BPS_ROUND_TRIP = 2.0   # assumed cost per daily round trip, in basis points


def daily_returns(bars):
    bars = bars.sort_values(["date", "time"]).copy()
    g = bars.groupby("date", sort=True)
    first = g.first()                       # 09:30 bar: its close is the 10:00 price
    last = g.last()                         # final regular-session bar (15:30, or 12:30 on half days)
    day = pd.DataFrame({
        "first_bar_time": first["time"],
        "p_1000": first["close"],
        "last_bar_time": last["time"],
        "p_last30": last["open"],
        "close": last["close"],
        "n_bars": g.size(),
    })
    day["prev_close"] = day["close"].shift(1)
    day = day[day["first_bar_time"] == "09:30"].dropna()
    day = day[day["n_bars"] >= 2]
    day["r_first"] = day["p_1000"] / day["prev_close"] - 1
    day["r_last"] = day["close"] / day["p_last30"] - 1
    day["r_bh"] = day["close"] / day["prev_close"] - 1
    day.index = pd.to_datetime(day.index)
    return day


def nw_tstat(x, y=None, lags=None):
    """OLS of y on [1, x] (or mean of x if y is None) with Newey-West SE. Returns (coef, t)."""
    if y is None:
        X = np.ones((len(x), 1)); yv = np.asarray(x, float)
    else:
        X = np.column_stack([np.ones(len(x)), np.asarray(x, float)]); yv = np.asarray(y, float)
    n, k = X.shape
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ yv
    u = yv - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta[-1], beta[-1] / se[-1], u


def in_sample(day):
    b, t, u = nw_tstat(day["r_first"].values, day["r_last"].values)
    r2 = 1 - np.var(u) / np.var(day["r_last"])
    return b, t, r2


def out_of_sample(day, init_years=5):
    start = day.index.min() + pd.DateOffset(years=init_years)
    x, y = day["r_first"].values, day["r_last"].values
    idx = np.where(day.index >= start)[0]
    f_model, f_mean = [], []
    for i in idx:                                   # estimate on data up to t-1 only
        xs, ys = x[:i], y[:i]
        b = np.cov(xs, ys, bias=True)[0, 1] / np.var(xs)
        a = ys.mean() - b * xs.mean()
        f_model.append(a + b * x[i]); f_mean.append(ys.mean())
    yo, fm, fh = y[idx], np.array(f_model), np.array(f_mean)
    r2oos = 1 - np.sum((yo - fm) ** 2) / np.sum((yo - fh) ** 2)
    f = (yo - fh) ** 2 - ((yo - fm) ** 2 - (fh - fm) ** 2)
    _, cw_t, _ = nw_tstat(f)
    return r2oos, cw_t, day.index[idx[0]].date(), len(idx)


def ann(r):
    m, s = r.mean() * 252, r.std() * np.sqrt(252)
    return m, s, m / s if s > 0 else np.nan


def run(path):
    bars = pd.read_csv(path)
    day = daily_returns(bars)
    name = os.path.basename(path).replace(".csv", "")
    sign = np.where(day["r_first"] > 0, 1.0, -1.0)
    day["r_itsm"] = sign * day["r_last"]
    day["r_itsm_net"] = day["r_itsm"] - COST_BPS_ROUND_TRIP / 1e4

    b, t, r2 = in_sample(day)
    r2oos, cw, oos_start, n_oos = out_of_sample(day)
    _, t_itsm, _ = nw_tstat(day["r_itsm"].values)
    rows = [("ITSM (gross)", *ann(day["r_itsm"])),
            (f"ITSM (net of {COST_BPS_ROUND_TRIP:g} bp/day)", *ann(day["r_itsm_net"])),
            ("Buy-and-hold", *ann(day["r_bh"]))]

    by_year = []
    for yr, d in day.groupby(day.index.year):
        if len(d) > 50:
            by, ty, _ = nw_tstat(d["r_first"].values, d["r_last"].values)
            by_year.append((yr, len(d), by * 100, ty, ann(d["r_itsm"])[2]))

    lines = [f"# ITSM replication: {name}", "",
             f"Trading days: {len(day)} ({day.index.min().date()} to {day.index.max().date()})", "",
             "## In-sample predictive regression (r_last on r_first)", "",
             f"- slope x100 = {b*100:.2f}, Newey-West t = {t:.2f}, R2 = {r2*100:.2f}%", "",
             "## Out-of-sample (expanding window, first 5 years initial)", "",
             f"- OOS period starts {oos_start}, {n_oos} days",
             f"- R2_OOS vs historical mean = {r2oos*100:.2f}%",
             f"- Clark-West MSPE-adjusted t = {cw:.2f} (one-sided; 1.645 = 5%)", "",
             "## Market-timing strategy (annualized)", "",
             "| Strategy | Mean % | SD % | Sharpe |", "|---|---|---|---|"]
    lines += [f"| {n} | {m*100:.2f} | {s*100:.2f} | {sr:.2f} |" for n, m, s, sr in rows]
    lines += ["", f"ITSM mean daily return t-stat (Newey-West) = {t_itsm:.2f}", "",
              "## By year", "", "| Year | Days | Slope x100 | t | ITSM Sharpe |", "|---|---|---|---|---|"]
    lines += [f"| {y} | {n} | {s:.2f} | {tt:.2f} | {sr:.2f} |" for y, n, s, tt, sr in by_year]
    out = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    open(os.path.join(ROOT, "results", f"itsm_{name}.md"), "w").write(out)
    day.to_csv(os.path.join(ROOT, "results", f"itsm_{name}_daily.csv"))
    print(out)
    return day


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "intraday", "SPY_30min_2016_2025.csv"))
