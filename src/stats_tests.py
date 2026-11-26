"""
Significance tests used in the final evaluation.

    wilson_ci(k, n)                  95% interval for an accuracy of k correct out of n
    mcnemar(y, pred_a, pred_b)       exact McNemar test: do two sets of predictions differ in accuracy?
    diebold_mariano(loss_a, loss_b)  Diebold and Mariano (1995) test on two daily loss series,
                                     with a Newey-West variance for autocorrelation
    sharpe_bootstrap(net_a, net_b)   circular block bootstrap of a strategy's Sharpe ratio and of its
                                     difference from a benchmark (e.g. buy and hold)
"""
import numpy as np
from scipy import stats

DAYS = 252


def wilson_ci(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(y, pred_a, pred_b):
    """Two-sided exact test. Returns (days only A got right, days only B got right, p-value)."""
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    only_a = int(np.sum((a == y) & (b != y)))
    only_b = int(np.sum((a != y) & (b == y)))
    n = only_a + only_b
    p = 1.0 if n == 0 else stats.binomtest(only_a, n, 0.5).pvalue
    return only_a, only_b, p


def diebold_mariano(loss_a, loss_b):
    """Positive statistic means A has the larger loss. Returns (statistic, two-sided p-value)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    var = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def _sharpe(x):
    sd = x.std(ddof=1, axis=-1)
    return np.where(sd > 0, np.sqrt(DAYS) * x.mean(axis=-1) / np.where(sd > 0, sd, 1), np.nan)


def sharpe_bootstrap(net, bench=None, block=20, reps=2000, seed=0):
    """Returns dict with the Sharpe ratio's 95% interval and, if a benchmark is given, the 95%
    interval of (strategy Sharpe - benchmark Sharpe), resampling the same days for both."""
    net = np.asarray(net, float)
    n = len(net)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    s = _sharpe(net[idx])
    out = {"sharpe_lo": np.nanpercentile(s, 2.5), "sharpe_hi": np.nanpercentile(s, 97.5)}
    if bench is not None:
        diff = s - _sharpe(np.asarray(bench, float)[idx])
        out.update(diff_lo=np.nanpercentile(diff, 2.5), diff_hi=np.nanpercentile(diff, 97.5))
    return out
