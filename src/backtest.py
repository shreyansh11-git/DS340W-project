"""
Trading simulation in the spirit of Khan et al. (2023): USD 10,000 starting capital and a
0.1% commission on every buy or sell. Positions are fractions of capital (1 = fully invested,
0 = in cash, -1 = short), held from one close to the next.
"""
import numpy as np

CAPITAL = 10_000
COST = 0.001
DAYS = 252


def simulate(position, ret, cost=COST):
    """position[t] is held over ret[t]. Commission is charged on each change of position."""
    position = np.asarray(position, float)
    ret = np.asarray(ret, float)
    trades = np.abs(np.diff(np.r_[0.0, position]))
    net = position * ret - cost * trades
    return net, trades.sum()


def metrics(net):
    net = np.asarray(net, float)
    equity = CAPITAL * np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[CAPITAL, equity])[1:]
    sd = net.std(ddof=1)
    return {
        "final_capital": equity[-1],
        "total_return_pct": 100 * (equity[-1] / CAPITAL - 1),
        "sharpe": np.nan if sd == 0 else np.sqrt(DAYS) * net.mean() / sd,
        "max_drawdown_pct": 100 * (equity / peak - 1).min(),
    }
