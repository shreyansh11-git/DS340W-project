"""
The nine input features of Khan et al. (2023), Table 2, built from daily OHLCV bars.

Indicators are implemented directly in pandas (no TA-Lib dependency):
    RSI(14), SMA(50), ADX(20), Volume, Correlation(24), Momentum(20),
    Previous (Open-Close), Previous (Close-High), Previous (Close-Low).

The three "Previous" features can be built three ways, selected by `prev_mode`:
    "lagged"    -> value of the PREVIOUS bar, .shift(+1). This is what the paper's
                   Table 2 specifies ("Previous").
    "same_day"  -> value of the CURRENT bar, no shift. Known at the close of day t,
                   so it is also free of lookahead for a next-day target.
    "lookahead" -> value of the NEXT bar, .shift(-1). This is what the authors'
                   published notebook does. It leaks day t+1 into the row for day t.

Target: 1 if close[t+1] > close[t] (move up), 0 if it moves down. Flat days are dropped.
"""
import numpy as np
import pandas as pd

FEATURES = ["rsi", "sma", "adx", "volume", "corr", "prev_open_close",
            "prev_close_high", "prev_close_low", "momentum"]


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def adx(high, low, close, n=20):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build(df, prev_mode="lagged", price="adj_"):
    """df: daily bars with columns adj_open, adj_high, adj_low, adj_close, adj_volume (or raw)."""
    o, h, l, c = (df[price + k] for k in ("open", "high", "low", "close"))
    v = df[price + "volume"] if price + "volume" in df else df["volume"]
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]
    out["rsi"] = rsi(c, 14)
    out["sma"] = c.rolling(50).mean()
    out["adx"] = adx(h, l, c, 20)
    out["volume"] = v
    out["corr"] = c.rolling(24).corr(out["sma"])
    out["momentum"] = c - c.shift(20)

    shift = {"lagged": 1, "same_day": 0, "lookahead": -1}[prev_mode]
    out["prev_open_close"] = (o - c).shift(shift)
    out["prev_close_high"] = (c - h).shift(shift)
    out["prev_close_low"] = (c - l).shift(shift)

    nxt = c.shift(-1)
    out["target"] = np.where(nxt > c, 1, np.where(nxt < c, 0, np.nan))
    out.loc[nxt.isna(), "target"] = np.nan
    return out.dropna().reset_index(drop=True)
