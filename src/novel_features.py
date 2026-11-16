"""
Feature blocks added for the novel approach. Each takes the frame returned by
src.khan_features.build() and returns a new frame, so experiments can switch them on or off.

    scale_free(feat, daily, price)     price-level features rewritten as ratios
    opening_return(feat, daily, first15, price)
                                       adds open_ret = first-15-minute close / previous close - 1
                                       (the opening-interval signal of Li, Sakkas and Urquhart, 2022)
    add_next_return(feat, daily, price) adds ret_next = close[t+1] / close[t] - 1, used by the
                                       return-forecasting model and the trading simulation
"""
import pandas as pd


def _by_date(df, col):
    return df.set_index("date")[col]


def scale_free(feat, daily, price="adj_"):
    """SMA, momentum and the three 'Previous' differences as fractions of price; volume as a
    ratio to its own 20-day mean. Fixes models trained on dollar levels that later see prices
    several times higher than anything in training."""
    g = feat.copy()
    c = _by_date(daily, price + "close").reindex(g["date"]).values
    vol = _by_date(daily, price + "volume")
    g["sma"] = c / g["sma"] - 1
    g["momentum"] = g["momentum"] / (c - g["momentum"])
    for col in ("prev_open_close", "prev_close_high", "prev_close_low"):
        g[col] = g[col] / c
    g["volume"] = (vol / vol.rolling(20).mean()).reindex(g["date"]).values
    return g.dropna().reset_index(drop=True)


def opening_return(feat, daily, first15, price="adj_"):
    """Return from the previous day's close to the close of today's first 15-minute bar.
    Known by the close of day t, so it is a valid input for predicting day t+1."""
    g = feat.copy()
    prev_close = _by_date(daily, price + "close").shift(1)
    f15 = _by_date(first15, price + "close")
    g["open_ret"] = (f15 / prev_close - 1).reindex(g["date"]).values
    return g.dropna().reset_index(drop=True)


def add_next_return(feat, daily, price="adj_"):
    g = feat.copy()
    c = _by_date(daily, price + "close")
    g["ret_next"] = (c.shift(-1) / c - 1).reindex(g["date"]).values
    return g
