"""
Indicator library.

Design notes
------------
*  Everything is plain float64 numpy.  Input arrays are *dense* (no interior
   NaNs) because universe.py reindexes each coin onto its own continuous daily
   range before we ever get here.
*  Every function returns an array the same length as its input, with a proper
   NaN warm-up prefix.  The engine refuses to trade while any required
   indicator is NaN, which is what kills look-ahead bias at the edges.
*  A bounded module-level cache means that when 400 Optuna trials all ask for
   EMA(BTC, 200) it is computed once.  This is the single biggest speed-up in
   the whole project.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ cache ----
_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_CACHE_MAX = 40_000


def cache_clear() -> None:
    _CACHE.clear()


def _cached(key, fn):
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    val = fn()
    val.flags.writeable = False  # cached arrays must never be mutated in place
    _CACHE[key] = val
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return val


# ------------------------------------------------------------ primitives ----
def _nan_prefix(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    k = min(int(k), out.size)
    if k > 0:
        out[:k] = np.nan
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    if n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        c = np.concatenate(([0.0], np.cumsum(x)))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    if n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    s = pd.Series(np.asarray(x, dtype=np.float64))
    out = s.ewm(alpha=2.0 / (n + 1.0), adjust=False).mean().to_numpy(dtype=np.float64)
    return _nan_prefix(out, n - 1)


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder / SMMA smoothing."""
    n = int(n)
    if n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    s = pd.Series(np.asarray(x, dtype=np.float64))
    out = s.ewm(alpha=1.0 / n, adjust=False).mean().to_numpy(dtype=np.float64)
    return _nan_prefix(out, n - 1)


def wma(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    if n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if x.size >= n:
        w = np.arange(1, n + 1, dtype=np.float64)
        out[n - 1:] = np.convolve(x, w[::-1], mode="valid") / w.sum()
    return out


def dema(x: np.ndarray, n: int) -> np.ndarray:
    e1 = ema(x, n)
    # EMA of an array that starts with NaNs: run the second pass on the tail
    e1f = np.nan_to_num(e1, nan=np.asarray(x, dtype=np.float64)[0] if len(x) else 0.0)
    e2 = ema(e1f, n)
    out = 2.0 * e1 - e2
    return _nan_prefix(out, min(2 * (n - 1), out.size))


_MA_FUNCS = (sma, ema, dema, wma, rma)


def moving_average(x: np.ndarray, n: int, kind: int) -> np.ndarray:
    """kind: 0 SMA, 1 EMA, 2 DEMA, 3 WMA, 4 SMMA/RMA."""
    return _MA_FUNCS[int(kind)](x, int(n))


def ma_cached(sym: str, field: str, x: np.ndarray, n: int, kind: int) -> np.ndarray:
    return _cached(("ma", sym, field, int(n), int(kind)), lambda: moving_average(x, n, kind))


# --------------------------------------------------------------- oscillators -
def rsi(close: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    close = np.asarray(close, dtype=np.float64)
    d = np.diff(close, prepend=close[0] if close.size else 0.0)
    d[0] = 0.0
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = rma(up, n)
    ad = rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(ad > 0, au / ad, np.inf)
        out = 100.0 - 100.0 / (1.0 + rs)
    out[~np.isfinite(out) & ~np.isnan(au)] = 100.0
    return _nan_prefix(out, min(n, out.size))


def rsi_smoothed_cached(sym: str, close: np.ndarray, n: int, smooth: int) -> np.ndarray:
    def build():
        base = _cached(("rsi", sym, int(n)), lambda: rsi(close, n))
        return rma(np.nan_to_num(base, nan=50.0), smooth)

    return _cached(("rsis", sym, int(n), int(smooth)), build)


def true_range(high, low, close) -> np.ndarray:
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    close = np.asarray(close, float)
    pc = np.roll(close, 1)
    pc[0] = close[0] if close.size else np.nan
    return np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return _nan_prefix(rma(true_range(high, low, close), n), n)


def atr_cached(sym, high, low, close, n=14) -> np.ndarray:
    return _cached(("atr", sym, int(n)), lambda: atr(high, low, close, n))


def adx(high, low, close, n: int = 14) -> np.ndarray:
    """Classic Wilder ADX."""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    up = np.diff(high, prepend=high[0] if high.size else 0.0)
    dn = -np.diff(low, prepend=low[0] if low.size else 0.0)
    up[0] = dn[0] = 0.0
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_n = rma(true_range(high, low, close), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * rma(plus_dm, n) / tr_n
        mdi = 100.0 * rma(minus_dm, n) / tr_n
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.where(np.isfinite(dx), dx, 0.0)
    return _nan_prefix(rma(dx, n), min(2 * n, dx.size))


def adx_cached(sym, high, low, close, n=14) -> np.ndarray:
    return _cached(("adx", sym, int(n)), lambda: adx(high, low, close, n))


# ------------------------------------------------------------------ rolling --
def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    x = np.asarray(x, float)
    out = pd.Series(x).rolling(n, min_periods=n).max().to_numpy(dtype=np.float64)
    return out


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    return sma(x, n)


def rolling_max_cached(sym, field, x, n) -> np.ndarray:
    return _cached(("rmax", sym, field, int(n)), lambda: rolling_max(x, n))


def dollar_volume_ok(sym, close, volume, quote_volume, window, floor) -> np.ndarray:
    """Trailing `window`-day average daily dollar volume >= floor.

    Uses exchange-reported quote volume when available (more accurate than
    close*volume, which ignores intrabar price path).
    """

    def build():
        dv = quote_volume if quote_volume is not None else np.asarray(close, float) * np.asarray(volume, float)
        avg = sma(np.asarray(dv, float), window)
        return (avg >= floor).astype(np.float64)

    return _cached(("liq", sym, int(window), float(floor)), build) > 0.5


# ---------------------------------------------------------------- crossings --
def cross_above(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    now = a > b
    prev = np.roll(a, 1) <= np.roll(b, 1)
    ok = np.isfinite(a) & np.isfinite(b)
    ok_prev = np.roll(ok, 1)
    ok_prev[0] = False
    out = now & prev & ok & ok_prev
    out[0] = False
    return out


def cross_below(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cross_above(b, a)
