"""
Signal construction (plan steps 4, 5 and the signal-driven half of step 10).

Contract with the engine
------------------------
Every array returned here is aligned to the MASTER date grid and every boolean
at index ``t`` means "this was true at the CLOSE of bar t".  The engine acts on
it at the OPEN of bar ``t+1``.  That one-bar offset is the only thing standing
between you and a beautiful, completely fake equity curve, so it is enforced
here rather than left to the caller.

A signal is additionally forced to False while *any* indicator it depends on is
still inside its warm-up window (NaN).
"""
from __future__ import annotations

import numpy as np

from . import indicators as I
from .universe import CoinData, Universe


# ------------------------------------------------------------------ helpers --
def to_master(local: np.ndarray, coin: CoinData, n: int, fill=np.nan) -> np.ndarray:
    out = np.full(n, fill, dtype=np.float64 if not isinstance(fill, bool) else bool)
    out[coin.first_idx: coin.last_idx + 1] = local
    return out


def _bool_master(local_bool: np.ndarray, coin: CoinData, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[coin.first_idx: coin.last_idx + 1] = local_bool
    return out


def _finite(*arrays) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


# =============================================================== BTC context ==
def btc_regime(uni: Universe, p: dict) -> np.ndarray:
    """Master-aligned boolean: BTC close is above its own MA."""
    n = len(uni.dates)
    if not (p.get("use_btc_entry_gate") or p.get("use_btc_exit_override")):
        return np.ones(n, dtype=bool)
    b = uni.coins[uni.btc]
    ma = I.ma_cached(b.base, "close", b.close, p["btc_ma_len"], p["btc_ma_type"])
    ok = np.isfinite(ma) & (b.close > ma)
    out = np.zeros(n, dtype=bool)
    out[b.first_idx: b.last_idx + 1] = ok
    return out


# ================================================================== entries ===
def entry_signal(coin: CoinData, p: dict, n: int) -> np.ndarray:
    et = p["entry_type"]
    c, o, h, v = coin.close, coin.open, coin.high, coin.volume
    sym = coin.base

    if et == 0:  # ---------------------------------------- MA breakout
        ma = I.ma_cached(sym, "close", c, p["entry_ma_len"], p["entry_ma_type"])
        raw = I.cross_above(c, ma) if p.get("entry_require_cross") else (c > ma)
        raw = raw & np.isfinite(ma)

    elif et == 1:  # -------------------------------------- smoothed RSI x-over
        f = I.rsi_smoothed_cached(sym, c, p["rsi_f_len"], p["rsi_f_smt"])
        s = I.rsi_smoothed_cached(sym, c, p["rsi_s_len"], p["rsi_s_smt"])
        raw = I.cross_above(f, s)

    elif et == 2:  # -------------------------------------- MA cross-over
        short = I.ma_cached(sym, "close", c, p["xover_short_len"], p["xover_short_type"])
        long_ = I.ma_cached(sym, "close", c, p["xover_long_len"], p["xover_long_type"])
        raw = I.cross_above(short, long_)

    else:  # et == 3 -------------------------------------- volume breakout
        vma = I.ma_cached(sym, "volume", v, p["vol_ma_len"], 0)  # SMA per spec
        hh = I.rolling_max_cached(sym, "high", h, p["price_lookback"])
        hh_prev = np.roll(hh, 1)
        hh_prev[0] = np.nan
        a14 = I.atr_cached(sym, h, coin.low, c, 14)
        raw = (
            (v > p["vol_mult"] * vma)
            & (c > hh_prev)
            & ((c - o) >= p["body_atr_mult"] * a14)
            & _finite(vma, hh_prev, a14)
        )

    return _bool_master(np.asarray(raw, bool), coin, n)


# ================================================================== filters ===
def entry_filters(coin: CoinData, p: dict, n: int, btc_ok: np.ndarray) -> np.ndarray:
    """All optional gates AND-ed together, master aligned."""
    from . import config as C

    sym = coin.base
    c, h, l, v = coin.close, coin.high, coin.low, coin.volume

    # 5.4 -- fixed rule, always on
    ok = I.dollar_volume_ok(sym, c, v, coin.qvolume,
                            C.LIQUIDITY_WINDOW, C.LIQUIDITY_FLOOR_USD)

    # 5.2 -- ADX trend strength (0.0 disables)
    if p.get("adx_thresh", 0.0) > 0:
        a = I.adx_cached(sym, h, l, c, 14)
        ok = ok & np.isfinite(a) & (a >= p["adx_thresh"])

    # 5.3 -- RSI trend confirmation, only meaningful for entry_type == 1
    if p["entry_type"] == 1 and p.get("use_rsi_trend_filter"):
        ma = I.ma_cached(sym, "close", c, p["rsi_trend_ma_len"], p["rsi_trend_ma_type"])
        ok = ok & np.isfinite(ma) & (c > ma)

    out = _bool_master(np.asarray(ok, bool), coin, n)

    # 5.1 -- BTC macro regime gate (master level)
    if p.get("use_btc_entry_gate"):
        out = out & btc_ok
    return out


# ============================================== signal-driven primary exits ===
def exit_signal(coin: CoinData, p: dict, n: int) -> np.ndarray:
    xt = p["exit_type"]
    c, sym = coin.close, coin.base

    if xt == 3:
        ma = I.ma_cached(sym, "close", c, p["exit_ma_len"], p["exit_ma_type"])
        raw = np.isfinite(ma) & (c < ma)
    elif xt == 4:
        f = I.rsi_smoothed_cached(sym, c, p["rsi_f_len"], p["rsi_f_smt"])
        s = I.rsi_smoothed_cached(sym, c, p["rsi_s_len"], p["rsi_s_smt"])
        raw = I.cross_below(f, s)
    elif xt == 5:
        short = I.ma_cached(sym, "close", c, p["xover_short_len"], p["xover_short_type"])
        long_ = I.ma_cached(sym, "close", c, p["xover_long_len"], p["xover_long_type"])
        raw = I.cross_below(short, long_)
    else:
        raw = np.zeros(c.size, dtype=bool)

    return _bool_master(np.asarray(raw, bool), coin, n)


# ================================================================== bundle ====
class SignalSet:
    """Everything the engine needs for one coin under one parameter set."""

    __slots__ = ("entry", "exit", "atr", "mom", "trigger_price")

    def __init__(self, entry, exit_, atr, mom, trigger_price):
        self.entry = entry
        self.exit = exit_
        self.atr = atr
        self.mom = mom
        self.trigger_price = trigger_price


def build_signals(uni: Universe, p: dict) -> tuple[list[SignalSet], np.ndarray]:
    n = len(uni.dates)
    btc_ok = btc_regime(uni, p)
    mom_lb = int(p.get("wl_mom_lookback", 20))
    out: list[SignalSet] = []

    btc_mom = None
    if p.get("wl_mom_vs_btc"):
        b = uni.coins[uni.btc]
        bm = np.full(n, np.nan)
        r = np.full(b.close.size, np.nan)
        r[mom_lb:] = b.close[mom_lb:] / b.close[:-mom_lb] - 1.0
        bm[b.first_idx: b.last_idx + 1] = r
        btc_mom = bm

    for coin in uni.coins:
        e = entry_signal(coin, p, n) & entry_filters(coin, p, n, btc_ok)
        x = exit_signal(coin, p, n)

        a14 = I.atr_cached(coin.base, coin.high, coin.low, coin.close, 14)
        atr_m = to_master(a14, coin, n, np.nan)

        r = np.full(coin.close.size, np.nan)
        if coin.close.size > mom_lb:
            r[mom_lb:] = coin.close[mom_lb:] / coin.close[:-mom_lb] - 1.0
        mom = to_master(r, coin, n, np.nan)
        if btc_mom is not None:
            mom = mom - btc_mom

        out.append(SignalSet(e, x, atr_m, mom, to_master(coin.close, coin, n, np.nan)))
    return out, btc_ok
