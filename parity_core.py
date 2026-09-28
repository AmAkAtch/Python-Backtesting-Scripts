#!/usr/bin/env python3
"""
Shared parity kernel for the V10.x NSE/Crypto live companions.

The objective is to remove indicator/sizing drift between the live companions
and their respective V10.x backtest engines. It intentionally contains no
broker/exchange execution API.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import copy
import numpy as np
import pandas as pd


def shift_1d(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    res = np.empty_like(arr)
    res[0] = fill_value
    res[1:] = arr[:-1]
    return res


def np_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < w:
        return out
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    clean = np.where(valid_mask, arr, 0.0)
    cumsum = np.cumsum(clean)
    cumsum = np.insert(cumsum, 0, 0.0)
    vals = (cumsum[w:] - cumsum[:-w]) / float(w)
    out[w - 1:] = vals
    out[:first_valid + w - 1] = np.nan
    return out


def np_rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    n = len(arr)
    if n < w:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    windows = sliding_window_view(arr, window_shape=w)
    out[w - 1:] = np.max(windows, axis=-1)
    out[:first_valid + w - 1] = np.nan
    return out


def np_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < w:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    windows = sliding_window_view(arr, window_shape=w)
    out[w - 1:] = np.std(windows, axis=-1, ddof=0)
    out[:first_valid + w - 1] = np.nan
    return out


def np_ewm_mean(arr: np.ndarray, span: int) -> np.ndarray:
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    out[first_valid] = arr[first_valid]
    for i in range(first_valid + 1, n):
        val = arr[i]
        if np.isnan(val):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * val + (1.0 - alpha) * out[i - 1]
    return out


class FastIndicators:
    @staticmethod
    def moving_average(arr: np.ndarray, length: int, kind: int) -> np.ndarray:
        length = max(2, int(length))
        if kind == 0:
            return np_rolling_mean(arr, length)
        elif kind == 1:
            return np_ewm_mean(arr, length)
        elif kind == 2:
            e1 = np_ewm_mean(arr, length)
            e2 = np_ewm_mean(e1, length)
            return 2.0 * e1 - e2
        elif kind == 3:
            valid_mask = ~np.isnan(arr)
            out = np.full_like(arr, np.nan, dtype=np.float64)
            if not np.any(valid_mask):
                return out
            first_valid = int(np.argmax(valid_mask))
            n_valid = len(arr) - first_valid
            if n_valid < length:
                return out
            w = np.arange(1, length + 1, dtype=float)
            w_norm = w / w.sum()
            clean_tail = arr[first_valid:]
            conv = np.convolve(clean_tail, w_norm[::-1], mode="full")[:n_valid]
            conv[:length - 1] = np.nan
            out[first_valid:] = conv
            return out
        elif kind == 4:
            alpha = 1.0 / length
            span = int(round((2.0 / alpha) - 1.0))
            return np_ewm_mean(arr, span)
        return np_rolling_mean(arr, length)

    @staticmethod
    def atr_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
        cp = shift_1d(c, fill_value=c[0])
        tr1 = h - l
        tr2 = np.abs(h - cp)
        tr3 = np.abs(l - cp)
        tr = np.fmax(tr1, np.fmax(tr2, tr3))
        alpha = 1.0 / max(2, length)
        span = int(round((2.0 / alpha) - 1.0))
        return np_ewm_mean(tr, span)

    @staticmethod
    def adx_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
        length = max(2, int(length))
        span = int(round((2.0 * length) - 1.0))
        up = h - shift_1d(h, fill_value=h[0])
        down = shift_1d(l, fill_value=l[0]) - l
        p_dm = np.where((up > down) & (up > 0.0), up, 0.0)
        m_dm = np.where((down > up) & (down > 0.0), down, 0.0)
        atr_arr = FastIndicators.atr_1d(h, l, c, length)
        p_di = 100.0 * np_ewm_mean(p_dm, span) / (atr_arr + 1e-9)
        m_di = 100.0 * np_ewm_mean(m_dm, span) / (atr_arr + 1e-9)
        dx = 100.0 * np.abs(p_di - m_di) / (p_di + m_di + 1e-9)
        return np_ewm_mean(dx, span)

    @staticmethod
    def rsi_smoothed(c: np.ndarray, length: int, smooth: int) -> np.ndarray:
        length, smooth = max(2, int(length)), max(1, int(smooth))
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        span = int(round((2.0 * length) - 1.0))
        avg_gain = np_ewm_mean(gain, span)
        avg_loss = np_ewm_mean(loss, span)
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return np_rolling_mean(rsi, smooth)

    @staticmethod
    def bollinger_bands(c: np.ndarray, length: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = max(2, int(length))
        mid = np_rolling_mean(c, length)
        std = np_rolling_std(c, length)
        return mid + (std_mult * std), mid, mid - (std_mult * std)


@dataclass
class SignalArrays:
    raw_signal: np.ndarray
    entry: np.ndarray
    state: np.ndarray
    exit_sig: np.ndarray
    atr: np.ndarray
    adx: np.ndarray


def compile_single_symbol(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    dvol30_arr: np.ndarray,
    alive_arr: np.ndarray,
    macro_ok: np.ndarray,
    p: Dict[str, Any],
    liquidity_floor: float,
) -> SignalArrays:
    n = len(close_arr)
    atr14 = FastIndicators.atr_1d(high_arr, low_arr, close_arr, 14)
    adx14 = FastIndicators.adx_1d(high_arr, low_arr, close_arr, 14)

    et, xt = int(p["entry_type"]), int(p["exit_type"])
    raw_entry = np.zeros(n, dtype=bool)
    state_entry = np.zeros(n, dtype=bool)

    if et == 0:
        ma = FastIndicators.moving_average(close_arr, p["entry_ma_len"], p["entry_ma_type"])
        raw_entry = (~np.isnan(ma)) & (close_arr > ma)
        state_entry = raw_entry.copy()
    elif et == 1:
        rf = FastIndicators.rsi_smoothed(close_arr, p["rsi_f_len"], p["rsi_f_smt"])
        rs = FastIndicators.rsi_smoothed(close_arr, p["rsi_s_len"], p["rsi_s_smt"])
        raw_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs) & (shift_1d(rf) <= shift_1d(rs))
        if p.get("use_rsi_trend_filter", False):
            rma = FastIndicators.moving_average(close_arr, p["rsi_trend_ma_len"], p["rsi_trend_ma_type"])
            raw_entry = raw_entry & (~np.isnan(rma)) & (close_arr > rma)
            state_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs) & (~np.isnan(rma)) & (close_arr > rma)
        else:
            state_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs)
    elif et == 2:
        s_ma = FastIndicators.moving_average(close_arr, p["xover_short_len"], p["xover_short_type"])
        l_ma = FastIndicators.moving_average(close_arr, p["xover_long_len"], p["xover_long_type"])
        raw_entry = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma > l_ma) & (shift_1d(s_ma) <= shift_1d(l_ma))
        state_entry = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma > l_ma)
    elif et == 3:
        v_shifted = shift_1d(volume_arr)
        vma = np_rolling_mean(v_shifted, int(p["vol_ma_len"]))
        h_shifted = shift_1d(high_arr)
        hhv = np_rolling_max(h_shifted, int(p["price_lookback"]))
        body = close_arr - open_arr
        raw_entry = (~np.isnan(vma)) & (~np.isnan(hhv)) & (volume_arr > (p["vol_mult"] * vma)) & \
            (close_arr > hhv) & (body >= (p["body_atr_mult"] * atr14))
        base_ma = FastIndicators.moving_average(close_arr, int(p["price_lookback"]), 0)
        state_entry = (~np.isnan(base_ma)) & (close_arr > base_ma)
    elif et == 4:
        b_up, b_mid, _ = FastIndicators.bollinger_bands(close_arr, p["bb_entry_len"], p["bb_entry_std"])
        raw_entry = (~np.isnan(b_up)) & (close_arr > b_up) & (shift_1d(close_arr) <= shift_1d(b_up))
        state_entry = (~np.isnan(b_mid)) & (close_arr > b_mid)

    liq_ok = dvol30_arr >= float(liquidity_floor)
    adx_t = float(p.get("adx_thresh", 0.0))
    adx_ok = (adx14 >= adx_t) if adx_t > 0.0 else np.ones(n, dtype=bool)

    raw_signal = raw_entry & alive_arr
    entry = raw_entry & liq_ok & macro_ok & adx_ok & alive_arr
    state = state_entry & liq_ok & macro_ok & adx_ok & alive_arr

    exit_sig = np.zeros(n, dtype=bool)
    if xt == 3:
        ma_val = FastIndicators.moving_average(close_arr, p["exit_ma_len"], p["exit_ma_type"])
        exit_sig = (~np.isnan(ma_val)) & (close_arr < ma_val)
    elif xt == 4:
        rf = FastIndicators.rsi_smoothed(close_arr, p["exit_rsi_f_len"], p["exit_rsi_f_smt"])
        rs = FastIndicators.rsi_smoothed(close_arr, p["exit_rsi_s_len"], p["exit_rsi_s_smt"])
        exit_sig = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf < rs)
    elif xt == 5:
        s_ma = FastIndicators.moving_average(close_arr, p["exit_xover_short_len"], p["exit_xover_short_type"])
        l_ma = FastIndicators.moving_average(close_arr, p["exit_xover_long_len"], p["exit_xover_long_type"])
        exit_sig = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma < l_ma)
    elif xt == 6:
        v_shifted = shift_1d(volume_arr)
        vma = np_rolling_mean(v_shifted, int(p["exit_vol_ma_len"]))
        exit_sig = (~np.isnan(vma)) & (volume_arr > (p["exit_vol_mult"] * vma)) & (close_arr < open_arr)
    elif xt == 7:
        _, b_mid, _ = FastIndicators.bollinger_bands(close_arr, p["bb_exit_len"], 2.0)
        exit_sig = (~np.isnan(b_mid)) & (close_arr < b_mid)

    return SignalArrays(raw_signal, entry, state, exit_sig, atr14, adx14)


def reconstitute_params(params: Dict[str, Any], max_xover_long: int = 200, max_exit_xover_long: int = 200) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    for k in list(p.keys()):
        if isinstance(p[k], float):
            p[k] = round(p[k], 2)
    if p.get("entry_type") == 2 and "xover_gap" in p:
        p["xover_long_len"] = min(max_xover_long, p["xover_short_len"] + p["xover_gap"])
    if p.get("exit_type") == 5 and "exit_xover_gap" in p:
        p["exit_xover_long_len"] = min(max_exit_xover_long, p["exit_xover_short_len"] + p["exit_xover_gap"])
    p.setdefault("use_market_macro_system", False)
    p.setdefault("macro_active_exit", False)
    p.setdefault("max_pyramid_layers", 1)
    p.setdefault("adx_thresh", 0.0)
    p.setdefault("be_trigger_atr", 0.0)
    p.setdefault("trail_atr_mult", 0.0)
    p.setdefault("use_global_tp", False)
    p.setdefault("wl_mode", "WL_NONE")
    return p
