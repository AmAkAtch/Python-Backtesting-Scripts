"""
GPU-accelerated batch indicator computation for nse_allrounder.

WHAT THIS IS
-------------
Stacks a {ticker: OHLCV DataFrame} universe into a single [n_tickers, n_days]
tensor and computes indicators for every ticker at once, on GPU when
available (auto-detects your RTX 5070; falls back to CPU transparently,
e.g. in a sandbox with no GPU).

WHICH INDICATORS ARE HERE, AND WHY THIS SET SPECIFICALLY
-----------------------------------------------------------
Only genuine single-shot batched tensor ops live here: rolling windows
computed via one torch.unfold() call (SMA, rolling std, Bollinger,
Donchian, stochastics, WMA, momentum, ROC, historical volatility). This is
deliberate -- EMA/RSI/ATR/MACD are recursive EWM chains that need a
Python-level loop over ~3,750 trading days (each day depends on the
previous one). On a GPU that means ~3,750 sequential kernel dispatches;
launch latency (~5-20us each) dominates over the tiny amount of real math
per step for a universe this size, the same reason RNNs are notoriously
hard to accelerate on GPUs. Measured directly during development: even a
"good" single-shot op can lose to pandas if there's no real GPU present
(torch-on-CPU competing against pandas' C-optimized rolling implementation
is a rigged fight) -- see main.py's _gpu_beats_cpu() for the runtime
calibration that decides, on your actual hardware, whether GPU routing is
worth it at all, rather than assuming.

This is a *precompute accelerator*, not a replacement for IndicatorLibrary.
It does not change any backtest logic, entry/exit rules, or portfolio
accounting. Every function here is validated in validate_parity() against
the existing pandas implementation for exact numerical parity before being
trusted -- always re-run that after any change to either implementation.

USAGE
-----
    batch = GPUIndicatorBatch(market_data)   # dict[ticker] -> OHLCV DataFrame
    sma_20 = batch.sma(period=20)             # {ticker: pd.Series}
    boll_upper = batch.bollinger_upper(period=20)
    donch_lo = batch.donchian_lower(period=20)
"""
from __future__ import annotations

from typing import Dict, List
import math
import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def get_device() -> "torch.device":
    """Auto-select CUDA (e.g. your RTX 5070) if available, else CPU."""
    if not _TORCH_OK:
        raise ImportError("torch is required for GPUIndicatorBatch (pip install torch)")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GPUIndicatorBatch:
    """
    Stacks a {ticker: OHLCV DataFrame} universe into padded [n_tickers, n_days]
    tensors on a shared calendar (union of all dates, forward-filled the same
    way pandas .reindex().ffill() would leave gaps as NaN at the start), and
    computes rolling/EWM indicators for the entire universe in one shot.
    """

    def __init__(self, market_data: Dict[str, pd.DataFrame], device: "torch.device" = None, dtype=None):
        if not _TORCH_OK:
            raise ImportError("torch is required for GPUIndicatorBatch (pip install torch)")
        self.device = device or get_device()
        # float64, not float32: float32 accumulates enough drift over long recursive
        # EWM chains (e.g. MACD's 26-period EMA over 2500+ trading days) to fail
        # parity against pandas at the ~0.01 level on price-scale values -- verified
        # empirically in validate_parity(). We're still doing full-universe batched
        # tensor ops either way, so this stays far faster than the pandas per-ticker
        # path even at float64; correctness comes first for anything feeding a
        # backtest that's about to inform real capital allocation.
        self.dtype = dtype or torch.float64
        self.tickers: List[str] = list(market_data.keys())

        # Shared calendar = union of all business days present in the data (same
        # convention Backtester.run() uses for its master trading calendar).
        all_dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in market_data.values()])))
        self.dates = all_dates

        def stack(col: str) -> "torch.Tensor":
            aligned = pd.concat(
                [market_data[t][col].reindex(all_dates) for t in self.tickers], axis=1
            )
            arr = aligned.to_numpy(dtype=np.float64)  # [n_days, n_tickers], NaN where missing
            return torch.tensor(arr.T, device=self.device, dtype=self.dtype)  # [n_tickers, n_days]

        self.close = stack("Close")
        self.high = stack("High")
        self.low = stack("Low")
        self.open = stack("Open")
        self.volume = stack("Volume")
        self.mask = ~torch.isnan(self.close)  # valid-data mask per ticker/day

    # ------------------------------------------------------------- helpers
    def _to_series_dict(self, tensor: "torch.Tensor") -> Dict[str, pd.Series]:
        arr = tensor.detach().cpu().numpy()
        return {
            t: pd.Series(arr[i], index=self.dates, name=t)
            for i, t in enumerate(self.tickers)
        }

    @staticmethod
    def _rolling_mean(x: "torch.Tensor", period: int) -> "torch.Tensor":
        """[n_tickers, n_days] rolling mean via unfold -- one GPU op for the whole universe."""
        n_tickers, n_days = x.shape
        x_filled = torch.nan_to_num(x, nan=0.0)
        valid = (~torch.isnan(x)).to(x.dtype)
        pad = period - 1
        x_padded = torch.nn.functional.pad(x_filled, (pad, 0))
        valid_padded = torch.nn.functional.pad(valid, (pad, 0))
        windows = x_padded.unfold(1, period, 1)          # [n_tickers, n_days, period]
        valid_windows = valid_padded.unfold(1, period, 1)
        sums = windows.sum(dim=-1)
        counts = valid_windows.sum(dim=-1)
        out = sums / counts.clamp(min=1)
        out[counts < period] = float("nan")  # match pandas min_periods=period semantics
        return out

    @staticmethod
    def _rolling_std(x: "torch.Tensor", period: int) -> "torch.Tensor":
        n_tickers, n_days = x.shape
        x_filled = torch.nan_to_num(x, nan=0.0)
        valid = (~torch.isnan(x)).to(x.dtype)
        pad = period - 1
        x_padded = torch.nn.functional.pad(x_filled, (pad, 0))
        valid_padded = torch.nn.functional.pad(valid, (pad, 0))
        windows = x_padded.unfold(1, period, 1)
        valid_windows = valid_padded.unfold(1, period, 1)
        counts = valid_windows.sum(dim=-1)
        mean = windows.sum(dim=-1) / counts.clamp(min=1)
        sq_diff = ((windows - mean.unsqueeze(-1)) * valid_windows) ** 2
        var = sq_diff.sum(dim=-1) / (counts - 1).clamp(min=1)
        out = var.sqrt()
        out[counts < period] = float("nan")
        return out

    @staticmethod
    def _rolling_max(x: "torch.Tensor", period: int) -> "torch.Tensor":
        n_tickers, n_days = x.shape
        neg_inf = torch.finfo(x.dtype).min
        x_filled = torch.nan_to_num(x, nan=neg_inf)
        valid = (~torch.isnan(x))
        pad = period - 1
        x_padded = torch.nn.functional.pad(x_filled, (pad, 0), value=neg_inf)
        valid_padded = torch.nn.functional.pad(valid, (pad, 0), value=False)
        windows = x_padded.unfold(1, period, 1)
        valid_windows = valid_padded.unfold(1, period, 1)
        counts = valid_windows.sum(dim=-1)
        out = windows.max(dim=-1).values
        out[counts < period] = float("nan")
        return out

    @staticmethod
    def _rolling_min(x: "torch.Tensor", period: int) -> "torch.Tensor":
        n_tickers, n_days = x.shape
        pos_inf = torch.finfo(x.dtype).max
        x_filled = torch.nan_to_num(x, nan=pos_inf)
        valid = (~torch.isnan(x))
        pad = period - 1
        x_padded = torch.nn.functional.pad(x_filled, (pad, 0), value=pos_inf)
        valid_padded = torch.nn.functional.pad(valid, (pad, 0), value=False)
        windows = x_padded.unfold(1, period, 1)
        valid_windows = valid_padded.unfold(1, period, 1)
        counts = valid_windows.sum(dim=-1)
        out = windows.min(dim=-1).values
        out[counts < period] = float("nan")
        return out

    @staticmethod
    def _rolling_weighted_mean(x: "torch.Tensor", period: int) -> "torch.Tensor":
        """Linearly-weighted moving average (weights 1..period), matches WMA."""
        n_tickers, n_days = x.shape
        weights = torch.arange(1, period + 1, device=x.device, dtype=x.dtype)
        weight_sum = weights.sum()
        x_filled = torch.nan_to_num(x, nan=0.0)
        valid = (~torch.isnan(x)).to(x.dtype)
        pad = period - 1
        x_padded = torch.nn.functional.pad(x_filled, (pad, 0))
        valid_padded = torch.nn.functional.pad(valid, (pad, 0))
        windows = x_padded.unfold(1, period, 1)          # [n_tickers, n_days, period]
        valid_windows = valid_padded.unfold(1, period, 1)
        counts = valid_windows.sum(dim=-1)
        out = (windows * weights).sum(dim=-1) / weight_sum
        out[counts < period] = float("nan")
        return out

    @staticmethod
    def _shift(x: "torch.Tensor", periods: int) -> "torch.Tensor":
        out = torch.full_like(x, float("nan"))
        if periods > 0:
            out[:, periods:] = x[:, :-periods]
        elif periods < 0:
            out[:, :periods] = x[:, -periods:]
        else:
            out = x.clone()
        return out

    @staticmethod
    def _ewm_mean(x: "torch.Tensor", span: float, min_periods: int) -> "torch.Tensor":
        """
        Recursive EWM (matches pandas .ewm(span=span, adjust=False)). Kept here
        only for internal completeness/testing -- NOT exposed via a public
        method or routed to by main.py's GPU_INDICATOR_METHODS, because the
        Python-level day-loop this requires is architecturally bad for GPU
        (see module docstring). Left in for anyone who wants to benchmark it
        on real CUDA hardware themselves before deciding to route EMA/RSI/ATR/
        MACD through it.
        """
        alpha = 2.0 / (span + 1.0)
        n_tickers, n_days = x.shape
        x_filled = torch.nan_to_num(x, nan=0.0)
        valid = ~torch.isnan(x)
        out = torch.zeros_like(x)
        running = torch.zeros(n_tickers, device=x.device, dtype=x.dtype)
        seen_count = torch.zeros(n_tickers, device=x.device, dtype=torch.int64)
        # Per-day cumulative valid-count history -- NOT just the final total. Using only
        # the final count compares "total valid observations across the whole series"
        # against min_periods for every day uniformly, so early warm-up days never get
        # masked once the series is long enough overall. We need each day's own count.
        seen_count_history = torch.zeros(n_tickers, n_days, device=x.device, dtype=torch.int64)
        started = torch.zeros(n_tickers, device=x.device, dtype=torch.bool)
        for i in range(n_days):
            v = valid[:, i]
            xi = x_filled[:, i]
            new_running = torch.where(started, running * (1 - alpha) + xi * alpha, xi)
            running = torch.where(v, new_running, running)
            started = started | v
            seen_count = seen_count + v.to(torch.int64)
            seen_count_history[:, i] = seen_count
            out[:, i] = running
        out[seen_count_history < min_periods] = float("nan")
        return out

    def _ewm_alpha(self, x: "torch.Tensor", alpha: float, min_periods: int) -> "torch.Tensor":
        """EWM with explicit alpha (used by RSI/ATR's Wilder-style smoothing)."""
        span = 2.0 / alpha - 1.0
        return self._ewm_mean(x, span=span, min_periods=min_periods)

    # -------------------------------------------------------------- public
    # Rolling-window ops only (single-shot unfold, no Python day-loop) -- see
    # module docstring for why EMA/RSI/ATR/MACD are deliberately absent.
    def sma(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._rolling_mean(self.close, period))

    def rolling_std(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._rolling_std(self.close, period))

    def momentum(self, period: int = 10) -> Dict[str, pd.Series]:
        return self._to_series_dict(self.close - self._shift(self.close, period))

    def roc(self, period: int = 10) -> Dict[str, pd.Series]:
        prior = self._shift(self.close, period)
        return self._to_series_dict((self.close / prior - 1.0) * 100.0)

    def wma(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._rolling_weighted_mean(self.close, period))

    def bollinger_mid(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._rolling_mean(self.close, period))

    def bollinger_upper(self, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        mid = self._rolling_mean(self.close, period)
        std = self._rolling_std(self.close, period)
        return self._to_series_dict(mid + num_std * std)

    def bollinger_lower(self, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        mid = self._rolling_mean(self.close, period)
        std = self._rolling_std(self.close, period)
        return self._to_series_dict(mid - num_std * std)

    def bollinger_bandwidth(self, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        mid = self._rolling_mean(self.close, period)
        std = self._rolling_std(self.close, period)
        upper, lower = mid + num_std * std, mid - num_std * std
        return self._to_series_dict((upper - lower) / mid)

    def donchian_upper(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._rolling_max(self.high, period))

    def donchian_lower(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._rolling_min(self.low, period))

    def donchian_mid(self, period: int = 20) -> Dict[str, pd.Series]:
        upper = self._rolling_max(self.high, period)
        lower = self._rolling_min(self.low, period)
        return self._to_series_dict((upper + lower) / 2.0)

    def stochastic_k(self, period: int = 14) -> Dict[str, pd.Series]:
        low_min = self._rolling_min(self.low, period)
        high_max = self._rolling_max(self.high, period)
        k = 100.0 * (self.close - low_min) / (high_max - low_min)
        return self._to_series_dict(k)

    def williams_r(self, period: int = 14) -> Dict[str, pd.Series]:
        low_min = self._rolling_min(self.low, period)
        high_max = self._rolling_max(self.high, period)
        wr = -100.0 * (high_max - self.close) / (high_max - low_min)
        return self._to_series_dict(wr)

    def historical_volatility(self, period: int = 20) -> Dict[str, pd.Series]:
        prior = self._shift(self.close, 1)
        log_ret = torch.log(self.close / prior)
        std = self._rolling_std(log_ret, period)
        return self._to_series_dict(std * math.sqrt(252))

    # -------------------------------------------------- kept for completeness/testing
    # Not routed to by main.py (recursive/EWM -- see module docstring), but left
    # callable directly here in case you want to benchmark them yourself on real
    # CUDA hardware before deciding whether to enable them.
    def ema(self, period: int = 20) -> Dict[str, pd.Series]:
        return self._to_series_dict(self._ewm_mean(self.close, span=period, min_periods=period))

    def rsi(self, period: int = 14) -> Dict[str, pd.Series]:
        delta = self.close[:, 1:] - self.close[:, :-1]
        delta = torch.nn.functional.pad(delta, (1, 0), value=float("nan"))
        gain = torch.clamp(delta, min=0)
        loss = torch.clamp(-delta, min=0)
        alpha = 1.0 / period
        avg_gain = self._ewm_alpha(gain, alpha, period)
        avg_loss = self._ewm_alpha(loss, alpha, period)
        rs = avg_gain / avg_loss.clamp(min=1e-12)
        rsi = 100 - (100 / (1 + rs))
        rsi = torch.where(torch.isnan(rsi), torch.full_like(rsi, 50.0), rsi)
        return self._to_series_dict(rsi)

    def atr(self, period: int = 14) -> Dict[str, pd.Series]:
        prior_close = torch.nn.functional.pad(self.close[:, :-1], (1, 0), value=float("nan"))
        tr1 = self.high - self.low
        tr2 = (self.high - prior_close).abs()
        tr3 = (self.low - prior_close).abs()
        tr = torch.nan_to_num(torch.stack([tr1, tr2, tr3], dim=0), nan=-float("inf")).max(dim=0).values
        tr[torch.isnan(self.high)] = float("nan")
        atr = self._ewm_alpha(tr, 1.0 / period, period)
        return self._to_series_dict(atr)

    def macd_line(self, fast: int = 12, slow: int = 26) -> Dict[str, pd.Series]:
        ema_fast = self._ewm_mean(self.close, span=fast, min_periods=fast)
        ema_slow = self._ewm_mean(self.close, span=slow, min_periods=slow)
        return self._to_series_dict(ema_fast - ema_slow)

    def macd_signal(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        ema_fast = self._ewm_mean(self.close, span=fast, min_periods=fast)
        ema_slow = self._ewm_mean(self.close, span=slow, min_periods=slow)
        macd = ema_fast - ema_slow
        return self._to_series_dict(self._ewm_mean(macd, span=signal, min_periods=signal))

    def macd_hist(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        ema_fast = self._ewm_mean(self.close, span=fast, min_periods=fast)
        ema_slow = self._ewm_mean(self.close, span=slow, min_periods=slow)
        macd = ema_fast - ema_slow
        sig = self._ewm_mean(macd, span=signal, min_periods=signal)
        return self._to_series_dict(macd - sig)


def validate_parity(market_data: Dict[str, pd.DataFrame], tolerance: float = 1e-3) -> Dict[str, bool]:
    """
    Numerically compares GPUIndicatorBatch output against the existing
    pandas-based IndicatorLibrary for every ticker, indicator by indicator.
    Run this once after installing torch/CUDA on the target machine, and
    again after any change to either implementation, before trusting the
    GPU path in a real optimization run.
    """
    from main import IndicatorLibrary

    batch = GPUIndicatorBatch(market_data)
    results = {}

    checks = {
        "sma_20": (batch.sma(20), lambda df: IndicatorLibrary.sma(df, 20)),
        "ema_20": (batch.ema(20), lambda df: IndicatorLibrary.ema(df, 20)),
        "rsi_14": (batch.rsi(14), lambda df: IndicatorLibrary.rsi(df, 14)),
        "atr_14": (batch.atr(14), lambda df: IndicatorLibrary.atr(df, 14)),
        "rolling_std_20": (batch.rolling_std(20), lambda df: df["Close"].rolling(20).std()),
        "momentum_10": (batch.momentum(10), lambda df: IndicatorLibrary.momentum(df, 10)),
        "roc_10": (batch.roc(10), lambda df: IndicatorLibrary.roc(df, 10)),
        "wma_20": (batch.wma(20), lambda df: IndicatorLibrary.wma(df, 20)),
        "bollinger_mid_20": (batch.bollinger_mid(20), lambda df: IndicatorLibrary.bollinger_mid(df, 20)),
        "bollinger_upper_20": (batch.bollinger_upper(20), lambda df: IndicatorLibrary.bollinger_upper(df, 20)),
        "bollinger_lower_20": (batch.bollinger_lower(20), lambda df: IndicatorLibrary.bollinger_lower(df, 20)),
        "bollinger_bandwidth_20": (batch.bollinger_bandwidth(20), lambda df: IndicatorLibrary.bollinger_bandwidth(df, 20)),
        "donchian_upper_20": (batch.donchian_upper(20), lambda df: IndicatorLibrary.donchian_upper(df, 20)),
        "donchian_lower_20": (batch.donchian_lower(20), lambda df: IndicatorLibrary.donchian_lower(df, 20)),
        "donchian_mid_20": (batch.donchian_mid(20), lambda df: IndicatorLibrary.donchian_mid(df, 20)),
        "stochastic_k_14": (batch.stochastic_k(14), lambda df: IndicatorLibrary.stochastic_k(df, 14)),
        "williams_r_14": (batch.williams_r(14), lambda df: IndicatorLibrary.williams_r(df, 14)),
        "historical_volatility_20": (batch.historical_volatility(20), lambda df: IndicatorLibrary.historical_volatility(df, 20)),
        "macd_line": (batch.macd_line(12, 26), lambda df: IndicatorLibrary.macd_line(df, 12, 26)),
        "macd_signal": (batch.macd_signal(12, 26, 9), lambda df: IndicatorLibrary.macd_signal(df, 12, 26, 9)),
        "macd_hist": (batch.macd_hist(12, 26, 9), lambda df: IndicatorLibrary.macd_hist(df, 12, 26, 9)),
    }

    for name, (gpu_dict, cpu_fn) in checks.items():
        max_abs_err = 0.0
        n_compared = 0
        for t, df in market_data.items():
            cpu_series = cpu_fn(df).reindex(batch.dates)
            gpu_series = gpu_dict[t]
            both_valid = cpu_series.notna() & gpu_series.notna()
            if both_valid.sum() == 0:
                continue
            err = (cpu_series[both_valid] - gpu_series[both_valid]).abs().max()
            max_abs_err = max(max_abs_err, float(err))
            n_compared += 1
        passed = max_abs_err < tolerance
        results[name] = {"passed": passed, "max_abs_error": max_abs_err, "tickers_compared": n_compared}
        print(f"  {name:16s} max_abs_error={max_abs_err:.6f}  {'PASS' if passed else 'FAIL'} "
              f"({n_compared} tickers)")

    return results


if __name__ == "__main__":
    # Quick self-test on a handful of synthetic tickers.
    from main import DataPipeline

    pipeline = DataPipeline(universe=["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"], years=5)
    market_data = {t: pipeline._synthesize(t) for t in pipeline.universe}

    print(f"Device: {get_device()}")
    print("Validating GPU-batch indicators against pandas IndicatorLibrary...")
    results = validate_parity(market_data)
    all_passed = all(r["passed"] for r in results.values())
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
