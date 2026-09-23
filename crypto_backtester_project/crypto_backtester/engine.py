"""
The portfolio simulator (plan steps 6-11).

Bar protocol -- read this before changing anything
--------------------------------------------------
For master bar ``t`` the engine does, strictly in this order:

  0. credit the monthly contribution if t is the first bar of a month
  1. force-liquidate anything that just got delisted / halted
  2. BTC macro override + signal-driven exits, filled at OPEN[t]
     (flags were raised at the CLOSE of t-1)
  3. entries, filled at OPEN[t], using cash freed in step 2
  4. intrabar risk pass on every live tranche using HIGH[t]/LOW[t] against a
     stop level that was frozen at the CLOSE of t-1
  5. mark to market at CLOSE[t], ratchet peaks and trailing stops for t+1

Consequences that matter:
  * no signal ever reads the bar it trades on -> no look-ahead;
  * a stop can fire on the same bar an entry is filled (realistic);
  * when a stop and the take-profit are both inside one bar, the STOP wins
    (we cannot see the intrabar path, so we take the pessimistic branch).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .signals import build_signals
from .universe import Universe

EPS = 1e-12


# ================================================================= tranche ====
@dataclass
class Tranche:
    tid: int
    ci: int
    shadow: bool
    entry_t: int
    entry_price: float
    qty: float
    cost: float
    atr_entry: float
    stop: float
    peak: float
    tp_level: float
    tp_done: bool = False
    layer: int = 1
    trigger_t: int = -1
    trigger_price: float = np.nan
    proceeds: float = 0.0
    qty_left: float = 0.0
    wait_days: int = 0
    from_watchlist: bool = False

    def __post_init__(self):
        self.qty_left = self.qty


# ================================================================== result ====
@dataclass
class Result:
    ok: bool = True
    reason: str = ""
    dates: object = None
    equity: np.ndarray = field(default_factory=lambda: np.empty(0))
    mv: np.ndarray = field(default_factory=lambda: np.empty(0))
    cash: np.ndarray = field(default_factory=lambda: np.empty(0))
    basis: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_open: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_wl: np.ndarray = field(default_factory=lambda: np.empty(0))
    r_sleeve: np.ndarray = field(default_factory=lambda: np.empty(0))
    contrib: np.ndarray = field(default_factory=lambda: np.empty(0))
    trades: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    missed: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ============================================================ cost modelling ==
def _slip(atr_ratio: float) -> float:
    return C.SLIPPAGE_BPS / 1e4 + C.SLIPPAGE_ATR_COEF * (atr_ratio if np.isfinite(atr_ratio) else 0.0)


def _buy(px: float, usd: float, atr_ratio: float) -> tuple[float, float]:
    """Return (qty, effective_fill_price). `usd` is all-in cash out."""
    fill = px * (1.0 + _slip(atr_ratio))
    fee = usd * C.TAKER_FEE_BPS / 1e4
    qty = max(usd - fee, 0.0) / max(fill, EPS)
    return qty, fill


def _sell(px: float, qty: float, cost_part: float, atr_ratio: float) -> float:
    fill = px * (1.0 - _slip(atr_ratio))
    gross = qty * fill
    net = gross * (1.0 - C.TAKER_FEE_BPS / 1e4)
    gain = net - cost_part
    if gain > 0 and C.TAX_RATE_ON_GAINS > 0:
        net -= gain * C.TAX_RATE_ON_GAINS
    return net


# =================================================================== engine ===
def run_backtest(uni: Universe, p: dict, *, trial=None, collect=False,
                 start_idx: int | None = None, end_idx: int | None = None) -> Result:
    sigs, btc_ok = build_signals(uni, p)
    dates = uni.dates
    n = len(dates)
    t0 = uni.start_idx if start_idx is None else start_idx
    t1 = int(uni.meta.get("end_idx", n - 1)) if end_idx is None else end_idx
    t0 = max(t0, 1)
    if t1 <= t0 + 30:
        return Result(ok=False, reason="window too short")

    coins = uni.coins
    nc = len(coins)
    btc_ci = uni.btc

    # ---- flatten signals into per-bar lists (much faster than scanning) ----
    ent = np.vstack([s.entry for s in sigs])
    exi = np.vstack([s.exit for s in sigs])
    ent_ci, ent_t = np.nonzero(ent)
    exi_ci, exi_t = np.nonzero(exi)
    ent_by_bar: dict[int, list[int]] = {}
    for ci, t in zip(ent_ci.tolist(), ent_t.tolist()):
        ent_by_bar.setdefault(t, []).append(ci)
    exi_by_bar: dict[int, list[int]] = {}
    for ci, t in zip(exi_ci.tolist(), exi_t.tolist()):
        exi_by_bar.setdefault(t, []).append(ci)

    atr = [s.atr for s in sigs]
    mom = [s.mom for s in sigs]
    op = [c.m_open for c in coins]
    hi = [c.m_high for c in coins]
    lo = [c.m_low for c in coins]
    cl = [c.m_close for c in coins]
    alive = [c.m_alive for c in coins]
    halt = [c.halt_idx for c in coins]

    d = pd.DatetimeIndex(dates)
    is_month_start = np.zeros(n, dtype=bool)
    ym = d.year * 100 + d.month
    is_month_start[0] = True
    is_month_start[1:] = ym[1:] != ym[:-1]
    years = d.year.to_numpy()

    # ---- parameters ----
    xt = p["exit_type"]
    wl_mode = p["wl_mode"]
    max_layers = int(p["max_pyramid_layers"])
    pyr_gap = int(p.get("pyramid_min_gap", 0))
    pyr_profit = bool(p.get("pyramid_require_profit", False))
    wl_max_age = int(p.get("wl_max_age", 0))  # 0 = never expire
    use_tp = bool(p.get("use_global_tp", False))
    tp_mult = float(p.get("tp_mult", 0.0))
    tp_size = float(p.get("tp_size_pct", 0.0)) / 100.0
    tp_be = bool(p.get("tp_move_sl_be", False))
    use_btc_exit = bool(p.get("use_btc_exit_override", False))
    safety = bool(p.get("use_safety_stop", False))
    safety_mult = float(p.get("safety_sl_mult", 0.0))

    # ---- state ----
    cash = float(C.INITIAL_CASH)
    live: list[Tranche] = []
    wl: list[Tranche] = []
    tid = 0
    last_entry_t = np.full(nc, -10**9)

    equity = np.zeros(n)
    mvs = np.zeros(n)
    cashs = np.zeros(n)
    basis = np.zeros(n)
    nopen = np.zeros(n)
    nwl = np.zeros(n)
    rsl = np.zeros(n)
    contrib = np.zeros(n)

    trades: list[dict] = []
    fills: list[dict] = []
    missed: list[dict] = []
    mv_prev = 0.0
    max_concurrent = 0

    def _mk_stop(ci: int, t: int, px: float, a: float) -> float:
        if xt == 0:
            return px - p["sl_mult"] * a
        if xt == 1:
            return px * (1.0 - p["trail_pct"] / 100.0)
        if xt == 2:
            return px - p["exit_atr_mult"] * a
        if safety:
            return px - safety_mult * a
        return -np.inf

    def _ratchet(tr: Tranche, t: int) -> None:
        a = atr[tr.ci][t]
        if not np.isfinite(a):
            a = tr.atr_entry
        if xt == 0:
            cand = tr.peak - p["trail_mult"] * a
        elif xt == 1:
            cand = tr.peak * (1.0 - p["trail_pct"] / 100.0)
        elif xt == 2:
            cand = tr.peak - p["exit_atr_mult"] * a
        else:
            return  # signal exits do not trail; safety stop stays fixed
        if np.isfinite(cand) and cand > tr.stop:
            tr.stop = cand

    def _close(tr: Tranche, t: int, price: float, frac: float, reason: str) -> float:
        """Sell `frac` of the remaining quantity. Returns cash credited."""
        nonlocal cash
        q = tr.qty_left * frac
        if q <= 0:
            return 0.0
        cost_part = tr.cost * (q / max(tr.qty, EPS))
        a = atr[tr.ci][t]
        ar = (a / cl[tr.ci][t]) if (np.isfinite(a) and np.isfinite(cl[tr.ci][t]) and cl[tr.ci][t] > 0) else 0.0
        net = _sell(price, q, cost_part, ar) if price > 0 else 0.0
        tr.qty_left -= q
        tr.proceeds += net
        if not tr.shadow:
            cash += net
            if collect:
                fills.append({"tid": tr.tid, "coin": coins[tr.ci].base,
                              "date": dates[t], "price": float(price),
                              "qty": float(q), "net": float(net),
                              "reason": reason, "partial": frac < 0.999})
        if tr.qty_left <= tr.qty * 1e-6:
            _retire(tr, t, reason)
            return net
        return net

    def _retire(tr: Tranche, t: int, reason: str) -> None:
        pnl = tr.proceeds - tr.cost
        rec = {
            "tid": tr.tid, "coin": coins[tr.ci].base, "shadow": tr.shadow,
            "entry_date": dates[tr.entry_t], "exit_date": dates[t],
            "entry_price": float(tr.entry_price),
            "exit_price": float(cl[tr.ci][t]) if np.isfinite(cl[tr.ci][t]) else 0.0,
            "cost": float(tr.cost), "proceeds": float(tr.proceeds),
            "pnl": float(pnl), "ret": float(pnl / max(tr.cost, EPS)),
            "bars": int(t - tr.entry_t), "layer": int(tr.layer),
            "reason": reason, "year": int(years[tr.entry_t]),
            "exit_year": int(years[t]),
            "from_watchlist": bool(tr.from_watchlist),
            "wait_days": int(tr.wait_days),
        }
        if tr.shadow:
            if collect:
                missed.append(rec)
        else:
            trades.append(rec)

    def _open(ci: int, t: int, trig_t: int, trig_px: float, shadow: bool,
              layer: int, wait: int, from_wl: bool) -> Tranche | None:
        nonlocal tid, cash
        px = op[ci][t]
        if not np.isfinite(px) or px <= 0:
            return None
        a = atr[ci][t - 1] if np.isfinite(atr[ci][t - 1]) else atr[ci][t]
        if not np.isfinite(a) or a <= 0:
            return None
        ar = a / max(px, EPS)
        qty, fill = _buy(px, C.TRANCHE_USD, ar)
        tid += 1
        tr = Tranche(tid=tid, ci=ci, shadow=shadow, entry_t=t, entry_price=fill,
                     qty=qty, cost=C.TRANCHE_USD, atr_entry=a,
                     stop=_mk_stop(ci, t, fill, a), peak=fill,
                     tp_level=(fill + tp_mult * a) if use_tp else np.inf,
                     layer=layer, trigger_t=trig_t, trigger_price=trig_px,
                     wait_days=wait, from_watchlist=from_wl)
        if not shadow:
            cash -= C.TRANCHE_USD
            last_entry_t[ci] = t
        return tr

    def _layers(ci: int) -> int:
        return sum(1 for x in live if x.ci == ci) + sum(1 for x in wl if x.ci == ci)

    # ==================================================================== loop
    year_marks = np.where(np.diff(years[t0:t1 + 1]) != 0)[0] + t0
    yi = 0

    for t in range(t0, t1 + 1):
        # ---- 0. monthly funding -------------------------------------------
        if is_month_start[t]:
            cash += C.MONTHLY_CONTRIBUTION
            contrib[t] = C.MONTHLY_CONTRIBUTION
        buys = sells = 0.0

        # ---- 1. delisting / halt write-offs --------------------------------
        for tr in list(live):
            h = halt[tr.ci]
            if (h is not None and t >= h) or not alive[tr.ci][t]:
                px = 0.0 if C.DELIST_VALUATION <= 0 else cl[tr.ci][max(t - 1, 0)] * C.DELIST_VALUATION
                before = cash
                _close(tr, max(t - 1, 0), px, 1.0, "DELISTED")
                sells += cash - before
                live.remove(tr)
        for tr in list(wl):
            h = halt[tr.ci]
            if (h is not None and t >= h) or not alive[tr.ci][t]:
                _close(tr, max(t - 1, 0), 0.0, 1.0, "DELISTED")
                wl.remove(tr)

        # ---- 2. signal / macro exits at the open ---------------------------
        macro_kill = use_btc_exit and (not btc_ok[t - 1])
        sig_exit = set(exi_by_bar.get(t - 1, ()))

        for tr in list(live):
            reason = None
            if macro_kill and (tr.ci != btc_ci or C.BTC_EXIT_LIQUIDATES_BTC):
                reason = "BTC_MACRO_EXIT"
            elif tr.ci in sig_exit:
                reason = "EXIT_SIGNAL"
            if reason and np.isfinite(op[tr.ci][t]):
                before = cash
                _close(tr, t, op[tr.ci][t], 1.0, reason)
                sells += cash - before
                live.remove(tr)

        for tr in list(wl):
            if (macro_kill and C.BTC_EXIT_CLEARS_WATCHLIST and tr.ci != btc_ci) or tr.ci in sig_exit:
                if np.isfinite(op[tr.ci][t]):
                    _close(tr, t, op[tr.ci][t], 1.0,
                           "BTC_MACRO_EXIT" if macro_kill else "EXIT_SIGNAL")
                    wl.remove(tr)

        if wl_max_age > 0:
            for tr in list(wl):
                if t - tr.trigger_t > wl_max_age:
                    _close(tr, t, cl[tr.ci][t - 1], 1.0, "WL_EXPIRED")
                    wl.remove(tr)

        # ---- 3. entries -----------------------------------------------------
        fresh = [ci for ci in ent_by_bar.get(t - 1, ()) if alive[ci][t]]
        cand: list[tuple] = []  # (key, is_shadow, ci, trigger_t, trigger_px, obj)
        for tr in wl:
            if alive[tr.ci][t]:
                cand.append((tr.ci, tr.trigger_t, tr.trigger_price, tr))
        if wl_mode == 5:
            cand = []  # no watchlist: only today's signals are eligible
            for tr in list(wl):
                wl.remove(tr)
        for ci in fresh:
            if any(x[3] is not None and x[0] == ci and x[1] == t - 1 for x in cand):
                continue
            cand.append((ci, t - 1, cl[ci][t - 1], None))

        if cand and (cash >= C.TRANCHE_USD - 1e-9 or wl_mode != 5):
            px_now = np.array([cl[c[0]][t - 1] for c in cand], dtype=float)
            trig = np.array([c[2] for c in cand], dtype=float)
            if wl_mode == 0:  # deepest discount below the trigger level
                key = -(trig - px_now) / np.where(trig > 0, trig, np.nan)
            elif wl_mode == 1:  # least drift from the breakout bar
                key = np.abs(px_now - trig) / np.where(trig > 0, trig, np.nan)
            elif wl_mode == 2:  # strongest (relative) momentum
                key = -np.array([mom[c[0]][t - 1] for c in cand], dtype=float)
            elif wl_mode == 3:  # first come first served
                key = np.array([c[1] for c in cand], dtype=float)
            elif wl_mode == 4:  # last come first served
                key = -np.array([c[1] for c in cand], dtype=float)
            else:  # WL_NONE -> momentum tiebreak among same-day signals
                key = -np.array([mom[c[0]][t - 1] for c in cand], dtype=float)
            key = np.where(np.isfinite(key), key, 1e18)
            order = np.argsort(key, kind="stable")
        else:
            order = np.arange(len(cand))

        for k in order:
            ci, trig_t, trig_px, obj = cand[k]
            live_layers = sum(1 for x in live if x.ci == ci)
            if live_layers >= max_layers:
                if obj is not None and obj in wl:
                    _close(obj, t, cl[ci][t - 1], 1.0, "LAYER_CAP")
                    wl.remove(obj)
                continue
            gap_ok = (t - last_entry_t[ci]) >= pyr_gap
            prof_ok = True
            if pyr_profit and live_layers > 0:
                held = [x for x in live if x.ci == ci]
                prof_ok = np.isfinite(cl[ci][t - 1]) and all(cl[ci][t - 1] > x.entry_price for x in held)
            if cash >= C.TRANCHE_USD - 1e-9 and gap_ok and prof_ok:
                wait = (t - obj.entry_t) if obj is not None else 0
                tr = _open(ci, t, trig_t, trig_px, False, live_layers + 1, wait, obj is not None)
                if tr is not None:
                    live.append(tr)
                    buys += C.TRANCHE_USD
                    if obj is not None and obj in wl:
                        obj.qty_left = 0.0
                        wl.remove(obj)  # converted, not a miss
            elif wl_mode != 5 and obj is None and _layers(ci) < max_layers:
                sh = _open(ci, t, trig_t, trig_px, True, _layers(ci) + 1, 0, False)
                if sh is not None:
                    wl.append(sh)

        # ---- 4. intrabar risk pass (stops frozen at the close of t-1) -------
        for tr in list(live):
            h, l = hi[tr.ci][t], lo[tr.ci][t]
            if not np.isfinite(h):
                continue
            if np.isfinite(tr.stop) and l <= tr.stop:
                fill = min(op[tr.ci][t], tr.stop)  # gap-through protection
                before = cash
                _close(tr, t, fill, 1.0, "STOP")
                sells += cash - before
                live.remove(tr)
                continue
            if use_tp and not tr.tp_done and h >= tr.tp_level:
                before = cash
                _close(tr, t, tr.tp_level, tp_size, "TAKE_PROFIT")
                sells += cash - before
                tr.tp_done = True
                if tp_be:
                    tr.stop = max(tr.stop, tr.entry_price)
                if tr.qty_left <= tr.qty * 1e-6 and tr in live:
                    live.remove(tr)

        for tr in list(wl):  # shadows: same rules, no cash
            h, l = hi[tr.ci][t], lo[tr.ci][t]
            if not np.isfinite(h):
                continue
            if np.isfinite(tr.stop) and l <= tr.stop:
                _close(tr, t, min(op[tr.ci][t], tr.stop), 1.0, "STOP")
                wl.remove(tr)
                continue
            if use_tp and not tr.tp_done and h >= tr.tp_level:
                _close(tr, t, tr.tp_level, tp_size, "TAKE_PROFIT")
                tr.tp_done = True
                if tp_be:
                    tr.stop = max(tr.stop, tr.entry_price)
                if tr.qty_left <= tr.qty * 1e-6 and tr in wl:
                    wl.remove(tr)

        # ---- 4b. end of test: liquidate at the final close -------------------
        if t == t1:
            for tr in list(live):
                px = cl[tr.ci][t1]
                before = cash
                _close(tr, t1, px if np.isfinite(px) else 0.0, 1.0, "END_OF_TEST")
                sells += cash - before
                live.remove(tr)
            for tr in list(wl):
                _close(tr, t1, cl[tr.ci][t1], 1.0, "END_OF_TEST")
                wl.remove(tr)

        # ---- 5. mark to market + ratchet ------------------------------------
        mv = 0.0
        cb = 0.0
        for tr in live:
            c_ = cl[tr.ci][t]
            if np.isfinite(c_):
                mv += tr.qty_left * c_
                cb += tr.cost * (tr.qty_left / max(tr.qty, EPS))
                if c_ > tr.peak:
                    tr.peak = c_
            h_ = hi[tr.ci][t]
            if np.isfinite(h_) and h_ > tr.peak:
                tr.peak = h_
            _ratchet(tr, t)
        for tr in wl:
            h_ = hi[tr.ci][t]
            if np.isfinite(h_) and h_ > tr.peak:
                tr.peak = h_
            _ratchet(tr, t)

        denom = mv_prev + buys
        rsl[t] = ((mv + sells - buys - mv_prev) / denom) if denom > EPS else 0.0
        equity[t] = cash + mv
        mvs[t] = mv
        cashs[t] = cash
        basis[t] = cb
        nopen[t] = len(live)
        nwl[t] = len(wl)
        max_concurrent = max(max_concurrent, len(live))
        mv_prev = mv

        # ---- optional Optuna pruning at year boundaries ---------------------
        if trial is not None and yi < len(year_marks) and t == year_marks[yi]:
            yi += 1
            if yi >= C.PRUNER_WARMUP_YEARS:
                paid = C.INITIAL_CASH + float(np.sum(contrib[t0:t + 1]))
                cap = max(float(np.mean(basis[t0:t + 1])), C.TRANCHE_USD)
                part = float((equity[t] - paid) / cap)
                trial.report(part, yi)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

    res = Result(ok=True, dates=dates[t0:t1 + 1],
                 equity=equity[t0:t1 + 1], mv=mvs[t0:t1 + 1],
                 cash=cashs[t0:t1 + 1], basis=basis[t0:t1 + 1],
                 n_open=nopen[t0:t1 + 1], n_wl=nwl[t0:t1 + 1],
                 r_sleeve=rsl[t0:t1 + 1], contrib=contrib[t0:t1 + 1],
                 trades=trades, fills=fills, missed=missed)
    res.stats = {
        "t0": t0, "t1": t1,
        "start": str(pd.Timestamp(dates[t0]).date()),
        "end": str(pd.Timestamp(dates[t1]).date()),
        "max_concurrent": int(max_concurrent),
        "n_tranches": len(trades),
        "n_shadow_expired": len(missed),
    }
    return res
