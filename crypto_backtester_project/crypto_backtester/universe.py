"""
Data acquisition + universe construction  (plan steps 2 and 3).

Providers, in order of preference
--------------------------------
1. Binance spot klines via ``data-api.binance.vision`` (public-data mirror, no
   key, no geo block in most places) then the regular api hosts.
2. Bybit v5 spot klines.
3. OKX v5 history-candles.
4. ccxt, if the user happens to have it installed (universal fallback).

Everything is cached to disk so a 40-coin, 8-year download happens once.

Universe filtering is deliberately belt-and-braces:
  * curated deny-lists for stablecoins / wrappers / LSTs / leveraged tokens,
  * *plus* two data-driven detectors that catch anything the lists miss:
      - a peg detector (daily return stdev < 0.5 %),
      - a duplicate detector (return corr > 0.99 AND a near-constant price
        ratio against an already-accepted coin -> WBTC vs BTC, stETH vs ETH).
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from . import config as C

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "crypto-backtester/1.0"})

BINANCE_HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
)


# ============================================================== http helpers ==
def _get(url: str, params: dict | None = None, tries: int | None = None):
    tries = tries or C.MAX_RETRIES
    last = None
    for i in range(tries):
        try:
            r = _SESSION.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2 ** i + 1)
                continue
            if r.status_code >= 400:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 403, 404, 451):
                    return None  # bad symbol / blocked -- no point retrying
                time.sleep(0.6 * (i + 1))
                continue
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            time.sleep(0.6 * (i + 1))
    if last:
        print(f"    [warn] {url} -> {last}")
    return None


# =============================================================== cache layer ==
def _cache_path(provider: str, symbol: str) -> "object":
    return C.DATA_DIR / f"{provider}_{symbol}_{C.INTERVAL}.pkl"


def _cache_load(provider: str, symbol: str, max_age_h: float):
    p = _cache_path(provider, symbol)
    if not p.exists():
        return None
    if max_age_h is not None and (time.time() - p.stat().st_mtime) > max_age_h * 3600:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001
        return None


def _cache_save(provider: str, symbol: str, df: pd.DataFrame) -> None:
    try:
        with open(_cache_path(provider, symbol), "wb") as f:
            pickle.dump(df, f, protocol=4)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] cache write failed for {symbol}: {e}")


# ============================================================ raw OHLCV pulls =
_COLS = ["date", "open", "high", "low", "close", "volume", "quote_volume"]


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLS)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for c in _COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _binance(symbol: str) -> pd.DataFrame | None:
    rows, start = [], 1_400_000_000_000  # 2014-05
    host_ok = None
    while True:
        data = None
        hosts = (host_ok,) if host_ok else BINANCE_HOSTS
        for h in hosts:
            data = _get(f"{h}/api/v3/klines",
                        {"symbol": symbol, "interval": C.INTERVAL,
                         "startTime": start, "limit": 1000})
            if data is not None:
                host_ok = h
                break
        if not data:
            break
        rows += [[k[0], k[1], k[2], k[3], k[4], k[5], k[7]] for k in data]
        if len(data) < 1000:
            break
        start = data[-1][0] + 1
        time.sleep(0.05)
    return _frame(rows) if rows else None


def _bybit(symbol: str) -> pd.DataFrame | None:
    rows, end = [], int(time.time() * 1000)
    for _ in range(60):
        d = _get("https://api.bybit.com/v5/market/kline",
                 {"category": "spot", "symbol": symbol, "interval": "D",
                  "end": end, "limit": 1000})
        lst = ((d or {}).get("result") or {}).get("list") or []
        if not lst:
            break
        rows += [[int(k[0]), k[1], k[2], k[3], k[4], k[5], k[6]] for k in lst]
        end = int(lst[-1][0]) - 1
        if len(lst) < 1000:
            break
        time.sleep(0.05)
    return _frame(rows) if rows else None


def _okx(symbol: str) -> pd.DataFrame | None:
    inst = symbol.replace(C.QUOTE_ASSET, f"-{C.QUOTE_ASSET}")
    rows, after = [], ""
    for _ in range(200):
        p = {"instId": inst, "bar": "1Dutc", "limit": 100}
        if after:
            p["after"] = after
        d = _get("https://www.okx.com/api/v5/market/history-candles", p)
        lst = (d or {}).get("data") or []
        if not lst:
            break
        rows += [[int(k[0]), k[1], k[2], k[3], k[4], k[5], k[7]] for k in lst]
        after = lst[-1][0]
        if len(lst) < 100:
            break
        time.sleep(0.06)
    return _frame(rows) if rows else None


def _ccxt(symbol: str) -> pd.DataFrame | None:
    try:
        import ccxt  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    for name in ("binance", "bybit", "okx", "kucoin", "gateio"):
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            mkt = symbol.replace(C.QUOTE_ASSET, f"/{C.QUOTE_ASSET}")
            rows, since = [], 1_400_000_000_000
            while True:
                batch = ex.fetch_ohlcv(mkt, "1d", since=since, limit=1000)
                if not batch:
                    break
                rows += [[b[0], b[1], b[2], b[3], b[4], b[5], b[4] * b[5]] for b in batch]
                if len(batch) < 1000:
                    break
                since = batch[-1][0] + 1
            if rows:
                return _frame(rows)
        except Exception:  # noqa: BLE001
            continue
    return None


_PROVIDER_FN = {"binance": _binance, "bybit": _bybit, "okx": _okx, "ccxt": _ccxt}

# A geo-blocked or firewalled provider fails identically for every symbol, so
# retrying it 40 times turns a 2-minute download into a 20-minute one. Strike
# a provider out after this many consecutive empty responses.
_STRIKES: dict[str, int] = {}
_MAX_STRIKES = 3


def fetch_ohlcv(base: str, refresh: bool = False) -> tuple[pd.DataFrame | None, str]:
    """Return (dataframe, provider_name) for BASE/QUOTE daily candles."""
    symbol = f"{base}{C.QUOTE_ASSET}"
    for prov in list(C.OHLCV_PROVIDERS) + ["ccxt"]:
        if _STRIKES.get(prov, 0) >= _MAX_STRIKES:
            continue
        if not refresh:
            cached = _cache_load(prov, symbol, C.CACHE_MAX_AGE_HOURS)
            if cached is not None and len(cached):
                return cached, prov
        stale = _cache_load(prov, symbol, None)
        fn = _PROVIDER_FN.get(prov)
        if fn is None:
            continue
        df = fn(symbol)
        if df is not None and len(df) >= 50:
            _STRIKES[prov] = 0
            _cache_save(prov, symbol, df)
            return df, prov
        _STRIKES[prov] = _STRIKES.get(prov, 0) + 1
        if _STRIKES[prov] == _MAX_STRIKES:
            print(f"    [warn] provider '{prov}' returned nothing {_MAX_STRIKES}x -- "
                  f"skipping it for the rest of this run")
        if stale is not None and len(stale):  # network down -> use the old cache
            return stale, prov + "(stale-cache)"
    return None, "none"


# ========================================================= candidate ranking ==
def _coingecko_rank(n: int) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    per = 250
    for page in range(1, int(np.ceil(n / per)) + 1):
        d = _get("https://api.coingecko.com/api/v3/coins/markets",
                 {"vs_currency": "usd", "order": "market_cap_desc",
                  "per_page": per, "page": page, "sparkline": "false"})
        if not d:
            break
        out += [(c["symbol"].upper(), float(c.get("market_cap") or 0.0)) for c in d]
        time.sleep(2.5)  # free tier is rate limited; be polite
    return out[:n]


def _binance_volume_rank(n: int) -> list[tuple[str, float]]:
    for h in BINANCE_HOSTS:
        d = _get(f"{h}/api/v3/ticker/24hr")
        if isinstance(d, list) and d:
            rows = []
            for t in d:
                s = t.get("symbol", "")
                if s.endswith(C.QUOTE_ASSET):
                    rows.append((s[: -len(C.QUOTE_ASSET)], float(t.get("quoteVolume") or 0)))
            rows.sort(key=lambda r: -r[1])
            return rows[:n]
    return []


def rank_candidates(n: int) -> list[tuple[str, float]]:
    src = C.UNIVERSE_SOURCE
    if src == "static":
        return [(s, 0.0) for s in C.STATIC_UNIVERSE]
    if src == "coingecko":
        r = _coingecko_rank(n)
        if r:
            return r
        print("  [warn] CoinGecko unavailable -> falling back to Binance volume rank")
    return _binance_volume_rank(n)


# ============================================================= name filtering =
def name_verdict(base: str) -> tuple[bool, str]:
    b = base.upper()
    if b in C.STABLE_DENY:
        return False, "stablecoin (deny-list)"
    if b in C.WRAPPED_DENY:
        return False, "wrapped / staked derivative (deny-list)"
    for pat in C.LEVERAGED_PATTERNS:
        if b.endswith(pat) and len(b) > len(pat) and b not in C.CORE_LEGACY:
            return False, f"leveraged / index product (suffix '{pat}')"
    if len(b) > 12:
        return False, "implausible ticker length"
    return True, "ok"


def _looks_like_peg(close: np.ndarray) -> bool:
    if close.size < 120:
        return False
    r = np.diff(np.log(np.clip(close, 1e-12, None)))
    return float(np.nanstd(r)) < C.STABLE_MAX_DAILY_VOL


def _is_duplicate(c_new: pd.DataFrame, accepted: list["CoinData"]) -> str | None:
    s_new = c_new.set_index("date")["close"]
    for a in accepted:
        s_old = pd.Series(a.close_raw, index=a.dates_raw)
        j = s_new.index.intersection(s_old.index)
        if len(j) < 200:
            continue
        x, y = s_new.loc[j].to_numpy(float), s_old.loc[j].to_numpy(float)
        rx, ry = np.diff(np.log(x)), np.diff(np.log(y))
        if rx.std() == 0 or ry.std() == 0:
            continue
        corr = float(np.corrcoef(rx, ry)[0, 1])
        ratio = x / y
        cv = float(np.std(ratio) / max(abs(np.mean(ratio)), 1e-12))
        if corr >= C.DUP_MIN_CORR and cv <= C.DUP_MAX_RATIO_CV:
            return f"duplicate of {a.base} (corr={corr:.3f}, ratio CV={cv:.3f})"
    return None


# ================================================================ containers ==
@dataclass
class CoinData:
    base: str
    provider: str
    first_idx: int = 0
    last_idx: int = 0
    open: np.ndarray = field(default_factory=lambda: np.empty(0))
    high: np.ndarray = field(default_factory=lambda: np.empty(0))
    low: np.ndarray = field(default_factory=lambda: np.empty(0))
    close: np.ndarray = field(default_factory=lambda: np.empty(0))
    volume: np.ndarray = field(default_factory=lambda: np.empty(0))
    qvolume: np.ndarray = field(default_factory=lambda: np.empty(0))
    filled: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    halt_idx: int | None = None  # master index of the first halted bar
    # master-grid aligned copies (NaN outside the listed window) -- the engine
    # indexes these directly by master bar number, which removes a whole class
    # of off-by-one bugs.
    m_open: np.ndarray = field(default_factory=lambda: np.empty(0))
    m_high: np.ndarray = field(default_factory=lambda: np.empty(0))
    m_low: np.ndarray = field(default_factory=lambda: np.empty(0))
    m_close: np.ndarray = field(default_factory=lambda: np.empty(0))
    m_alive: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    # raw (pre-grid) copies kept only for the duplicate detector
    dates_raw: object = None
    close_raw: object = None

    def loc(self, t: int) -> int:
        return t - self.first_idx

    def alive(self, t: int) -> bool:
        return self.first_idx <= t <= self.last_idx


@dataclass
class Universe:
    dates: np.ndarray
    coins: list[CoinData]
    btc: int
    start_idx: int
    log: list[dict]
    meta: dict

    @property
    def n(self) -> int:
        return len(self.coins)

    def index_of(self, base: str) -> int | None:
        for i, c in enumerate(self.coins):
            if c.base == base:
                return i
        return None


# ============================================================ build pipeline ==
def build_universe(n_coins: int | None = None, refresh: bool = False,
                   verbose: bool = True) -> Universe:
    n_coins = n_coins or C.N_COINS
    log: list[dict] = []
    accepted: list[CoinData] = []
    seen: set[str] = set()

    want = int(n_coins * C.UNIVERSE_OVERFETCH) + len(C.CORE_LEGACY) + 40
    if verbose:
        print(f"[universe] ranking up to {want} candidates via '{C.UNIVERSE_SOURCE}' ...")
    ranked = rank_candidates(want)
    rank_map = {b: i for i, (b, _) in enumerate(ranked)}
    cap_map = {b: c for b, c in ranked}

    # Priority order: BTC, the legacy must-haves, then market-cap order,
    # then the optional dead-coin survivorship patch.
    order: list[str] = ["BTC"]
    order += [b for b in C.CORE_LEGACY if b != "BTC"]
    order += [b for b, _ in ranked if b not in order]
    tail = [b for b in C.INCLUDE_DELISTED if b not in order]

    def consider(base: str, is_core: bool, is_tail: bool) -> bool:
        if base in seen:
            return False
        seen.add(base)
        ok, why = name_verdict(base)
        if not ok:
            log.append({"base": base, "status": "rejected", "reason": why})
            return False
        df, prov = fetch_ohlcv(base, refresh=refresh)
        if df is None or len(df) < C.MIN_HISTORY_DAYS:
            log.append({"base": base, "status": "rejected",
                        "reason": f"no/insufficient history ({0 if df is None else len(df)} bars)"})
            return False
        cl = df["close"].to_numpy(float)
        if C.DETECT_STABLE_BY_DATA and _looks_like_peg(cl):
            log.append({"base": base, "status": "rejected",
                        "reason": f"peg detected from data (daily vol "
                                  f"{np.nanstd(np.diff(np.log(cl))) * 100:.2f}%)"})
            return False
        if C.DETECT_DUPLICATES_BY_DATA and not is_core:
            dup = _is_duplicate(df, accepted)
            if dup:
                log.append({"base": base, "status": "rejected", "reason": dup})
                return False
        cd = CoinData(base=base, provider=prov)
        cd.dates_raw = df["date"].to_numpy()
        cd.close_raw = cl
        cd._df = df  # noqa: SLF001  (temporary, dropped after gridding)
        accepted.append(cd)
        log.append({"base": base, "status": "accepted",
                    "reason": "core legacy" if is_core else ("survivorship patch" if is_tail else "market-cap rank"),
                    "rank": rank_map.get(base), "market_cap": cap_map.get(base),
                    "bars": len(df), "provider": prov,
                    "first": str(df["date"].iloc[0].date()),
                    "last": str(df["date"].iloc[-1].date())})
        if verbose:
            print(f"    + {base:<8} {len(df):>5} bars  {df['date'].iloc[0].date()} -> "
                  f"{df['date'].iloc[-1].date()}  [{prov}]")
        return True

    core_set = set(C.CORE_LEGACY) | {"BTC"}
    for base in order:
        if len(accepted) >= n_coins and base not in core_set:
            break
        consider(base, base in core_set, False)
    for base in tail:  # extras, they do not displace live coins
        consider(base, False, True)

    if not accepted:
        raise RuntimeError(
            "Universe is empty: no OHLCV provider returned data.\n"
            "  * check connectivity / firewall / VPN (Binance is geo-blocked in some regions)\n"
            "  * try OHLCV_PROVIDERS = ('bybit', 'okx') in config.py\n"
            "  * or `pip install ccxt`, which is used automatically as a last resort")

    uni = _to_grid(accepted, log, verbose=verbose)
    if verbose:
        print(f"[universe] {uni.n} coins | grid {uni.dates[0]} .. {uni.dates[-1]} "
              f"| backtest starts {uni.dates[uni.start_idx]}")
    return uni


def _to_grid(coins: list[CoinData], log: list[dict], verbose=True) -> Universe:
    """Align every coin onto one continuous daily master grid."""
    lo = min(c._df["date"].iloc[0] for c in coins)
    hi = max(c._df["date"].iloc[-1] for c in coins)
    grid = pd.date_range(lo, hi, freq="D")
    gpos = {d: i for i, d in enumerate(grid)}

    for c in coins:
        df = c._df
        local = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D")
        df = df.set_index("date").reindex(local)
        filled = df["close"].isna().to_numpy()
        df = df.ffill()
        df["volume"] = np.where(filled, 0.0, df["volume"].to_numpy(float))
        df["quote_volume"] = np.where(filled, 0.0, df["quote_volume"].to_numpy(float))
        c.first_idx = gpos[local[0]]
        c.last_idx = gpos[local[-1]]
        c.open = df["open"].to_numpy(float)
        c.high = df["high"].to_numpy(float)
        c.low = df["low"].to_numpy(float)
        c.close = df["close"].to_numpy(float)
        c.volume = df["volume"].to_numpy(float)
        c.qvolume = df["quote_volume"].to_numpy(float)
        c.filled = filled

        ng = len(grid)
        sl = slice(c.first_idx, c.last_idx + 1)
        for name, src in (("m_open", c.open), ("m_high", c.high),
                          ("m_low", c.low), ("m_close", c.close)):
            arr = np.full(ng, np.nan)
            arr[sl] = src
            setattr(c, name, arr)
        alive = np.zeros(ng, dtype=bool)
        alive[sl] = True
        c.m_alive = alive

        # halted = DELIST_GAP_DAYS consecutive synthetic bars, or series ends
        run, halt = 0, None
        for i, f in enumerate(filled):
            run = run + 1 if f else 0
            if run >= C.DELIST_GAP_DAYS:
                halt = c.first_idx + i - run + 1
                break
        if halt is None and c.last_idx < len(grid) - 1 - C.DELIST_GAP_DAYS:
            halt = c.last_idx + 1
        c.halt_idx = halt
        del c._df

    # --- start date: first bar where >= START_COVERAGE of coins are tradeable
    n = len(coins)
    cover = np.zeros(len(grid))
    for c in coins:
        cover[c.first_idx: c.last_idx + 1] += 1.0
    cover /= n
    ok = np.where(cover >= C.START_COVERAGE)[0]
    start = int(ok[0]) if ok.size else 0
    start = min(start + C.WARMUP_CAP, len(grid) - 2)  # leave room for warm-up

    if C.HARD_START:
        hs = pd.Timestamp(C.HARD_START)
        start = max(start, int(np.searchsorted(grid.values, np.datetime64(hs))))
    end = len(grid) - 1
    if C.HARD_END:
        he = pd.Timestamp(C.HARD_END)
        end = min(end, int(np.searchsorted(grid.values, np.datetime64(he))))

    btc = next((i for i, c in enumerate(coins) if c.base == "BTC"), 0)
    meta = {
        "coverage_rule": C.START_COVERAGE,
        "coverage_at_start": float(cover[start]),
        "warmup_bars": C.WARMUP_CAP,
        "grid_start": str(grid[0].date()),
        "grid_end": str(grid[-1].date()),
        "end_idx": int(end),
        "n_coins": n,
        "providers": sorted({c.provider for c in coins}),
    }
    return Universe(dates=grid.values, coins=coins, btc=btc, start_idx=start,
                    log=log, meta=meta)


def save_universe_log(uni: Universe, path=None) -> None:
    path = path or (C.OUT_DIR / "universe_selection.json")
    payload = {"meta": uni.meta,
               "accepted": [r for r in uni.log if r["status"] == "accepted"],
               "rejected": [r for r in uni.log if r["status"] == "rejected"]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
