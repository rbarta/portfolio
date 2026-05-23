"""
momentum.py
-----------
Top Momentum Candidates engine for Portfolio Sentinel.

Source strategy: backtest_cli.py — Momentum + Correlation rotation.

Strategy summary
----------------
1. Compute N-period rolling return on period-end prices (weekly or monthly).
2. Rank all tickers by that momentum score — take the top K candidates.
3. Optionally filter by Bull-Bear Index (BBI) and/or Enhanced Trend Score (ETS).
4. From the filtered pool, select PORTFOLIO_SLOTS holdings that minimise
   average pairwise correlation (or use industry-aware top-K if disabled).
5. Optionally apply inverse-volatility weights across the selected slots.

Public API
----------
get_momentum_cfg(cfg)           -> dict  (settings with defaults)
fetch_momentum_prices(tickers)  -> pd.DataFrame  (daily Close, cached)
fetch_momentum_hl(tickers)      -> (highs_df, lows_df)  (cached)
fetch_vix(days)                 -> pd.Series  (cached)
score_candidates(prices_df,
                 highs_df,
                 lows_df,
                 vix_series,
                 mcfg)          -> dict with keys:
                                     ranked          : pd.DataFrame
                                     selected        : list[str]
                                     inv_vol_weights : dict
                                     decision_date   : Timestamp
                                     vix_rank        : float | None
                                     vix_blocked     : bool
"""

from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
from data_loader import _load_ticker_local, _local_to_ohlcv_frame

# Optional talib -- pure-numpy fallback when absent
try:
    import talib as _talib
    _TALIB_AVAILABLE = True
except ImportError:
    _talib = None
    _TALIB_AVAILABLE = False


# ==============================================================================
# CONFIG
# ==============================================================================

_DEFAULTS: dict = {
    "top_n":             30,      # candidates shown in the tab
    "use_monthly":       False,   # True = monthly, False = weekly
    "weekday":           4,       # 0=Mon ... 4=Fri (used when use_monthly=False)
    "month_end":         True,    # True = month-end, False = month-start
    "mom_lookback":      8,       # N periods for rolling momentum
    "require_positive":  True,    # only consider positive-momentum tickers
    "portfolio_slots":   3,       # final holdings count
    "top_candidates":    6,       # pre-filter count before corr selection
    "corr_lb":           20,      # correlation lookback (trading days)
    "use_correlation":   True,    # use lowest-avg-corr combo selection
    "use_bbi":           False,
    "bbi_lookback":      20,
    "use_ets":           False,
    "ets_window":        14,
    "rsi_period":        14,
    "use_logistic_prob": False,
    "rsr_length":        25,
    "use_vix_rank":      False,
    "max_vix_rank":      70.0,
    "show_invvol":       True,
    "vol_lb":            20,
    "vol_cap":           0.45,
    "cash_proxy":        "SHY",
}


def get_momentum_cfg(cfg: dict) -> dict:
    """Extract 'momentum' block; missing keys fall back to _DEFAULTS."""
    raw = cfg.get("momentum", {})
    return {k: raw.get(k, v) for k, v in _DEFAULTS.items()}


# ==============================================================================
# DATA FETCHING
# ==============================================================================

@st.cache_data(ttl=300)
def fetch_momentum_prices(
    tickers: tuple[str, ...],
    days: int = 400,
    use_yfinance: bool = True,
    data_dir: str = "data",
) -> pd.DataFrame:
    """Unified daily Close DataFrame. 400 days ~ 8 weekly + correlation history."""
    end       = datetime.today()
    start     = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    series: dict = {}

    if use_yfinance:
        for t in tickers:
            try:
                df = yf.download(t, start=start_str, end=end_str,
                                 auto_adjust=True, progress=False)
                if df is None or df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                c = df["Close"].squeeze().dropna()
                if len(c) >= 20:
                    series[t] = c
            except Exception:
                pass
    else:
        for t in tickers:
            df_local = _load_ticker_local(t, data_dir, start_str, end_str)
            if df_local is not None:
                df_local = df_local.ffill().dropna(subset=["close"])
                if len(df_local) >= 20:
                    series[t] = df_local["close"]

    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).dropna(axis=1, how="all").ffill().dropna()


@st.cache_data(ttl=300)
def fetch_momentum_hl(
    tickers: tuple[str, ...],
    days: int = 400,
    use_yfinance: bool = True,
    data_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily High and Low DataFrames (for BBI calculation)."""
    end       = datetime.today()
    start     = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    highs: dict = {}; lows: dict = {}

    if use_yfinance:
        for t in tickers:
            try:
                df = yf.download(t, start=start_str, end=end_str,
                                 auto_adjust=True, progress=False)
                if df is None or df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                highs[t] = df["High"].squeeze().dropna()
                lows[t]  = df["Low"].squeeze().dropna()
            except Exception:
                pass
    else:
        for t in tickers:
            df_local = _load_ticker_local(t, data_dir, start_str, end_str)
            if df_local is not None:
                df_local = df_local.ffill().dropna(subset=["high", "low"])
                highs[t] = df_local["high"]
                lows[t]  = df_local["low"]

    h = pd.concat(highs, axis=1).dropna(axis=1, how="all").ffill() if highs else pd.DataFrame()
    l = pd.concat(lows,  axis=1).dropna(axis=1, how="all").ffill() if lows  else pd.DataFrame()
    return h, l


@st.cache_data(ttl=300)
def fetch_vix(days: int = 400) -> pd.Series:
    """VIX close series with 400-day history for a full 252-bar rank lookback."""
    end   = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = yf.download("^VIX", start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].squeeze().dropna()
    except Exception:
        return pd.Series(dtype=float)


# ==============================================================================
# INDICATOR CALCULATIONS  (extracted from backtest_cli.py)
# ==============================================================================

def _compute_vix_rank(vix: pd.Series, asof, lookback_days: int = 365) -> float | None:
    s = vix.loc[:asof].dropna()
    if s.empty:
        return None
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)
    window = s.loc[s.index >= cutoff]
    if len(window) < 20:
        return None
    return float((window < float(s.iloc[-1])).sum()) / len(window) * 100.0


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    alpha = 1.0 / period
    ag    = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al    = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _rsi_for_candidates(tickers: list, px: pd.DataFrame, asof, period: int = 14) -> dict:
    result: dict = {}
    for t in tickers:
        if t not in px.columns:
            result[t] = 50.0; continue
        s = px[t].loc[:asof].dropna()
        if len(s) < period * 3:
            result[t] = 50.0; continue
        rsi = _compute_rsi(s, period).dropna()
        result[t] = float(round(rsi.iloc[-1], 1)) if not rsi.empty else 50.0
    return result


def _bull_bear_index(hi: pd.Series, lo: pd.Series, lookback: int) -> float:
    from scipy.stats import t as t_dist
    hi = hi.dropna().tail(lookback + 1)
    lo = lo.dropna().tail(lookback + 1)
    common = hi.index.intersection(lo.index)
    hi, lo = hi.loc[common], lo.loc[common]
    if len(hi) < 4:
        return 50.0
    new_hi = (hi > hi.shift(1)).astype(float).iloc[1:].tail(lookback)
    new_lo = (lo < lo.shift(1)).astype(float).iloc[1:].tail(lookback)
    n = len(new_hi)
    if n < 4:
        return 50.0
    nhp = float(new_hi.sum()) / n
    nlp = float(new_lo.sum()) / n
    pc  = (nhp + nlp) / 2.0
    dsq = 2.0 * pc * (1.0 - pc) / n
    if dsq <= 0.0:
        return 100.0 if nhp > nlp else (0.0 if nlp > nhp else 50.0)
    return float(t_dist.cdf((nhp - nlp) / np.sqrt(dsq), df=n - 1) * 100.0)


def _bbi_for_candidates(tickers: list, daily_hi: pd.DataFrame,
                         daily_lo: pd.DataFrame, asof, lookback: int) -> dict:
    result: dict = {}
    for t in tickers:
        if t not in daily_hi.columns or t not in daily_lo.columns:
            result[t] = 50.0; continue
        result[t] = _bull_bear_index(daily_hi[t].loc[:asof].dropna(),
                                     daily_lo[t].loc[:asof].dropna(), lookback)
    return result


def _get_enhanced_trend_score(px: pd.Series, window: int = 14) -> int:
    from sklearn.linear_model import LinearRegression
    df = pd.DataFrame({"close": px.values}, index=px.index)
    df["Net_Change"] = df["close"].diff()
    df["AD_Line"]    = df["Net_Change"].fillna(0).cumsum()
    df["MA_Fast"]    = df["close"].rolling(5).mean()
    df["MA_Slow"]    = df["close"].rolling(20).mean()

    def slope(s: pd.Series) -> float:
        v = s.dropna().values
        if len(v) < window:
            return 0.0
        y = v[-window:].reshape(-1, 1)
        x = np.arange(window).reshape(-1, 1)
        yn = (y - np.mean(y)) / (np.std(y) + 1e-9)
        return float(LinearRegression().fit(x, yn).coef_[0][0])

    ps = slope(df["close"]); ads = slope(df["AD_Line"])
    if abs(ps) < 0.1 and abs(ads) < 0.1:
        return 0
    if ps > 0 and ads > 0:
        return 3 if df["MA_Fast"].iloc[-1] > df["MA_Slow"].iloc[-1] else 2
    if ps < 0 and ads < 0:
        return -3 if df["MA_Fast"].iloc[-1] < df["MA_Slow"].iloc[-1] else -2
    return 1 if ads > ps else -1


def _ets_for_candidates(tickers: list, px: pd.DataFrame, asof, window: int = 14) -> dict:
    min_bars = max(20, window) + window
    result: dict = {}
    for t in tickers:
        if t not in px.columns:
            result[t] = 0; continue
        s = px[t].loc[:asof].dropna()
        if len(s) < min_bars:
            result[t] = 0; continue
        result[t] = _get_enhanced_trend_score(s, window=window)
    return result


def _linearreg_slope_np(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr); out = np.full(n, np.nan)
    x = np.arange(period, dtype=float) - (period - 1) / 2.0
    denom = float((x * x).sum())
    for i in range(period - 1, n):
        y = arr[i - period + 1: i + 1].astype(float)
        if np.any(np.isnan(y)):
            continue
        out[i] = float((x * (y - y.mean())).sum() / denom)
    return out


def _calc_rsr(close: pd.Series, length: int = 25) -> tuple[float, float]:
    arr = close.values.astype(float)
    if len(arr) < length + 1:
        return np.nan, np.nan
    slope_arr = (_talib.LINEARREG_SLOPE(arr, timeperiod=length)
                 if _TALIB_AVAILABLE else _linearreg_slope_np(arr, length))
    s    = pd.Series(arr)
    avg  = s.rolling(length).mean().values[-1]
    stdp = pd.Series(s.pct_change().values * 100.0).rolling(length).std(ddof=0).values[-1]
    sl   = slope_arr[-1]
    if any(np.isnan(v) for v in [sl, avg, stdp]) or stdp == 0:
        return np.nan, np.nan
    denom = avg - 12.0 * sl
    if denom == 0:
        return np.nan, np.nan
    rsr     = ((2400.0 * sl / denom) - 0.015) / (5.0 * stdp)
    logprob = 1.0 / (1.0 + np.exp(-1.5 * rsr))
    return float(rsr), float(logprob)


def _rsr_logprob_for_candidates(tickers: list, px: pd.DataFrame,
                                  asof, length: int = 25) -> tuple[dict, dict]:
    rsr_s: dict = {}; lp_s: dict = {}
    for t in tickers:
        if t not in px.columns:
            rsr_s[t] = lp_s[t] = np.nan; continue
        s = px[t].loc[:asof].dropna()
        if len(s) < length + 2:
            rsr_s[t] = lp_s[t] = np.nan; continue
        rsr_s[t], lp_s[t] = _calc_rsr(s, length=length)
    return rsr_s, lp_s


def _lowest_avg_corr_combo(candidates: list, px: pd.DataFrame,
                            asof, lookback_days: int, k: int) -> list:
    """Pick the k-item subset with the lowest average pairwise correlation."""
    fallback = candidates[:k]
    if len(candidates) <= k:
        return fallback
    rets = (px[candidates].loc[:asof].tail(lookback_days + 5)
            .ffill().pct_change().dropna())
    if rets.shape[0] < max(5, int(lookback_days * 0.6)):
        return fallback
    best, best_score = None, np.inf
    for combo in combinations(candidates, k):
        sub = rets[list(combo)].dropna()
        if sub.shape[0] < 5 or sub.shape[1] < k:
            continue
        c     = sub.corr()
        score = float(np.mean([c.iloc[i, j]
                                for i in range(k) for j in range(i + 1, k)] or [0]))
        if score < best_score:
            best_score, best = score, list(combo)
    return best if best is not None else fallback


def _inv_vol_weights(px: pd.DataFrame, slot_list: list,
                      asof, lookback_days: int = 20, cap: float = 0.45) -> dict:
    uniq = list(dict.fromkeys(slot_list))
    rets = (px[uniq].loc[:asof].tail(lookback_days + 1)
            .ffill().pct_change(fill_method=None).dropna())
    if rets.shape[0] < max(5, int(lookback_days * 0.5)):
        return {t: 1.0 / len(slot_list) for t in slot_list}
    vol = rets.std(ddof=0).replace(0.0, np.nan)
    if vol.isna().any():
        return {t: 1.0 / len(slot_list) for t in slot_list}
    raw = np.array([1.0 / vol[t] for t in slot_list], dtype=float)
    raw /= raw.sum()
    w = {}
    for t, ws in zip(slot_list, raw):
        w[t] = w.get(t, 0.0) + float(ws)
    for _ in range(10):
        over = {t: wt for t, wt in w.items() if wt > cap}
        if not over:
            break
        excess = sum(w[t] - cap for t in over)
        for t in over:
            w[t] = cap
        rest = {t: wt for t, wt in w.items() if t not in over}
        rs = sum(rest.values())
        if rs <= 0:
            return {t: 1.0 / len(slot_list) for t in slot_list}
        for t in rest:
            w[t] += excess * (rest[t] / rs)
    s = sum(w.values())
    return {t: wt / s for t, wt in w.items()} if s > 0 else w


def _period_end_prices(px: pd.DataFrame, use_monthly: bool,
                        month_end: bool, weekday: int) -> pd.DataFrame:
    freq = ("ME" if month_end else "MS") if use_monthly else \
           f"W-{['MON','TUE','WED','THU','FRI','SAT','SUN'][weekday]}"
    return px.resample(freq).last()


# ==============================================================================
# MAIN SCORING FUNCTION
# ==============================================================================

def score_candidates(
    daily_px:   pd.DataFrame,
    daily_high: pd.DataFrame,
    daily_low:  pd.DataFrame,
    vix_series: pd.Series,
    mcfg:       dict,
) -> dict:
    """
    Score and rank all tickers using the Momentum+Correlation strategy from
    backtest_cli.py.

    Returns
    -------
    ranked          : pd.DataFrame  — tickers ranked by N-period momentum score
    selected        : list[str]     — final portfolio slots after corr selection
    inv_vol_weights : dict          — inverse-vol weights for selected slots
    decision_date   : pd.Timestamp  — period-end date used as decision date
    asof_date       : pd.Timestamp  — last trading day <= decision_date
    vix_rank        : float | None  — current VIX percentile (0-100)
    vix_blocked     : bool          — True if VIX rank exceeds max_vix_rank
    top_candidates  : list[str]     — top-K by momentum (before corr filter)
    bbi_scores, ets_scores, rsi_scores : dict  (for display)
    """
    if daily_px.empty:
        return _empty_result()

    use_monthly  = bool(mcfg["use_monthly"])
    month_end    = bool(mcfg["month_end"])
    weekday      = int(mcfg["weekday"])
    mom_lb       = int(mcfg["mom_lookback"])
    req_positive = bool(mcfg["require_positive"])
    top_cands_n  = int(mcfg["top_candidates"])
    slots        = int(mcfg["portfolio_slots"])
    corr_lb      = int(mcfg["corr_lb"])
    use_corr     = bool(mcfg["use_correlation"])
    use_bbi      = bool(mcfg["use_bbi"])
    bbi_lb       = int(mcfg["bbi_lookback"])
    use_ets      = bool(mcfg["use_ets"])
    ets_win      = int(mcfg["ets_window"])
    rsi_period   = int(mcfg["rsi_period"])
    use_lp       = bool(mcfg["use_logistic_prob"])
    rsr_len      = int(mcfg["rsr_length"])
    use_vix_rank = bool(mcfg["use_vix_rank"])
    max_vr       = float(mcfg["max_vix_rank"])
    cash_proxy   = str(mcfg["cash_proxy"])
    vol_lb       = int(mcfg["vol_lb"])
    vol_cap      = float(mcfg["vol_cap"])
    top_n        = int(mcfg["top_n"])

    # Period-end prices + N-period momentum
    wp  = _period_end_prices(daily_px, use_monthly, month_end, weekday).dropna(how="all")
    if len(wp) < mom_lb + 2:
        return _empty_result()
    mom = wp.ffill().pct_change(mom_lb, fill_method=None)

    valid = mom.dropna(how="all")
    if valid.empty:
        return _empty_result()
    decision_date = valid.index[-1]

    avail     = daily_px.index[daily_px.index <= decision_date]
    asof_date = avail[-1] if len(avail) else decision_date

    mom_at_d = mom.loc[decision_date].copy()
    if cash_proxy in mom_at_d.index:
        mom_at_d = mom_at_d.drop(cash_proxy)
    if req_positive:
        mom_at_d = mom_at_d[mom_at_d > 0.0]

    # VIX rank gate
    vix_rank    = None
    vix_blocked = False
    if use_vix_rank and not vix_series.empty:
        vix_rank    = _compute_vix_rank(vix_series, asof_date)
        vix_blocked = vix_rank is not None and vix_rank > max_vr

    # Top candidates by raw momentum
    top_list = (mom_at_d.nlargest(min(top_cands_n, len(mom_at_d))).index.tolist()
                if not mom_at_d.empty else [])
    top_list = [t for t in top_list if t in daily_px.columns]

    # Indicators for top candidates
    bbi_scores = (_bbi_for_candidates(top_list, daily_high, daily_low, asof_date, bbi_lb)
                  if use_bbi and not daily_high.empty else {t: 50.0 for t in top_list})
    ets_scores = (_ets_for_candidates(top_list, daily_px, asof_date, window=ets_win)
                  if use_ets else {t: 0 for t in top_list})
    rsi_scores = _rsi_for_candidates(top_list, daily_px, asof_date, period=rsi_period)
    lp_scores: dict = {}; rsr_scores: dict = {}
    if use_lp and top_list:
        rsr_scores, lp_scores = _rsr_logprob_for_candidates(
            top_list, daily_px, asof_date, length=rsr_len)

    # Indicators for full universe (for ranked table display)
    all_tickers = [t for t in mom.columns if t in daily_px.columns and t != cash_proxy]
    rsi_all = _rsi_for_candidates(all_tickers, daily_px, asof_date, period=rsi_period)
    bbi_all = (_bbi_for_candidates(all_tickers, daily_high, daily_low, asof_date, bbi_lb)
               if use_bbi and not daily_high.empty else {t: 50.0 for t in all_tickers})
    ets_all = (_ets_for_candidates(all_tickers, daily_px, asof_date, window=ets_win)
               if use_ets else {t: 0 for t in all_tickers})

    # Build candidate pool (apply optional BBI / ETS / logprob filters)
    pool = top_list.copy()
    if use_bbi:
        pool = [t for t in pool if bbi_scores.get(t, 50.0) > 50.0]
    if use_ets:
        pool = [t for t in pool if ets_scores.get(t, 0) > 0]
    if use_lp and lp_scores:
        pool = sorted(
            pool,
            key=lambda t: (lp_scores.get(t, -1.0)
                           if not np.isnan(lp_scores.get(t, np.nan)) else -1.0),
            reverse=True,
        )

    # Final slot selection
    selected: list = []
    if not vix_blocked and pool:
        if use_corr and len(pool) > slots:
            selected = _lowest_avg_corr_combo(pool, daily_px, asof_date, corr_lb, slots)
        else:
            selected = pool[:slots]
    while len(selected) < slots:
        selected.append(cash_proxy)

    # Inverse-vol weights for non-cash slots
    non_cash  = [t for t in selected if t != cash_proxy and t in daily_px.columns]
    inv_vol_w: dict = _inv_vol_weights(daily_px, non_cash, asof_date, vol_lb, vol_cap) if non_cash else {}
    if cash_proxy in selected:
        inv_vol_w[cash_proxy] = max(0.0, 1.0 - sum(inv_vol_w.values()))

    # Build ranked DataFrame
    rows = []
    for t in all_tickers:
        m_val = mom.loc[decision_date, t] if t in mom.columns else np.nan
        rows.append({
            "ticker":       t,
            "mom_score":    float(m_val) if not pd.isna(m_val) else None,
            "rsi":          rsi_all.get(t),
            "bbi":          round(bbi_all.get(t, 50.0), 1) if use_bbi else None,
            "ets":          ets_all.get(t) if use_ets else None,
            "logprob":      round(lp_scores.get(t, np.nan), 3) if use_lp else None,
            "rsr":          round(rsr_scores.get(t, np.nan), 3) if use_lp else None,
            "in_top_cands": t in top_list,
            "passes_bbi":   bbi_all.get(t, 50.0) > 50.0 if use_bbi else None,
            "passes_ets":   ets_all.get(t, 0) > 0       if use_ets else None,
            "selected":     t in selected and t != cash_proxy,
        })

    ranked = (pd.DataFrame(rows)
              .sort_values("mom_score", ascending=False, na_position="last")
              .head(top_n)
              .reset_index(drop=True))

    return {
        "ranked":          ranked,
        "selected":        selected,
        "inv_vol_weights": inv_vol_w,
        "decision_date":   decision_date,
        "asof_date":       asof_date,
        "vix_rank":        vix_rank,
        "vix_blocked":     vix_blocked,
        "top_candidates":  top_list,
        "bbi_scores":      bbi_scores,
        "ets_scores":      ets_scores,
        "rsi_scores":      rsi_scores,
    }


def _empty_result() -> dict:
    return {
        "ranked": pd.DataFrame(), "selected": [], "inv_vol_weights": {},
        "decision_date": None, "asof_date": None,
        "vix_rank": None, "vix_blocked": False,
        "top_candidates": [], "bbi_scores": {}, "ets_scores": {}, "rsi_scores": {},
    }
