"""
indicators.py
─────────────
All technical indicator calculations live here.

INDICATOR REGISTRY
──────────────────
Each entry in INDICATOR_REGISTRY is a dict with:

  key         - unique string id, used as the key in the indicators{} dict
  label       - human-readable display name for the sidebar checkbox
  description - one-line legend text
  requires    - list of OHLCV columns needed beyond "Close"
  compute     - fn(df, bench_close) -> dict of result values stored in indicators[ticker]
  score       - fn(ind_values, day_chg) -> int  (-2 ... +2)
                contribution to composite score (return 0 if not applicable)
  badge       - fn(ind_values) -> (css_class, text) | None
                css_class: "ind-bull" | "ind-bear" | "ind-warn" | "ind-neut"
                Return None to suppress the badge entirely.

To ADD a new indicator:
  1. Write helper calc functions below.
  2. Add a new entry to INDICATOR_REGISTRY.
  That's it - the rest of the app picks it up automatically.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any


# ==============================================================================
# RAW CALCULATION HELPERS
# ==============================================================================

def _last(s: pd.Series | None) -> float | None:
    if s is None: return None
    clean = s.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None


# -- RSI -----------------------------------------------------------------------
def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# -- MACD ----------------------------------------------------------------------
def _calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast,   adjust=False).mean()
    ema_s = close.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal,  adjust=False).mean()
    return macd, sig, macd - sig   # line, signal, histogram


def _macd_label(hist: pd.Series) -> str:
    if hist is None or len(hist.dropna()) < 2:
        return "neutral"
    v = hist.dropna()
    last, prev = float(v.iloc[-1]), float(v.iloc[-2])
    if last > 0 and last > prev:  return "bullish"
    if last < 0 and last < prev:  return "bearish"
    if last > 0:                  return "weakening_bull"
    return "weakening_bear"


# -- Bollinger Bands -----------------------------------------------------------
def _calc_bollinger(close: pd.Series, period=20, num_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + num_std * std, sma, sma - num_std * std  # upper, mid, lower


def _bollinger_label(close: pd.Series, upper: pd.Series, lower: pd.Series) -> str:
    price = _last(close)
    u     = _last(upper)
    lo    = _last(lower)
    if price is None or u is None or lo is None:
        return "neutral"
    if price >= u:  return "overbought"
    if price <= lo: return "oversold"
    band_width = u - lo
    mid        = lo + band_width / 2
    if price > mid + band_width * 0.25: return "upper_half"
    if price < mid - band_width * 0.25: return "lower_half"
    return "mid_band"


# -- Moving Averages -----------------------------------------------------------
def _calc_ma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def _detect_ma_cross(close: pd.Series) -> str:
    ma50  = _calc_ma(close, 50).dropna()
    ma200 = _calc_ma(close, 200).dropna()
    if len(ma50) < 2 or len(ma200) < 2:
        return "insufficient"
    diff = (ma50 - ma200).reindex(ma200.index).dropna()
    if len(diff) < 2:
        return "insufficient"
    cur, prev = diff.iloc[-1], diff.iloc[-2]
    if cur > 0 and prev <= 0:  return "golden_cross"
    if cur < 0 and prev >= 0:  return "death_cross"
    return "bullish" if cur > 0 else "bearish"


# -- ATR -----------------------------------------------------------------------
def _calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    cp = c.shift(1)
    tr = pd.concat([(h - l), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# -- Beta ----------------------------------------------------------------------
def _calc_beta(stock_close: pd.Series, bench_close: pd.Series) -> float | None:
    sr  = stock_close.pct_change().dropna()
    br  = bench_close.pct_change().dropna()
    idx = sr.index.intersection(br.index)
    if len(idx) < 30:
        return None
    cov = np.cov(sr.loc[idx], br.loc[idx])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None


# -- RSC -----------------------------------------------------------------------
def _calc_rsc(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series | None:
    idx = stock_close.index.intersection(bench_close.index)
    return stock_close.loc[idx] / bench_close.loc[idx] if len(idx) >= 5 else None


# -- Volume Ratio --------------------------------------------------------------
def _calc_volume_ratio(volume: pd.Series, period=20) -> float | None:
    if volume is None or len(volume) < period + 2:
        return None
    avg = volume.iloc[-(period + 1):-1].mean()
    return float(volume.iloc[-1] / avg) if avg > 0 else None


# -- ADL (Accumulation / Distribution Line) ------------------------------------
def _calc_adl(df: pd.DataFrame) -> pd.Series:
    """
    Accumulation/Distribution Line.
    ADL_t = ADL_{t-1} + Money Flow Volume_t
    Money Flow Volume  = Money Flow Multiplier x Volume
    Money Flow Multiplier = ((Close-Low) - (High-Close)) / (High-Low)
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    vol   = df["Volume"]
    hl    = (high - low).replace(0, np.nan)
    mfm   = ((close - low) - (high - close)) / hl
    mfm   = mfm.fillna(0)
    mfv   = mfm * vol
    return mfv.cumsum()


def _adl_label(adl: pd.Series, close: pd.Series, window: int = 10) -> str:
    """
    Returns signal by comparing:
      - ADL trend (rising/falling over `window` bars)
      - Price trend
    and detecting divergence.
    """
    if adl is None or len(adl.dropna()) < window + 2:
        return "insufficient"
    a  = adl.dropna()
    c  = close.reindex(a.index).dropna()
    if len(a) < window + 2 or len(c) < window + 2:
        return "insufficient"

    adl_rising   = a.iloc[-1]  > a.iloc[-window]
    price_rising = c.iloc[-1]  > c.iloc[-window]

    if adl_rising and price_rising:         return "accumulation"
    if not adl_rising and not price_rising: return "distribution"
    if adl_rising and not price_rising:     return "bullish_divergence"
    return "bearish_divergence"


# ==============================================================================
# INDICATOR REGISTRY
# ==============================================================================
# Each entry is the single source of truth for one indicator:
#   compute  -> raw values stored in indicators[ticker]
#   score    -> integer contribution to composite score
#   badge    -> (css_class, text) pill on flag cards, or None to hide

INDICATOR_REGISTRY: list[dict] = [

    # -- RSI -------------------------------------------------------------------
    {
        "key":         "rsi",
        "label":       "RSI (14)",
        "description": "RSI <30 = Oversold · >70 = Overbought",
        "requires":    [],
        "compute": lambda df, bench: {
            "rsi_series": _calc_rsi(df["Close"]),
            "rsi":        _last(_calc_rsi(df["Close"])),
        },
        "score": lambda ind, day_chg: (
            -2 if (ind.get("rsi") or 50) < 25 else
            -1 if (ind.get("rsi") or 50) < 40 else
             2 if (ind.get("rsi") or 50) > 75 else
             1 if (ind.get("rsi") or 50) > 60 else 0
        ),
        "badge": lambda ind: (
            ("ind-bear", f"RSI {ind['rsi']:.0f} {'OS' if ind['rsi'] < 30 else ''}")
            if ind.get("rsi") and ind["rsi"] < 40 else
            ("ind-bull", f"RSI {ind['rsi']:.0f} {'OB' if ind['rsi'] > 70 else ''}")
            if ind.get("rsi") and ind["rsi"] > 60 else
            ("ind-neut", f"RSI {ind['rsi']:.0f}")
            if ind.get("rsi") else None
        ),
    },

    # -- MACD ------------------------------------------------------------------
    {
        "key":         "macd",
        "label":       "MACD (12/26/9)",
        "description": "MACD histogram direction = momentum",
        "requires":    [],
        "compute": lambda df, bench: (lambda l, s, h: {
            "macd_line": l, "macd_signal": s, "macd_hist": h,
            "macd_label": _macd_label(h),
        })(*_calc_macd(df["Close"])),
        "score": lambda ind, _: {
            "bullish":       1, "weakening_bull": 1,
            "bearish":      -1, "weakening_bear": -1,
        }.get(ind.get("macd_label", ""), 0),
        "badge": lambda ind: {
            "bullish":        ("ind-bull", "MACD BULL"),
            "weakening_bull": ("ind-warn", "MACD WEAK+"),
            "bearish":        ("ind-bear", "MACD BEAR"),
            "weakening_bear": ("ind-warn", "MACD WEAK-"),
        }.get(ind.get("macd_label", ""), ("ind-neut", "MACD NEUT")),
    },

    # -- Bollinger Bands -------------------------------------------------------
    {
        "key":         "bollinger",
        "label":       "Bollinger Bands (20,2s)",
        "description": "Price position relative to Bollinger Bands",
        "requires":    [],
        "compute": lambda df, bench: (lambda u, m, lo: {
            "bb_upper": u, "bb_mid": m, "bb_lower": lo,
            "bb_label": _bollinger_label(df["Close"], u, lo),
        })(*_calc_bollinger(df["Close"])),
        "score": lambda ind, _: {
            "overbought": -1, "oversold": 1,
        }.get(ind.get("bb_label", ""), 0),
        "badge": lambda ind: {
            "overbought":  ("ind-bear", "BB OB"),
            "oversold":    ("ind-bull", "BB OS"),
            "upper_half":  ("ind-bull", "BB UP"),
            "lower_half":  ("ind-bear", "BB DN"),
            "mid_band":    ("ind-neut", "BB MID"),
        }.get(ind.get("bb_label", ""), None),
    },

    # -- MA Cross --------------------------------------------------------------
    {
        "key":         "ma_cross",
        "label":       "MA Cross (50 / 200)",
        "description": "Golden Cross / Death Cross detection",
        "requires":    [],
        "compute": lambda df, bench: {
            "ma50":     _calc_ma(df["Close"], 50),
            "ma200":    _calc_ma(df["Close"], 200),
            "ma_cross": _detect_ma_cross(df["Close"]),
        },
        "score": lambda ind, _: {
            "golden_cross": 2, "bullish":  1,
            "death_cross": -2, "bearish": -1,
        }.get(ind.get("ma_cross", ""), 0),
        "badge": lambda ind: {
            "golden_cross": ("ind-bull", "GOLDEN X"),
            "bullish":      ("ind-bull", "50>200"),
            "death_cross":  ("ind-bear", "DEATH X"),
            "bearish":      ("ind-bear", "50<200"),
            "insufficient": ("ind-neut", "MA N/A"),
        }.get(ind.get("ma_cross", ""), None),
    },

    # -- ATR -------------------------------------------------------------------
    {
        "key":         "atr",
        "label":       "ATR% (14)",
        "description": "ATR% = avg true range as % of price (volatility, not directional)",
        "requires":    ["High", "Low"],
        "compute": lambda df, bench: (
            lambda atr_s, price: {
                "atr":     _last(atr_s),
                "atr_pct": (_last(atr_s) / price * 100) if (_last(atr_s) and price) else None,
            }
        )(_calc_atr(df), float(df["Close"].iloc[-1])),
        "score": lambda ind, _: 0,  # volatility context only
        "badge": lambda ind: (
            ("ind-warn", f"ATR {ind['atr_pct']:.1f}%")
            if ind.get("atr_pct") and ind["atr_pct"] > 3 else
            ("ind-neut", f"ATR {ind['atr_pct']:.1f}%")
            if ind.get("atr_pct") else None
        ),
    },

    # -- Beta ------------------------------------------------------------------
    {
        "key":         "beta",
        "label":       "Beta (b)",
        "description": "b = stock sensitivity vs benchmark (context, not directional)",
        "requires":    [],
        "compute": lambda df, bench: {
            "beta": _calc_beta(df["Close"], bench) if bench is not None else None,
        },
        "score": lambda ind, _: 0,  # context only
        "badge": lambda ind: (
            ("ind-warn", f"b {ind['beta']:.2f}")
            if ind.get("beta") is not None and abs(ind["beta"]) > 1.5 else
            ("ind-neut", f"b {ind['beta']:.2f}")
            if ind.get("beta") is not None else None
        ),
    },

    # -- RSC -------------------------------------------------------------------
    {
        "key":         "rsc",
        "label":       "RSC (vs Benchmark)",
        "description": "Rising RSC = accelerating vs benchmark",
        "requires":    [],
        "compute": lambda df, bench: (lambda rsc: {
            "rsc":       rsc,
            "rsc_trend": (
                "rising"  if (rsc is not None and len(rsc) > 10
                              and rsc.iloc[-1] > rsc.iloc[-10]) else
                "falling" if rsc is not None and len(rsc) > 10 else None
            ),
        })(_calc_rsc(df["Close"], bench) if bench is not None else None),
        "score": lambda ind, _: (
            1 if ind.get("rsc_trend") == "rising" else
           -1 if ind.get("rsc_trend") == "falling" else 0
        ),
        "badge": lambda ind: (
            ("ind-bull", "RSC UP") if ind.get("rsc_trend") == "rising" else
            ("ind-bear", "RSC DN") if ind.get("rsc_trend") == "falling" else None
        ),
    },

    # -- Volume Ratio ----------------------------------------------------------
    {
        "key":         "volume",
        "label":       "Volume Ratio (20D)",
        "description": "VOL = last day vs 20-day avg -- high vol on down day = distribution",
        "requires":    ["Volume"],
        "compute": lambda df, bench: {
            "vol_ratio": _calc_volume_ratio(
                df["Volume"] if "Volume" in df.columns else None
            ),
        },
        "score": lambda ind, day_chg: (
            -1 if (ind.get("vol_ratio") or 0) > 1.5 and (day_chg or 0) < 0 else
             1 if (ind.get("vol_ratio") or 0) > 1.5 and (day_chg or 0) > 0 else 0
        ),
        "badge": lambda ind: (
            ("ind-bull", f"VOL {ind['vol_ratio']:.1f}x")
            if (ind.get("vol_ratio") or 0) > 1.5 else
            ("ind-warn", f"VOL {ind['vol_ratio']:.1f}x")
            if (ind.get("vol_ratio") or 0) < 0.5 else
            ("ind-neut", f"VOL {ind['vol_ratio']:.1f}x")
            if ind.get("vol_ratio") else None
        ),
    },

    # -- ADL -------------------------------------------------------------------
    {
        "key":         "adl",
        "label":       "ADL (Acc/Dist Line)",
        "description": "Rising ADL = accumulation · Falling = distribution · Divergence signals reversals",
        "requires":    ["High", "Low", "Volume"],
        "compute": lambda df, bench: (lambda adl: {
            "adl":       adl,
            "adl_label": _adl_label(adl, df["Close"]),
        })(
            _calc_adl(df)
            if all(c in df.columns for c in ["High", "Low", "Volume"])
            else None
        ),
        "score": lambda ind, _: {
            "accumulation":       1,
            "bullish_divergence": 2,
            "distribution":      -1,
            "bearish_divergence":-2,
        }.get(ind.get("adl_label", ""), 0),
        "badge": lambda ind: {
            "accumulation":       ("ind-bull", "ADL ACCUM"),
            "bullish_divergence": ("ind-bull", "ADL BULL DIV"),
            "distribution":       ("ind-bear", "ADL DIST"),
            "bearish_divergence": ("ind-bear", "ADL BEAR DIV"),
            "insufficient":       ("ind-neut", "ADL N/A"),
        }.get(ind.get("adl_label", ""), None),
    },

]


# ==============================================================================
# PUBLIC API
# ==============================================================================

def compute_all_indicators(
    df: pd.DataFrame,
    bench_close: pd.Series | None,
    day_chg: float | None,
) -> dict[str, Any]:
    """
    Run every registered indicator against the given OHLCV DataFrame.
    Returns a flat dict of all computed values plus a composite score.
    """
    result: dict[str, Any] = {}

    for reg in INDICATOR_REGISTRY:
        required_cols = set(reg.get("requires", []))
        if required_cols and not required_cols.issubset(df.columns):
            continue  # silently skip if required data not present
        try:
            vals = reg["compute"](df, bench_close)
            result.update(vals)
        except Exception:
            pass  # robustness: one bad indicator never crashes the app

    # Composite score - sum contributions from every indicator
    total = 0
    for reg in INDICATOR_REGISTRY:
        try:
            total += reg["score"](result, day_chg)
        except Exception:
            pass
    result["composite"] = max(-4, min(4, total))

    return result


def badge_html(ind: dict, enabled_keys: set[str]) -> str:
    """
    Render badge pills for the given indicator dict.
    Only badges whose registry key is in `enabled_keys` are shown.
    """
    parts = []
    for reg in INDICATOR_REGISTRY:
        if reg["key"] not in enabled_keys:
            continue
        try:
            result = reg["badge"](ind)
            if result is None:
                continue
            css_cls, text = result
            parts.append(
                f'<span class="ind-badge {css_cls}">{text}</span>'
            )
        except Exception:
            pass
    return "".join(parts)


def registry_labels() -> dict[str, str]:
    """Return {key: label} for all registered indicators - used to build sidebar checkboxes."""
    return {r["key"]: r["label"] for r in INDICATOR_REGISTRY}


def registry_descriptions() -> dict[str, str]:
    """Return {key: description} for the legend panel."""
    return {r["key"]: r["description"] for r in INDICATOR_REGISTRY}
