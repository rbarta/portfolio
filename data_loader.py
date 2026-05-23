"""
data_loader.py
──────────────
All price data fetching — either live from yfinance or from local
CSV / HDF5 files in a user-specified directory.

Data source is controlled by two settings passed in from the sidebar:
  use_yfinance : bool  — True = download live, False = load from disk
  data_dir     : str   — path to local data folder (used when use_yfinance=False)

Public API
──────────
fetch_ohlcv(tickers, display_days, use_yfinance, data_dir)
    → dict[str, pd.DataFrame]   OHLCV frames keyed by ticker

fetch_current_info(tickers, use_yfinance, data_dir)
    → dict[str, {"current_price", "day_change_pct"}]

get_close(ohlcv, ticker, days)  → pd.Series | None
calc_return(series)             → float | None
normalize(series)               → pd.Series
"""
from __future__ import annotations

import pathlib
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Resolved once at import time — works on Windows (UNC or drive letter) and Linux.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# Always fetch at least 310 days so the 200-day MA always has data,
# regardless of the chosen display lookback period.
_MIN_FETCH_DAYS = 310


# ==============================================================================
# LOCAL FILE HELPERS  (ported from backtest_cli / app_fixed.py)
# ==============================================================================

def ticker_to_filename_stem(ticker: str) -> str:
    """Convert a ticker like '^VIX' or 'BN.TO' to a safe filename stem."""
    return ticker.replace("^", "_").replace("-", "_").replace(".", "_")


def find_local_file(ticker: str, data_dir: str) -> tuple[str | None, str | None]:
    """
    Search for a CSV or HDF5 file for `ticker` inside `data_dir`.
    Tries both the literal ticker and the sanitised stem.
    Checks the path as given (absolute) and relative to the script directory.
    Returns (path_str, format) where format is "csv" or "hdf5", or (None, None).
    """
    stem = ticker_to_filename_stem(ticker)
    p = pathlib.Path(data_dir)
    base_dirs: list[pathlib.Path] = (
        [p] if p.is_absolute() else [p, _SCRIPT_DIR / p]
    )
    for base_dir in base_dirs:
        for candidate in [stem, ticker]:
            for ext, fmt in [(".h5", "hdf5"), (".hdf5", "hdf5"), (".csv", "csv")]:
                path = base_dir / (candidate + ext)
                if path.is_file():
                    return str(path), fmt
    return None, None


def _try_hdf5_keys(path: str, ticker: str) -> pd.DataFrame | None:
    """Try common HDF5 key names; fall back to listing all keys via h5py."""
    stem = ticker_to_filename_stem(ticker)
    for key in [ticker, stem, f"/{ticker}", f"/{stem}", "data", "df", "ohlcv", "prices"]:
        try:
            df = pd.read_hdf(path, key=key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            continue
    try:
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
        for key in keys:
            try:
                df = pd.read_hdf(path, key=key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                continue
    except Exception:
        pass
    return None


def _normalise_local_df(
    df: pd.DataFrame,
    ticker: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame | None:
    """
    Standardise a raw local DataFrame into the same shape that yfinance produces:
      - DatetimeIndex
      - Columns: Close (adj), High, Low, Open (optional), Volume (optional)
      - adj_close preferred over raw close; High/Low adjusted by the same ratio
    Returns None when the result would be empty or is missing required columns.
    """
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Index → DatetimeIndex
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    # Date range filtering
    if start_date:
        df = df.loc[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df.loc[df.index <= pd.to_datetime(end_date)]
    if df.empty:
        return None

    # Prefer adj_close; adjust High/Low/Open by the same ratio so ATR etc. stay consistent
    if "adj_close" in df.columns:
        raw_close = df["close"].copy() if "close" in df.columns else df["adj_close"].copy()
        df["close"] = df["adj_close"]
        ratio = df["adj_close"] / raw_close.replace(0, np.nan)
        for col in ("high", "low", "open"):
            if col in df.columns:
                df[col] = df[col] * ratio
    elif "close" not in df.columns:
        return None

    required = {"close", "high", "low"}
    if not required.issubset(df.columns):
        return None

    keep = required | ({"open"} if "open" in df.columns else set()) | \
           ({"volume"} if "volume" in df.columns else set())
    return df[list(keep)]


def _load_ticker_local(
    ticker: str,
    data_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    """Load one ticker from disk; returns a normalised DataFrame or None."""
    path, fmt = find_local_file(ticker, data_dir)
    if path is None:
        return None
    df_raw = None
    if fmt == "csv":
        try:
            # Read with parsed dates so filtering in _normalise_local_df is fast.
            # Try to detect the date column; fall back to plain read if needed.
            df_raw = pd.read_csv(path, index_col=0, parse_dates=True)
            # If the index is not a DatetimeIndex after parse_dates, reset and
            # let _normalise_local_df handle it the normal way.
            if not isinstance(df_raw.index, pd.DatetimeIndex):
                df_raw = pd.read_csv(path)
        except Exception:
            try:
                df_raw = pd.read_csv(path)
            except Exception:
                return None
    else:
        # For HDF5, attempt a date-filtered read first (only works when the file
        # was written with data_columns=True); fall back to full read otherwise.
        path_str, _ = path, fmt
        stem = ticker_to_filename_stem(ticker)
        for key in [ticker, stem, f"/{ticker}", f"/{stem}", "data", "df", "ohlcv", "prices"]:
            try:
                if start_date and end_date:
                    try:
                        df_raw = pd.read_hdf(
                            path, key=key,
                            where=f'index >= "{start_date}" & index <= "{end_date}"',
                        )
                    except Exception:
                        df_raw = pd.read_hdf(path, key=key)
                else:
                    df_raw = pd.read_hdf(path, key=key)
                if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                    break
            except Exception:
                df_raw = None
                continue
        if df_raw is None:
            # Full fallback through _try_hdf5_keys
            df_raw = _try_hdf5_keys(path, ticker)
        if df_raw is None:
            return None
    return _normalise_local_df(df_raw, ticker, start_date, end_date)


def _local_to_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the normalised local DataFrame (lowercase columns) to the yfinance
    shape (Title-cased: Close, High, Low, Open, Volume) so the rest of the app
    can treat both sources identically.
    """
    rename = {"close": "Close", "high": "High", "low": "Low",
              "open": "Open", "volume": "Volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


# ==============================================================================
# YFINANCE HELPERS
# ==============================================================================

def _yf_squeeze(col) -> pd.Series:
    """Normalise yfinance's occasional MultiIndex-column result to a plain Series."""
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.squeeze()


# ==============================================================================
# PUBLIC API
# ==============================================================================

@st.cache_data(ttl=300)
def _fetch_single_ohlcv(
    ticker: str,
    start_str: str,
    end_str: str,
    use_yfinance: bool = True,
    data_dir: str = "data",
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single ticker (cached per-ticker).
    Returns a DataFrame or None if data is unavailable.
    """
    if use_yfinance:
        try:
            df = yf.download(
                ticker, start=start_str, end=end_str,
                progress=False, auto_adjust=True,
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            return df if not df.empty else None
        except Exception:
            return None
    else:
        df_local = _load_ticker_local(ticker, data_dir, start_str, end_str)
        if df_local is None:
            return None
        df_local = df_local.ffill().dropna(subset=["close", "high", "low"])
        return _local_to_ohlcv_frame(df_local)


def fetch_ohlcv(
    tickers: tuple[str, ...],
    display_days: int,
    use_yfinance: bool = True,
    data_dir: str = "data",
    progress_callback: callable | None = None,
    min_days: int = 310,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV history for every ticker, using either yfinance or local files.
    Calls progress_callback(done, total, ticker) after each ticker if provided.

    Parameters
    ----------
    tickers           : deduplicated tuple of ticker strings
    display_days      : desired lookback; actual fetch is at least min_days
    use_yfinance      : True → Yahoo Finance; False → local data_dir
    data_dir          : folder with per-ticker CSV / HDF5 files
    progress_callback : optional callable(done: int, total: int, ticker: str)
    min_days          : floor on the number of calendar days to fetch
                        (default 310 ensures MA-200 always has data;
                         pass a larger value when momentum needs more history)

    Returns
    -------
    dict[str, pd.DataFrame]  — OHLCV frames with columns Open, High, Low, Close, Volume
    """
    fetch_days = max(display_days + 10, min_days)
    end        = datetime.today()
    start      = end - timedelta(days=fetch_days)
    start_str  = start.strftime("%Y-%m-%d")
    end_str    = end.strftime("%Y-%m-%d")
    data: dict[str, pd.DataFrame] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        df = _fetch_single_ohlcv(ticker, start_str, end_str, use_yfinance, data_dir)
        if df is not None:
            data[ticker] = df
        if progress_callback:
            progress_callback(i, total, ticker)

    return data


@st.cache_data(ttl=300)
def _fetch_single_current_info(
    ticker: str,
    use_yfinance: bool = True,
    data_dir: str = "data",
) -> dict:
    """
    Fetch current price and day-change for one ticker (cached per-ticker).
    """
    if use_yfinance:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                cur  = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else cur
                return {
                    "current_price":  cur,
                    "day_change_pct": (cur - prev) / prev * 100 if prev else None,
                }
        except Exception:
            pass
        return {"current_price": None, "day_change_pct": None}
    else:
        df_local = _load_ticker_local(ticker, data_dir)
        if df_local is not None and len(df_local) >= 1:
            cur  = float(df_local["close"].iloc[-1])
            prev = float(df_local["close"].iloc[-2]) if len(df_local) >= 2 else cur
            return {
                "current_price":  cur,
                "day_change_pct": (cur - prev) / prev * 100 if prev else None,
            }
        return {"current_price": None, "day_change_pct": None}


def fetch_current_info(
    tickers: tuple[str, ...],
    use_yfinance: bool = True,
    data_dir: str = "data",
    progress_callback: callable | None = None,
) -> dict[str, dict]:
    """
    Fetch current price and 1-day change % for every ticker.
    Calls progress_callback(done, total, ticker) after each ticker if provided.

    In yfinance mode: pulls 5-day history from Yahoo Finance.
    In local mode:    reads the last two rows of the local file.

    Returns
    -------
    dict[str, {"current_price": float|None, "day_change_pct": float|None}]
    """
    info: dict[str, dict] = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        info[ticker] = _fetch_single_current_info(ticker, use_yfinance, data_dir)
        if progress_callback:
            progress_callback(i, total, ticker)
    return info


def current_info_from_ohlcv(
    ohlcv: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    Derive current_price and day_change_pct from the last two rows of each
    ticker's OHLCV data. Used in 'Last Close' mode to avoid a second round
    of live API calls — the data is already in the 310-day history.
    """
    info: dict[str, dict] = {}
    for ticker, df in ohlcv.items():
        if df is None or df.empty:
            info[ticker] = {"current_price": None, "day_change_pct": None}
            continue
        cur  = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else cur
        info[ticker] = {
            "current_price":  cur,
            "day_change_pct": (cur - prev) / prev * 100 if prev else None,
        }
    return info


def list_local_files(data_dir: str) -> list[str]:
    """
    Return a sorted list of ticker names found in `data_dir` based on filename stems.
    Used by the sidebar to show what's available in local mode.
    """
    p = pathlib.Path(data_dir)
    if not p.is_absolute():
        # try both relative and relative-to-script
        candidates = [p, _SCRIPT_DIR / p]
    else:
        candidates = [p]

    found = []
    for base in candidates:
        if base.is_dir():
            for ext in ("*.csv", "*.h5", "*.hdf5"):
                for f in sorted(base.glob(ext)):
                    stem = f.stem
                    # Reverse the sanitisation to recover the ticker
                    ticker = stem.replace("_", "^", 1) if stem.startswith("_") else stem
                    found.append(ticker)
            break  # stop at first valid directory

    return sorted(set(found))


# ==============================================================================
# UTILITY HELPERS
# ==============================================================================

def get_close(
    ohlcv: dict[str, pd.DataFrame],
    ticker: str,
    days: int | None = None,
) -> pd.Series | None:
    """
    Return the Close series for `ticker`, optionally trimmed to the last `days` days.
    Returns None when fewer than 2 data points remain.
    """
    df = ohlcv.get(ticker)
    if df is None or df.empty:
        return None
    s = df["Close"]
    if days:
        cutoff = pd.Timestamp(datetime.today() - timedelta(days=days))
        s = s[s.index >= cutoff]
    return s if len(s) >= 2 else None


def calc_return(series: pd.Series | None) -> float | None:
    """Percentage change from the first to the last value of a series."""
    if series is None or len(series) < 2:
        return None
    return float((series.iloc[-1] / series.iloc[0] - 1) * 100)


def normalize(series: pd.Series) -> pd.Series:
    """Re-index a series to 100 at its first point."""
    return series / series.iloc[0] * 100
