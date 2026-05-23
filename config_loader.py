"""
config_loader.py
────────────────
Handles loading, parsing, and validating portfolio_config.json.

Override flags
──────────────
Two optional flags in the config file control whether the sidebar can
replace the portfolio holdings or watchlist with user-typed JSON:

  "portfolio_override": false   (or true)
  "watchlist_override": false   (or true)

When a flag is true, a JSON text-area appears in the sidebar.

Accepted formats for the portfolio text area
─────────────────────────────────────────────
1. Ticker array (auto-expands to minimal holding entries):
       ["AAPL", "MSFT", "GOOGL"]

2. Holdings array (full or partial fields):
       [
         {"ticker": "AAPL", "name": "Apple Inc.", "shares": 50, "avg_cost": 165.00, "sector": "Technology"},
         {"ticker": "MSFT", "name": "Microsoft Corp.", "shares": 30, "avg_cost": 310.00, "sector": "Technology"}
       ]

3. Object with a "holdings" key:
       {"holdings": [ ... ]}

4. Full portfolio block (name/benchmark are preserved from the base config):
       {
         "portfolio": {
           "name": "My Portfolio",
           "benchmark": "^GSPC",
           "benchmark_name": "S&P 500",
           "holdings": [ ... ]
         }
       }

Accepted formats for the watchlist text area
─────────────────────────────────────────────
1. Ticker array (auto-expands to minimal watchlist entries):
       ["NVDA", "META", "AVGO"]

2. Watchlist array (full or partial fields):
       [
         {"ticker": "NVDA", "name": "NVIDIA Corp.", "reason": "AI leader", "sector": "Technology"},
         {"ticker": "META", "name": "Meta Platforms", "reason": "Ad recovery", "sector": "Communication Services"}
       ]

3. Object with a "watchlist" key:
       {"watchlist": [ ... ]}

Missing required fields are filled with safe defaults automatically.
"""
import json
from pathlib import Path

# ── Required structure ────────────────────────────────────────────────────────
_REQUIRED_TOP = {
    "portfolio": ["name", "benchmark", "benchmark_name", "holdings"],
    "watchlist": None,
    "settings":  ["lookback_periods", "default_lookback_days",
                  "underperformance_threshold_pct"],
}
_REQUIRED_HOLDING   = ["ticker", "name", "shares", "avg_cost"]
_REQUIRED_WATCHLIST = ["ticker", "name", "reason"]


def load_config(path: str) -> dict:
    """Load config from a file path."""
    with open(path) as f:
        return json.load(f)


def load_config_bytes(raw: bytes) -> dict:
    """Load config from raw bytes (uploaded file)."""
    return json.loads(raw)


def validate_config(cfg: dict) -> list[str]:
    """
    Validate config structure.
    Returns a list of human-readable error strings (empty = valid).
    """
    errors = []

    for top_key, sub_keys in _REQUIRED_TOP.items():
        if top_key not in cfg:
            errors.append(f"Missing top-level key: '{top_key}'")
            continue
        if sub_keys:
            for sk in sub_keys:
                if sk not in cfg[top_key]:
                    errors.append(f"'{top_key}' is missing field '{sk}'")

    if "portfolio" in cfg and "holdings" in cfg["portfolio"]:
        for i, h in enumerate(cfg["portfolio"]["holdings"]):
            for k in _REQUIRED_HOLDING:
                if k not in h:
                    errors.append(
                        f"holdings[{i}] ('{h.get('ticker', '?')}') missing '{k}'"
                    )

    if "watchlist" in cfg:
        for i, w in enumerate(cfg["watchlist"]):
            for k in _REQUIRED_WATCHLIST:
                if k not in w:
                    errors.append(
                        f"watchlist[{i}] ('{w.get('ticker', '?')}') missing '{k}'"
                    )

    return errors


def resolve_config(
    uploaded_file, default_path: Path
) -> tuple[dict | None, str | None, str | None]:
    """
    Resolve config from either an uploaded file or the default path.

    Returns (cfg, source_label, error_message).
    cfg is None when nothing could be loaded.
    """
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            cfg = load_config_bytes(uploaded_file.read())
            return cfg, f"📄 {uploaded_file.name}", None
        except json.JSONDecodeError as e:
            return None, None, f"Invalid JSON: {e}"

    if default_path.exists():
        cfg = load_config(str(default_path))
        return cfg, "📁 portfolio_config.json (default)", None

    return None, None, None


# ==============================================================================
# DEFAULT ENTRY BUILDERS
# ==============================================================================

def _make_holding(ticker: str) -> dict:
    """Minimal holding entry when only a ticker is supplied."""
    return {
        "ticker":   ticker.strip().upper(),
        "name":     ticker.strip().upper(),
        "shares":   1,
        "avg_cost": 0.0,
        "sector":   "Unknown",
    }


def _fill_holding_defaults(entry: dict) -> dict:
    """Fill any missing required fields with safe defaults."""
    entry = dict(entry)
    entry.setdefault("ticker",   entry.get("ticker", "?").strip().upper())
    entry.setdefault("name",     entry["ticker"])
    entry.setdefault("shares",   1)
    entry.setdefault("avg_cost", 0.0)
    entry.setdefault("sector",   "Unknown")
    return entry


def _make_watchlist_entry(ticker: str) -> dict:
    """Minimal watchlist entry when only a ticker is supplied."""
    return {
        "ticker": ticker.strip().upper(),
        "name":   ticker.strip().upper(),
        "reason": "",
        "sector": "",
    }


def _fill_watchlist_defaults(entry: dict) -> dict:
    """Fill any missing required fields with safe defaults."""
    entry = dict(entry)
    entry.setdefault("ticker", entry.get("ticker", "?").strip().upper())
    entry.setdefault("name",   entry["ticker"])
    entry.setdefault("reason", "")
    entry.setdefault("sector", "")
    return entry


# ==============================================================================
# OVERRIDE PARSERS
# ==============================================================================

def parse_portfolio_override(
    raw: str,
) -> tuple[list[dict] | None, str | None]:
    """
    Parse the portfolio override text area content.

    Returns (holdings_list, error_message).
    holdings_list is None when the input is empty or unparseable.

    Accepted formats (see module docstring for examples):
      - JSON array of ticker strings → expanded to minimal holding entries
      - JSON array of holding objects → missing fields filled with defaults
      - JSON object with "holdings" key
      - JSON object with "portfolio" key (full portfolio block)
    """
    raw = raw.strip()
    if not raw:
        return None, None

    # ── Try to parse as JSON ──────────────────────────────────────────────────
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Last-chance attempt: wrap a bare array-of-strings like AAPL,MSFT
        # that the user may have typed without brackets.
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if tickers:
            return [_make_holding(t) for t in tickers], None
        return None, f"Invalid JSON — could not parse portfolio override."

    # ── JSON array ────────────────────────────────────────────────────────────
    if isinstance(parsed, list):
        if not parsed:
            return None, None
        # Array of plain strings → minimal entries
        if all(isinstance(item, str) for item in parsed):
            return [_make_holding(t) for t in parsed], None
        # Array of dicts → fill defaults
        if all(isinstance(item, dict) for item in parsed):
            holdings = []
            for i, item in enumerate(parsed):
                if "ticker" not in item:
                    return None, f"holdings[{i}] is missing 'ticker'."
                holdings.append(_fill_holding_defaults(item))
            return holdings, None
        return None, "Portfolio array must contain either strings or objects."

    # ── JSON object ──────────────────────────────────────────────────────────
    if isinstance(parsed, dict):
        # {"portfolio": {"holdings": [...]}}
        if "portfolio" in parsed:
            port = parsed["portfolio"]
            if not isinstance(port, dict) or "holdings" not in port:
                return None, "'portfolio' block must contain a 'holdings' array."
            raw_holdings = port["holdings"]
        # {"holdings": [...]}
        elif "holdings" in parsed:
            raw_holdings = parsed["holdings"]
        else:
            return None, (
                "Object must have a 'holdings' key or a 'portfolio' key. "
                "Alternatively, supply a JSON array of tickers or holding objects."
            )

        if not isinstance(raw_holdings, list):
            return None, "'holdings' must be a JSON array."

        holdings = []
        for i, item in enumerate(raw_holdings):
            if isinstance(item, str):
                holdings.append(_make_holding(item))
            elif isinstance(item, dict):
                if "ticker" not in item:
                    return None, f"holdings[{i}] is missing 'ticker'."
                holdings.append(_fill_holding_defaults(item))
            else:
                return None, f"holdings[{i}] must be a string or object."
        return (holdings or None), None

    return None, "Unrecognised portfolio override format."


def parse_watchlist_override(
    raw: str,
) -> tuple[list[dict] | None, str | None]:
    """
    Parse the watchlist override text area content.

    Returns (watchlist_list, error_message).
    watchlist_list is None when the input is empty or unparseable.

    Accepted formats (see module docstring for examples):
      - JSON array of ticker strings → expanded to minimal watchlist entries
      - JSON array of watchlist objects → missing fields filled with defaults
      - JSON object with "watchlist" key
    """
    raw = raw.strip()
    if not raw:
        return None, None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if tickers:
            return [_make_watchlist_entry(t) for t in tickers], None
        return None, "Invalid JSON — could not parse watchlist override."

    # ── JSON array ────────────────────────────────────────────────────────────
    if isinstance(parsed, list):
        if not parsed:
            return None, None
        if all(isinstance(item, str) for item in parsed):
            return [_make_watchlist_entry(t) for t in parsed], None
        if all(isinstance(item, dict) for item in parsed):
            entries = []
            for i, item in enumerate(parsed):
                if "ticker" not in item:
                    return None, f"watchlist[{i}] is missing 'ticker'."
                entries.append(_fill_watchlist_defaults(item))
            return entries, None
        return None, "Watchlist array must contain either strings or objects."

    # ── JSON object ──────────────────────────────────────────────────────────
    if isinstance(parsed, dict):
        if "watchlist" not in parsed:
            return None, (
                "Object must have a 'watchlist' key. "
                "Alternatively, supply a JSON array of tickers or watchlist objects."
            )
        raw_wl = parsed["watchlist"]
        if not isinstance(raw_wl, list):
            return None, "'watchlist' must be a JSON array."

        entries = []
        for i, item in enumerate(raw_wl):
            if isinstance(item, str):
                entries.append(_make_watchlist_entry(item))
            elif isinstance(item, dict):
                if "ticker" not in item:
                    return None, f"watchlist[{i}] is missing 'ticker'."
                entries.append(_fill_watchlist_defaults(item))
            else:
                return None, f"watchlist[{i}] must be a string or object."
        return (entries or None), None

    return None, "Unrecognised watchlist override format."


# ==============================================================================
# APPLY OVERRIDES
# ==============================================================================

def apply_overrides(
    cfg: dict,
    portfolio_holdings: list[dict] | None,
    watchlist_entries:  list[dict] | None,
) -> dict:
    """
    Return a shallow-copy of `cfg` with holdings and/or watchlist replaced
    by the parsed override lists.

    Parameters
    ──────────
    cfg                : validated config dict
    portfolio_holdings : list of holding dicts, or None (no portfolio override)
    watchlist_entries  : list of watchlist dicts, or None (no watchlist override)
    """
    cfg = dict(cfg)

    if portfolio_holdings:
        cfg["portfolio"] = dict(cfg["portfolio"])
        cfg["portfolio"]["holdings"] = portfolio_holdings

    if watchlist_entries is not None:
        cfg["watchlist"] = watchlist_entries

    return cfg
