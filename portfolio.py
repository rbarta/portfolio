"""
app.py
------
Portfolio Sentinel - main Streamlit entry point.
Tabs: Performance | Flags & Actions | Technicals | Watchlist | Analytics | Momentum | Config
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

from config_loader import (resolve_config, validate_config,
                            apply_overrides,
                            parse_portfolio_override, parse_watchlist_override)
from data_loader   import (fetch_ohlcv, fetch_current_info,
                            get_close, calc_return, normalize,
                            list_local_files, current_info_from_ohlcv)
from indicators    import (compute_all_indicators, registry_labels,
                            registry_descriptions)
from cards         import (flag_card_html, holdings_glance_card_html,
                            watchlist_card_html, config_ok_banner,
                            config_error_banner, config_missing_banner)
from momentum      import (get_momentum_cfg, fetch_vix,
                            fetch_momentum_prices, fetch_momentum_hl,
                            score_candidates)

st.set_page_config(
    page_title="Portfolio Sentinel", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  html,body,[class*="css"]{ font-family:'Syne',sans-serif; background:#0d0f14; color:#e2e8f0; }
  .stApp{ background:#0d0f14; }
  section[data-testid="stSidebar"]{ background:#111318; border-right:1px solid #1e2330; }
  section[data-testid="stSidebar"] *{ color:#c9d1e0 !important; }
  .sentinel-title{ font-size:2.8rem; font-weight:800; letter-spacing:-1px;
    background:linear-gradient(135deg,#f0f4ff 0%,#7c9aff 50%,#4c6ef5 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0; }
  .sentinel-sub{ font-family:'DM Mono',monospace; font-size:0.75rem; color:#4c6ef5;
    letter-spacing:3px; text-transform:uppercase; margin-bottom:1.5rem; }
  .positive{ color:#4ade80; } .negative{ color:#f87171; } .neutral{ color:#94a3b8; }
  .ticker-badge{ font-family:'DM Mono',monospace; font-size:0.85rem; font-weight:500;
    background:#1e2330; padding:3px 9px; border-radius:5px; color:#a5b4fc; }
  .ind-badge{ display:inline-block; font-family:'DM Mono',monospace; font-size:0.68rem;
    padding:2px 7px; border-radius:4px; margin-right:4px; margin-top:3px; }
  .ind-bull{ background:#0f2d1a; border:1px solid #166534; color:#4ade80; }
  .ind-bear{ background:#2d1515; border:1px solid #7f1d1d; color:#f87171; }
  .ind-neut{ background:#1a1f2e; border:1px solid #2d3748; color:#94a3b8; }
  .ind-warn{ background:#2b2007; border:1px solid #78350f; color:#fbbf24; }
  .section-header{ font-family:'DM Mono',monospace; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:3px; color:#4c6ef5; padding-bottom:0.5rem; border-bottom:1px solid #1e2330;
    margin-bottom:1rem; margin-top:1.5rem; }
  .alert-red   { background:#2d1515; border:1px solid #7f1d1d; border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.6rem; }
  .alert-yellow{ background:#2b2007; border:1px solid #78350f; border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.6rem; }
  .alert-green { background:#0f2d1a; border:1px solid #14532d; border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.6rem; }
  .alert-blue  { background:#111c3b; border:1px solid #1e3a8a; border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.6rem; }
  .score-card       { display:inline-block; font-family:'DM Mono',monospace; font-size:0.85rem; font-weight:700; padding:3px 10px; border-radius:6px; margin-left:8px; }
  .score-strong-sell{ background:#450a0a; border:1px solid #dc2626; color:#f87171; }
  .score-sell       { background:#2d1515; border:1px solid #b91c1c; color:#fca5a5; }
  .score-neutral    { background:#1a1f2e; border:1px solid #374151; color:#94a3b8; }
  .score-buy        { background:#0f2d1a; border:1px solid #15803d; color:#86efac; }
  .score-strong-buy { background:#052e16; border:1px solid #16a34a; color:#4ade80; }
  div[data-testid="stMetric"]{ background:#13161e; border:1px solid #1e2330; border-radius:12px; padding:1rem; }
  div[data-testid="stMetricValue"]{ color:#e2e8f0 !important; font-family:'Syne',sans-serif; font-weight:700; }
  div[data-testid="stMetricLabel"]{ color:#5c667a !important; font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; }
  div[data-testid="stMetricDelta"] svg{ display:none; }
  .stSelectbox>div>div{ background:#13161e; border:1px solid #1e2330; color:#e2e8f0; }
  button[data-baseweb="tab"]{ font-family:'Syne',sans-serif; font-weight:600; color:#5c667a; }
  button[data-baseweb="tab"][aria-selected="true"]{ color:#7c9aff; border-bottom-color:#4c6ef5 !important; }
  ::-webkit-scrollbar{ width:6px; } ::-webkit-scrollbar-track{ background:#0d0f14; }
  ::-webkit-scrollbar-thumb{ background:#1e2330; border-radius:3px; }
  hr{ border-color:#1e2330; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0f14",
    font=dict(family="DM Mono", color="#94a3b8", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hovermode="x unified",
    xaxis=dict(gridcolor="#1e2330", showgrid=True),
    yaxis=dict(gridcolor="#1e2330", showgrid=True),
    margin=dict(l=0, r=0, t=10, b=0),
)

def fmt_pct(v, d=2):
    if v is None: return "N/A"
    return f"+{v:.{d}f}%" if v >= 0 else f"{v:.{d}f}%"

def color_cls(v):
    if v is None: return "neutral"
    return "positive" if v >= 0 else "negative"

def score_label(s: int) -> tuple[str, str]:
    if s <= -3: return "score-strong-sell", "STRONG SELL"
    if s == -2: return "score-sell",         "SELL"
    if s == -1: return "score-sell",         "LEAN SELL"
    if s ==  0: return "score-neutral",      "NEUTRAL"
    if s ==  1: return "score-buy",          "LEAN BUY"
    if s ==  2: return "score-buy",          "BUY"
    return           "score-strong-buy",     "STRONG BUY"

import re as _re

def _cfg_convert_symbol(raw: str) -> str:
    s = raw.strip().upper()
    if s.endswith("-T"):
        return s[:-2] + ".TO"
    elif "-" in s:
        return s.split("-")[0]
    return s

def _cfg_parse_price(raw: str) -> float:
    return float(raw.strip().lstrip("$").replace(",", ""))

def _cfg_split_delimited(line: str) -> list[str]:
    if "\t" in line:
        return [p.strip() for p in line.split("\t")]
    if "," in line and not line.strip().startswith("{"):
        return [p.strip() for p in line.split(",")]
    return [p.strip() for p in _re.split(r"\s{2,}", line)]

def _cfg_parse_brokerage(text: str) -> tuple[list[dict], list[str]]:
    KNOWN_HEADERS = {"account","sec","asset","date add","price","qty","date","quantity","shares","symbol","ticker"}
    warnings: list[str] = []
    holdings: list[dict] = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return [], []
    if lines[0].strip().lower() in KNOWN_HEADERS:
        data_lines = lines[6:]
        for i in range(0, len(data_lines) - 5, 6):
            chunk = [data_lines[i + j].strip() for j in range(6)]
            _account, sec, asset, _date, price_raw, qty_raw = chunk
            try:
                holdings.append({
                    "ticker":   _cfg_convert_symbol(asset),
                    "name":     _cfg_convert_symbol(asset),
                    "shares":   int(float(qty_raw.replace(",", ""))),
                    "avg_cost": round(_cfg_parse_price(price_raw), 4),
                    "sector":   sec,
                })
            except Exception as exc:
                warnings.append(f"Row {i // 6 + 1} skipped ({asset!r}): {exc}")
        return holdings, warnings
    rows = [_cfg_split_delimited(ln) for ln in lines if ln.strip()]
    rows = [r for r in rows if r]
    if not rows:
        return [], ["No parseable rows found."]
    start = 0
    first = [c.lower() for c in rows[0]]
    col = {"sec": None, "asset": None, "price": None, "qty": None}
    for idx, cell in enumerate(first):
        if cell in ("sec", "sector"):               col["sec"]   = idx
        elif cell in ("asset", "symbol", "ticker"): col["asset"] = idx
        elif "price" in cell or "avg" in cell:      col["price"] = idx
        elif cell in ("qty", "quantity", "shares"): col["qty"]   = idx
    if any(v is not None for v in col.values()):
        start = 1
    else:
        col = {"sec": 1, "asset": 2, "price": 4, "qty": 5}
    for row_num, row in enumerate(rows[start:], start=1):
        try:
            sec       = row[col["sec"]]   if col["sec"]   is not None and col["sec"]   < len(row) else ""
            asset     = row[col["asset"]] if col["asset"] is not None and col["asset"] < len(row) else ""
            price_raw = row[col["price"]] if col["price"] is not None and col["price"] < len(row) else ""
            qty_raw   = row[col["qty"]]   if col["qty"]   is not None and col["qty"]   < len(row) else ""
            if not asset or not price_raw or not qty_raw:
                continue
            holdings.append({
                "ticker":   _cfg_convert_symbol(asset),
                "name":     _cfg_convert_symbol(asset),
                "shares":   int(float(qty_raw.replace(",", ""))),
                "avg_cost": round(_cfg_parse_price(price_raw), 4),
                "sector":   sec,
            })
        except Exception as exc:
            warnings.append(f"Row {row_num} skipped: {exc}")
    return holdings, warnings

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### Configuration")

    st.markdown(
        '<div style=\'font-family:"DM Mono",monospace;font-size:0.7rem;color:#5c667a;'
        'margin-bottom:0.4rem;letter-spacing:1px;text-transform:uppercase\'>Portfolio Config</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        label="Drop portfolio_config.json here",
        type=["json"],
        help="Drag & drop your portfolio_config.json, or click to browse",
        label_visibility="collapsed",
    )

    # -- Scenario selector ---------------------------------------------------
    _SCENARIOS = [
        "All — Current Data",
        "All — Last Close",
        "Portfolio — Current Data",
        "Portfolio — Last Close",
        "Watchlist — Current Data",
        "Watchlist — Last Close",
    ]
    _scenario = st.selectbox(
        "Refresh scope",
        _SCENARIOS,
        key="fetch_scenario",
        help=(
            "Current Data — fetches live prices, clears cache.\n"
            "Last Close — uses cached / previously stored closing prices, no extra API call.\n"
            "All / Portfolio / Watchlist — limits which tickers are fetched."
        ),
    )
    _live  = "Current Data" in _scenario
    _scope = ("all"       if _scenario.startswith("All")
              else "portfolio" if _scenario.startswith("Portfolio")
              else "watchlist")

    _rb_col, _cb_col = st.columns([1, 1])
    with _rb_col:
        _manual_refresh = st.button("Refresh Data", use_container_width=True)
    with _cb_col:
        _auto_load = st.checkbox(
            "Auto-load", value=False, key="auto_load",
            help="Automatically fetch data on config load.",
        )

    import hashlib as _hl
    _cfg_raw, _, _ = resolve_config(
        uploaded_file, Path(__file__).parent / "portfolio_config.json"
    )
    _cfg_hash = _hl.md5(
        str(sorted(str(_cfg_raw).split())).encode()
    ).hexdigest() if _cfg_raw else ""

    if st.session_state.get("_last_cfg_hash") != _cfg_hash:
        st.session_state["_last_cfg_hash"] = _cfg_hash
        st.session_state["data_loaded"]         = False
        st.session_state["watchlist_processed"] = False
        st.session_state["momentum_processed"]  = False

    if _manual_refresh:
        if _live:
            st.cache_data.clear()
        st.session_state["data_loaded"]         = True
        st.session_state["watchlist_processed"] = False
        st.session_state["momentum_processed"]  = False

    if _auto_load:
        st.session_state["data_loaded"] = True

    cfg, config_source, load_err = resolve_config(
        uploaded_file,
        Path(__file__).parent / "portfolio_config.json",
    )

    # Config Editor session override
    if uploaded_file is None and st.session_state.get("cfg_session_override") is not None:
        cfg           = st.session_state["cfg_session_override"]
        config_source = "Config Editor (session)"
        load_err      = None
    elif uploaded_file is not None:
        st.session_state.pop("cfg_session_override", None)

    if load_err:
        st.error(load_err)
        st.stop()
    if cfg is None:
        st.markdown(config_missing_banner(), unsafe_allow_html=True)
        st.stop()

    errors = validate_config(cfg)
    if errors:
        st.markdown(config_error_banner(errors), unsafe_allow_html=True)
        st.stop()

    st.markdown(config_ok_banner(config_source), unsafe_allow_html=True)

    settings  = cfg["settings"]
    portfolio = cfg["portfolio"]
    watchlist = cfg["watchlist"]

    period_opts = {f"{d}D": d for d in settings["lookback_periods"]}
    sel_period  = st.selectbox(
        "Lookback period", list(period_opts.keys()),
        index=list(period_opts.values()).index(settings["default_lookback_days"]),
    )
    lookback_days = period_opts[sel_period]

    threshold = st.slider(
        "Underperformance threshold (vs benchmark, %)",
        min_value=-20, max_value=0,
        value=int(settings["underperformance_threshold_pct"]), step=1,
    )

    st.markdown("---")
    st.markdown(f"**Portfolio:** {portfolio['name']}")
    st.markdown(f"**Benchmark:** {portfolio['benchmark_name']}")
    st.markdown(f"**Holdings:** {len(portfolio['holdings'])} / **Watchlist:** {len(watchlist)}")

    st.markdown("---")
    st.markdown("**Data Source**")
    use_yfinance = st.checkbox(
        "Download from yfinance", value=True, key="use_yfinance",
        help="Uncheck to load from local CSV/HDF5 files instead.",
    )
    data_dir = st.text_input(
        "Local data directory", value="data", key="data_dir",
        help="Folder with per-ticker CSV or HDF5 files.",
        disabled=use_yfinance,
    )
    if not use_yfinance:
        available = list_local_files(data_dir)
        if available:
            st.caption(f"{len(available)} file(s) found in {data_dir}")
        else:
            st.warning(f"No files found in {data_dir}.")

    st.markdown("---")
    st.markdown("**Flag Card Badges**")
    st.caption("Choose which indicator pills appear on flag cards")
    all_ind_labels = registry_labels()
    enabled_badges: set[str] = set()
    for key, label in all_ind_labels.items():
        if st.checkbox(label, value=True, key=f"badge_{key}"):
            enabled_badges.add(key)

    st.markdown("---")
    with st.expander("Indicator Legend", expanded=False):
        for key, desc in registry_descriptions().items():
            st.markdown(
                f'<div style=\'font-family:"DM Mono",monospace;font-size:0.68rem;'
                f'color:#5c667a;margin-bottom:4px\'>'
                f'<b style="color:#7c9aff">{all_ind_labels[key]}</b><br>{desc}</div>',
                unsafe_allow_html=True,
            )

# ==============================================================================
# DATA LOADING
# ==============================================================================
if not st.session_state.get("data_loaded", False):
    st.markdown('<div class="sentinel-title">Portfolio Sentinel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sentinel-sub">Performance · Technicals · Rebalancing</div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.info(
        "**Config loaded.** Press **Refresh Data** in the sidebar to fetch market data, "
        "or enable **Auto-load**.",
        icon="ℹ️",
    )
    st.stop()

# Compute how many days of history are needed.
# Indicators (MA-200) need at least 310 calendar days.
# Momentum needs: lookback_periods * days_per_period + corr_lookback + buffer.
_mcfg_pre    = get_momentum_cfg(cfg)
_mom_cal_days = int(
    _mcfg_pre["mom_lookback"] * (35 if _mcfg_pre["use_monthly"] else 9)
    + _mcfg_pre["corr_lb"] * 2 + 60
)
_min_fetch_days = max(310, _mom_cal_days)

# Determine which tickers to fetch based on scope.
# Add VIX to the fetch list when the momentum VIX-rank gate is enabled so that
# momentum scoring can use ohlcv_data directly without a separate API call.
_vix_ticker = "^VIX"
_need_vix   = bool(_mcfg_pre.get("use_vix_rank", False))
_port_tickers  = [h["ticker"] for h in portfolio["holdings"]]
_wl_tickers    = [w["ticker"] for w in watchlist]
_bench_tickers = [portfolio["benchmark"]]

if _scope == "portfolio":
    fetch_tickers = tuple(dict.fromkeys(_port_tickers + _bench_tickers))
elif _scope == "watchlist":
    fetch_tickers = tuple(dict.fromkeys(_wl_tickers + _bench_tickers))
else:
    fetch_tickers = tuple(dict.fromkeys(_port_tickers + _wl_tickers + _bench_tickers))

# Always include VIX when the momentum gate needs it (small overhead; no High/Low needed)
if _need_vix and _vix_ticker not in fetch_tickers:
    fetch_tickers = fetch_tickers + (_vix_ticker,)

# all_tickers = full universe for Data Status table
all_tickers = tuple(dict.fromkeys(_port_tickers + _wl_tickers + _bench_tickers))

with st.spinner("Fetching market data..."):
    _prog_bar  = st.progress(0.0)
    _prog_text = st.empty()
    _total_t   = len(fetch_tickers)

    def _ohlcv_progress(done: int, total: int, ticker: str) -> None:
        _prog_bar.progress(done / total)
        _prog_text.caption(f"Fetching prices: **{ticker}** ({done} of {total})")

    ohlcv_data = fetch_ohlcv(fetch_tickers, lookback_days, use_yfinance, data_dir,
                              progress_callback=_ohlcv_progress,
                              min_days=_min_fetch_days)

    if _live:
        _prog_bar.progress(0.0)
        _prog_text.caption(f"Fetching current prices — 0 of {_total_t}")

        def _info_progress(done: int, total: int, ticker: str) -> None:
            _prog_bar.progress(done / total)
            _prog_text.caption(f"Fetching current price: **{ticker}** ({done} of {total})")

        current_info = fetch_current_info(fetch_tickers, use_yfinance, data_dir,
                                          progress_callback=_info_progress)
    else:
        current_info = current_info_from_ohlcv(ohlcv_data)

    _prog_bar.empty()
    _prog_text.empty()

bench_ticker = portfolio["benchmark"]
bench_full   = get_close(ohlcv_data, bench_ticker)
bench_period = get_close(ohlcv_data, bench_ticker, lookback_days)
bench_return = calc_return(bench_period)

# ==============================================================================
# INDICATORS
# ==============================================================================
indicators: dict[str, dict] = {}
for item in portfolio["holdings"] + watchlist:
    t       = item["ticker"]
    df      = ohlcv_data.get(t)
    day_chg = (current_info.get(t) or {}).get("day_change_pct")
    indicators[t] = (
        compute_all_indicators(df, bench_full, day_chg)
        if df is not None and not df.empty else {}
    )

# ==============================================================================
# PORTFOLIO METRICS
# ==============================================================================
portfolio_values = []
for h in portfolio["holdings"]:
    t    = h["ticker"]
    info = current_info.get(t, {})
    cp   = info.get("current_price")
    if cp:
        per_ret = calc_return(get_close(ohlcv_data, t, lookback_days))
        tot_ret = (cp - h["avg_cost"]) / h["avg_cost"] * 100
        portfolio_values.append({
            "ticker": t, "name": h["name"], "sector": h.get("sector", "-"),
            "shares": h["shares"], "avg_cost": h["avg_cost"],
            "current_val":   cp * h["shares"],
            "cost_basis":    h["avg_cost"] * h["shares"],
            "period_return": per_ret,
            "total_return":  tot_ret,
        })

total_port_val = sum(p["current_val"] for p in portfolio_values)
total_cost     = sum(p["cost_basis"]  for p in portfolio_values)
total_gain_pct = (total_port_val - total_cost) / total_cost * 100 if total_cost else 0
valid_rets     = [p["period_return"] for p in portfolio_values if p["period_return"] is not None]
avg_port_ret   = float(np.mean(valid_rets)) if valid_rets else None

underperformers = [
    pv["ticker"] for pv in portfolio_values
    if (pv["period_return"] is not None and bench_return is not None
        and pv["period_return"] - bench_return < threshold)
]
bearish_count = sum(
    1 for h in portfolio["holdings"]
    if indicators.get(h["ticker"], {}).get("composite", 0) < -1
)

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<div class="sentinel-title">Portfolio Sentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="sentinel-sub">Performance · Technicals · Rebalancing</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
diff_vs_bench = (avg_port_ret - bench_return) if (avg_port_ret and bench_return) else None
with c1: st.metric("Portfolio Value",     f"${total_port_val:,.0f}", delta=f"{total_gain_pct:+.2f}% total")
with c2: st.metric(portfolio["benchmark_name"], fmt_pct(bench_return), delta=f"{lookback_days}D period")
with c3: st.metric("Avg Portfolio Ret",   fmt_pct(avg_port_ret),     delta=f"{fmt_pct(diff_vs_bench)} vs BM")
with c4: st.metric("Underperformers",     str(len(underperformers)),  delta=f"of {len(portfolio['holdings'])} holdings")
with c5: st.metric("Bearish Tech Signal", str(bearish_count),         delta="composite score < -1")

# Scope / mode strip
_scope_color = {"all": "#0f2d1a", "portfolio": "#111c3b", "watchlist": "#2b2007"}
_scope_label = {"all": "All tickers", "portfolio": "Portfolio only", "watchlist": "Watchlist only"}
_mode_label  = "live prices" if _live else "last close"
st.markdown(
    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
    f'background:{_scope_color[_scope]};border-radius:6px;padding:5px 12px;'
    f'color:#94a3b8;margin-bottom:0.5rem">'
    f'Loaded: <b style="color:#e2e8f0">{_scope_label[_scope]}</b> &nbsp;&middot;&nbsp; '
    f'{_mode_label} &nbsp;&middot;&nbsp; {len(fetch_tickers)} tickers fetched</div>',
    unsafe_allow_html=True,
)
st.markdown("")

# ==============================================================================
# TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Performance", "Flags & Actions", "Technicals",
    "Watchlist", "Analytics", "Momentum", "Config",
])

# ------------------------------------------------------------------------------
# TAB 1 — PERFORMANCE
# ------------------------------------------------------------------------------
with tab1:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-header">Normalised Performance vs Benchmark</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        if bench_period is not None:
            fig.add_trace(go.Scatter(
                x=bench_period.index, y=normalize(bench_period),
                name=portfolio["benchmark_name"],
                line=dict(color="#4c6ef5", width=2.5, dash="dot"),
            ))
        cscale = px.colors.qualitative.Plotly
        for i, h in enumerate(portfolio["holdings"]):
            s = get_close(ohlcv_data, h["ticker"], lookback_days)
            if s is None: continue
            ret  = calc_return(s)
            diff = (ret - bench_return) if (ret and bench_return) else None
            is_under = diff is not None and diff < threshold
            fig.add_trace(go.Scatter(
                x=s.index, y=normalize(s), name=h["ticker"],
                line=dict(
                    color="#f87171" if is_under else cscale[i % len(cscale)],
                    width=2.2 if is_under else 1.5,
                ),
                opacity=1.0 if is_under else 0.7,
                hovertemplate=f"%{{y:.1f}}<extra>{h['ticker']}</extra>",
            ))
        fig.update_layout(**CHART_LAYOUT, height=420, yaxis_title="Indexed (base=100)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Holdings — Sorted by Period Return</div>',
                    unsafe_allow_html=True)
        for pv in sorted(portfolio_values, key=lambda x: (x["period_return"] or -999)):
            st.markdown(
                holdings_glance_card_html(
                    ticker=pv["ticker"],
                    period_return=pv["period_return"],
                    bench_return=bench_return,
                    threshold=threshold,
                    comp_score=indicators.get(pv["ticker"], {}).get("composite", 0),
                ),
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------------------
# TAB 2 — FLAGS & ACTIONS
# ------------------------------------------------------------------------------
with tab2:
    col_a, col_b = st.columns(2)

    red_flags, yellow_flags, green_flags = [], [], []
    for pv in portfolio_values:
        ret  = pv["period_return"]
        diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
        if diff is None: continue
        if diff < threshold:  red_flags.append(pv)
        elif diff < 0:        yellow_flags.append(pv)
        else:                 green_flags.append(pv)

    def render_flag_card(pv: dict, color: str) -> None:
        t    = pv["ticker"]
        info = current_info.get(t, {})
        st.markdown(
            flag_card_html(
                ticker=t, name=pv["name"], sector=pv["sector"],
                period_return=pv["period_return"], bench_return=bench_return,
                total_return=pv["total_return"],
                current_price=info.get("current_price"),
                ind=indicators.get(t, {}),
                alert_color=color,
                enabled_badges=enabled_badges,
            ),
            unsafe_allow_html=True,
        )

    with col_a:
        st.markdown('<div class="section-header">Remove — Underperforming Holdings</div>',
                    unsafe_allow_html=True)
        if not red_flags:
            st.markdown('<div class="alert-green"><b>No critical underperformers</b></div>',
                        unsafe_allow_html=True)
        else:
            for pv in sorted(red_flags, key=lambda x: x["period_return"] or 0):
                render_flag_card(pv, "red")

        st.markdown('<div class="section-header">Watch — Slightly Lagging</div>',
                    unsafe_allow_html=True)
        if not yellow_flags:
            st.markdown('<div class="alert-green"><b>No lagging holdings</b></div>',
                        unsafe_allow_html=True)
        else:
            for pv in sorted(yellow_flags, key=lambda x: x["period_return"] or 0):
                render_flag_card(pv, "yellow")

    with col_b:
        st.markdown('<div class="section-header">Strong Performers</div>',
                    unsafe_allow_html=True)
        for pv in sorted(green_flags, key=lambda x: -(x["period_return"] or 0))[:8]:
            render_flag_card(pv, "green")

        st.markdown('<div class="section-header">Action Summary</div>',
                    unsafe_allow_html=True)
        rows = []
        for pv in portfolio_values:
            ret  = pv["period_return"]
            diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            ind  = indicators.get(pv["ticker"], {})
            comp = ind.get("composite", 0)
            perf_action = (
                "REMOVE"           if diff is not None and diff < threshold else
                "MONITOR"          if diff is not None and diff < 0 else
                "HOLD"             if diff is not None else
                "Insufficient Data"
            )
            _, tech_action = score_label(comp)
            rows.append({
                "Ticker":      pv["ticker"],
                "Period Ret":  fmt_pct(ret),
                "vs BM":       fmt_pct(diff),
                "RSI":         f"{ind.get('rsi'):.0f}" if ind.get("rsi") else "N/A",
                "MA Cross":    (ind.get("ma_cross", "-") or "-").replace("_", " ").upper(),
                "Beta":        f"{ind.get('beta'):.2f}" if ind.get("beta") else "N/A",
                "ADL":         (ind.get("adl_label", "-") or "-").replace("_", " ").title(),
                "Perf Signal": perf_action,
                "Tech Signal": tech_action,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 3 — TECHNICALS
# ------------------------------------------------------------------------------
with tab3:
    all_tickers_ui = ([h["ticker"] for h in portfolio["holdings"]] +
                      [w["ticker"] for w in watchlist])
    ticker_names   = {h["ticker"]: h.get("name","")
                      for h in portfolio["holdings"] + watchlist}

    tc1, tc2 = st.columns([1, 3])
    with tc1:
        sel_ticker   = st.selectbox(
            "Select stock", all_tickers_ui,
            format_func=lambda t: f"{t} - {ticker_names.get(t,'')}",
        )
        chart_period = st.selectbox("Chart period", ["3M","6M","1Y","All"], index=1)

    ind = indicators.get(sel_ticker, {})
    with tc2:
        scls, slbl = score_label(ind.get("composite", 0))
        b1,b2,b3,b4,b5,b6,b7 = st.columns(7)
        rsi_v  = ind.get("rsi")
        beta_v = ind.get("beta")
        atr_v  = ind.get("atr_pct")
        vol_v  = ind.get("vol_ratio")
        adl_v  = ind.get("adl_label","-")
        with b1: st.metric("RSI (14)",  f"{rsi_v:.1f}"  if rsi_v  else "N/A",
                            delta="OB" if rsi_v and rsi_v>70 else ("OS" if rsi_v and rsi_v<30 else "-"))
        with b2: st.metric("MACD",      (ind.get("macd_label","-") or "-").replace("_"," ").title())
        with b3: st.metric("MA Cross",  (ind.get("ma_cross","-")   or "-").replace("_"," ").title())
        with b4: st.metric("ADL",        adl_v.replace("_"," ").title())
        with b5: st.metric("Beta",      f"{beta_v:.2f}" if beta_v  else "N/A")
        with b6: st.metric("ATR%",      f"{atr_v:.1f}%" if atr_v   else "N/A")
        with b7: st.metric("Vol Ratio", f"{vol_v:.2f}x" if vol_v   else "N/A")

    cp_days  = {"3M": 90, "6M": 180, "1Y": 365, "All": 9999}[chart_period]
    df_full  = ohlcv_data.get(sel_ticker)

    if df_full is not None and not df_full.empty:
        if cp_days < 9999:
            cutoff   = pd.Timestamp(datetime.today() - timedelta(days=cp_days))
            df_chart = df_full[df_full.index >= cutoff]
        else:
            df_chart = df_full
        c_idx = df_chart.index

        fig_tech = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
            subplot_titles=("Price + BB + MAs", "MACD", "RSI (14)"),
        )
        fig_tech.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"],   close=df_chart["Close"], name="Price",
            increasing_line_color="#4ade80", decreasing_line_color="#f87171",
            increasing_fillcolor="#4ade80",  decreasing_fillcolor="#f87171",
        ), row=1, col=1)

        for series_key, name, color, fill in [
            ("bb_upper", "BB Upper", "#60a5fa", "tonexty"),
            ("bb_mid",   "BB Mid",   "#93c5fd", None),
            ("bb_lower", "BB Lower", "#60a5fa", None),
        ]:
            s = ind.get(series_key)
            if s is not None:
                sv = s.reindex(c_idx)
                kw = dict(fill=fill, fillcolor="rgba(96,165,250,0.06)") if fill else {}
                fig_tech.add_trace(go.Scatter(
                    x=sv.index, y=sv, name=name,
                    line=dict(color=color, width=1, dash="dot"), **kw,
                ), row=1, col=1)

        for key, lbl, col in [("ma50","MA 50","#fbbf24"), ("ma200","MA 200","#f472b6")]:
            s = ind.get(key)
            if s is not None:
                sv = s.reindex(c_idx).dropna()
                if not sv.empty:
                    fig_tech.add_trace(go.Scatter(
                        x=sv.index, y=sv, name=lbl, line=dict(color=col, width=1.5),
                    ), row=1, col=1)

        ml = ind.get("macd_line"); ms_ = ind.get("macd_signal"); mh = ind.get("macd_hist")
        if ml is not None and mh is not None:
            mhv = mh.reindex(c_idx).dropna()
            fig_tech.add_trace(go.Bar(
                x=mhv.index, y=mhv, name="Histogram",
                marker_color=["#4ade80" if v >= 0 else "#f87171" for v in mhv], opacity=0.7,
            ), row=2, col=1)
            for s, name, col in [(ml, "MACD","#60a5fa"), (ms_, "Signal","#f472b6")]:
                sv = s.reindex(c_idx).dropna()
                fig_tech.add_trace(go.Scatter(x=sv.index, y=sv, name=name,
                    line=dict(color=col, width=1.5)), row=2, col=1)

        rsi_s = ind.get("rsi_series")
        if rsi_s is not None:
            rv = rsi_s.reindex(c_idx).dropna()
            if not rv.empty:
                fig_tech.add_trace(go.Scatter(x=rv.index, y=rv, name="RSI",
                    line=dict(color="#a78bfa", width=1.8)), row=3, col=1)
                fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.08)",
                    line_width=0, row=3, col=1)
                fig_tech.add_hrect(y0=0, y1=30, fillcolor="rgba(74,222,128,0.08)",
                    line_width=0, row=3, col=1)
                for y, col in [(70, "#f87171"), (30, "#4ade80")]:
                    fig_tech.add_hline(y=y, line_dash="dash", line_color=col,
                        line_width=1, row=3, col=1)

        fig_tech.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis","yaxis")},
            height=680, showlegend=True, xaxis_rangeslider_visible=False,
        )
        for i in range(1, 4):
            fig_tech.update_xaxes(gridcolor="#1e2330", row=i, col=1)
            fig_tech.update_yaxes(gridcolor="#1e2330", row=i, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        adl_s = ind.get("adl")
        if adl_s is not None:
            st.markdown('<div class="section-header">Accumulation / Distribution Line (ADL)</div>',
                        unsafe_allow_html=True)
            adl_d   = adl_s.reindex(c_idx).dropna()
            adl_lbl = ind.get("adl_label","insufficient")
            adl_col = "#4ade80" if "accum" in adl_lbl or "bull" in adl_lbl else "#f87171"
            adl_info = {
                "accumulation":       "Accumulation - money flowing IN, bullish",
                "distribution":       "Distribution - money flowing OUT, bearish",
                "bullish_divergence": "Bullish Divergence - ADL rising while price lags",
                "bearish_divergence": "Bearish Divergence - ADL falling while price holds",
                "insufficient":       "Insufficient data",
            }.get(adl_lbl, "")
            fig_adl = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.5, 0.5], vertical_spacing=0.05,
                                    subplot_titles=("Price", "ADL"))
            fig_adl.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["Close"], name="Price",
                line=dict(color="#7c9aff", width=1.5),
            ), row=1, col=1)
            fig_adl.add_trace(go.Scatter(
                x=adl_d.index, y=adl_d, name="ADL", fill="tozeroy",
                fillcolor=f"rgba({'74,222,128' if adl_col=='#4ade80' else '248,113,113'},0.08)",
                line=dict(color=adl_col, width=2),
            ), row=2, col=1)
            fig_adl.update_layout(
                **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis","yaxis")},
                height=340, xaxis_rangeslider_visible=False,
            )
            for i in range(1, 3):
                fig_adl.update_xaxes(gridcolor="#1e2330", row=i, col=1)
                fig_adl.update_yaxes(gridcolor="#1e2330", row=i, col=1)
            st.plotly_chart(fig_adl, use_container_width=True)
            if adl_info:
                st.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                    f'color:{adl_col};margin-top:-0.5rem">{adl_info}</div>',
                    unsafe_allow_html=True,
                )

        rsc_s = ind.get("rsc")
        if rsc_s is not None:
            st.markdown('<div class="section-header">Relative Strength Comparative (RSC)</div>',
                        unsafe_allow_html=True)
            rsc_d = rsc_s.reindex(c_idx).dropna()
            if len(rsc_d) > 1:
                rsc_norm  = normalize(rsc_d)
                rising    = rsc_norm.iloc[-1] > rsc_norm.iloc[0]
                rsc_color = "#4ade80" if rising else "#f87171"
                fig_rsc   = go.Figure(go.Scatter(
                    x=rsc_norm.index, y=rsc_norm, name="RSC", fill="tozeroy",
                    fillcolor=f"rgba({'74,222,128' if rising else '248,113,113'},0.08)",
                    line=dict(color=rsc_color, width=2),
                ))
                fig_rsc.add_hline(y=100, line_dash="dash", line_color="#4c6ef5", line_width=1)
                fig_rsc.update_layout(**CHART_LAYOUT, height=200, yaxis_title="RSC (indexed)")
                st.plotly_chart(fig_rsc, use_container_width=True)

        if "Volume" in df_chart.columns:
            st.markdown('<div class="section-header">Volume vs 20-Day Average</div>',
                        unsafe_allow_html=True)
            vol_s  = df_chart["Volume"]
            vol_ma = vol_s.rolling(20).mean()
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=vol_s.index, y=vol_s, name="Volume", opacity=0.7,
                marker_color=["#4ade80" if c >= o else "#f87171"
                              for c, o in zip(df_chart["Close"], df_chart["Open"])],
            ))
            fig_vol.add_trace(go.Scatter(x=vol_ma.index, y=vol_ma, name="20D Avg",
                line=dict(color="#fbbf24", width=2)))
            fig_vol.update_layout(**CHART_LAYOUT, height=180)
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning(f"No OHLCV data for {sel_ticker}.")


# ------------------------------------------------------------------------------
# TAB 4 — WATCHLIST
# ------------------------------------------------------------------------------
with tab4:
    _wl_has_data = _scope in ("all", "watchlist")
    if not _wl_has_data:
        st.info(
            "Watchlist data was not fetched. Change the **Refresh scope** to "
            "**All** or **Watchlist** and press **Refresh Data**.",
            icon="ℹ️",
        )
    elif not st.session_state.get("watchlist_processed", False):
        st.markdown("")
        st.markdown(
            '<div style="text-align:center;padding:3rem 0">'
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.9rem;font-weight:600;'
            'color:#e2e8f0;margin-bottom:0.6rem">Watchlist ready</div>'
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#5c667a;'
            'margin-bottom:1.5rem">Charts and indicators will render when you press the button below.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _wl_c1, _wl_c2, _wl_c3 = st.columns([1, 1, 1])
        with _wl_c2:
            if st.button("Process Watchlist", key="wl_process_btn", type="primary",
                         use_container_width=True):
                st.session_state["watchlist_processed"] = True
                st.rerun()
    else:
        wl_col1, wl_col2 = st.columns([1.4, 1])
        with wl_col1:
            st.markdown('<div class="section-header">Watchlist Performance vs Benchmark</div>',
                        unsafe_allow_html=True)
            fig_wl = go.Figure()
            if bench_period is not None:
                fig_wl.add_trace(go.Scatter(
                    x=bench_period.index, y=normalize(bench_period),
                    name=portfolio["benchmark_name"],
                    line=dict(color="#4c6ef5", width=2.5, dash="dot"),
                ))
            wl_colors = ["#a78bfa","#34d399","#fb923c","#f472b6","#38bdf8"]
            for i, w in enumerate(watchlist):
                s = get_close(ohlcv_data, w["ticker"], lookback_days)
                if s is not None and len(s) > 1:
                    fig_wl.add_trace(go.Scatter(
                        x=s.index, y=normalize(s), name=w["ticker"],
                        line=dict(color=wl_colors[i % len(wl_colors)], width=2),
                    ))
            fig_wl.update_layout(**CHART_LAYOUT, height=380, yaxis_title="Indexed (base=100)")
            st.plotly_chart(fig_wl, use_container_width=True)
        with wl_col2:
            st.markdown('<div class="section-header">Watchlist Detail</div>', unsafe_allow_html=True)
            for w in watchlist:
                t    = w["ticker"]
                info = current_info.get(t, {})
                st.markdown(
                    watchlist_card_html(
                        ticker=t, name=w["name"], sector=w.get("sector","-"),
                        reason=w["reason"],
                        period_return=calc_return(get_close(ohlcv_data, t, lookback_days)),
                        bench_return=bench_return,
                        current_price=info.get("current_price"),
                        day_change_pct=info.get("day_change_pct"),
                        ind=indicators.get(t, {}),
                        enabled_badges=enabled_badges,
                    ),
                    unsafe_allow_html=True,
                )
        st.markdown('<div class="section-header">Watchlist Indicator Summary</div>',
                    unsafe_allow_html=True)
        wl_rows = []
        for w in watchlist:
            t    = w["ticker"]
            ret  = calc_return(get_close(ohlcv_data, t, lookback_days))
            diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            ind  = indicators.get(t, {})
            _, slbl = score_label(ind.get("composite", 0))
            wl_rows.append({
                "Ticker": t, "Name": w["name"], "Sector": w.get("sector","-"),
                f"{lookback_days}D Ret": fmt_pct(ret), "vs BM": fmt_pct(diff),
                "RSI":   f"{ind.get('rsi'):.0f}" if ind.get("rsi") else "N/A",
                "MACD":  (ind.get("macd_label","-") or "-").replace("_"," ").title(),
                "MA":    (ind.get("ma_cross","-")   or "-").replace("_"," ").title(),
                "ADL":   (ind.get("adl_label","-")  or "-").replace("_"," ").title(),
                "Beta":  f"{ind.get('beta'):.2f}" if ind.get("beta") else "N/A",
                "ATR%":  f"{ind.get('atr_pct'):.1f}%" if ind.get("atr_pct") else "N/A",
                "Signal": slbl,
            })
        st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 5 — ANALYTICS
# ------------------------------------------------------------------------------
with tab5:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown('<div class="section-header">Returns vs Benchmark</div>',
                    unsafe_allow_html=True)
        bar_data = [{"Ticker": pv["ticker"],
                     "diff": (pv["period_return"] - bench_return)
                              if (pv["period_return"] is not None and bench_return is not None)
                              else 0}
                    for pv in portfolio_values]
        bar_df = pd.DataFrame(bar_data).sort_values("diff")
        fig_bar = go.Figure(go.Bar(
            x=bar_df["Ticker"], y=bar_df["diff"],
            marker_color=["#f87171" if v < threshold else ("#fbbf24" if v < 0 else "#4ade80")
                          for v in bar_df["diff"]],
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig_bar.add_hline(y=threshold, line_dash="dash", line_color="#f87171",
                          annotation_text=f"Threshold ({threshold}%)")
        fig_bar.add_hline(y=0, line_color="#4c6ef5", line_width=1)
        fig_bar.update_layout(**CHART_LAYOUT, height=340, yaxis_title="% vs Benchmark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with a2:
        st.markdown('<div class="section-header">Portfolio Allocation</div>',
                    unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=[pv["ticker"] for pv in portfolio_values],
            values=[pv["current_val"] for pv in portfolio_values],
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Bold,
                        line=dict(color="#0d0f14", width=2)),
            textfont=dict(family="DM Mono", size=11),
            hovertemplate="%{label}: $%{value:,.0f} (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="DM Mono", color="#94a3b8"),
                              legend=dict(bgcolor="rgba(0,0,0,0)"),
                              margin=dict(l=0,r=0,t=0,b=0), height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">RSI vs Period Return - Quadrant View</div>',
                unsafe_allow_html=True)
    sc_rows = [{"ticker": pv["ticker"], "rsi": indicators.get(pv["ticker"],{}).get("rsi"),
                "return": pv["period_return"],
                "beta": indicators.get(pv["ticker"],{}).get("beta") or 1.0,
                "sector": pv["sector"]}
               for pv in portfolio_values
               if indicators.get(pv["ticker"],{}).get("rsi") and pv["period_return"] is not None]
    if sc_rows:
        sc_df = pd.DataFrame(sc_rows)
        fig_sc = go.Figure(go.Scatter(
            x=sc_df["rsi"], y=sc_df["return"], mode="markers+text",
            text=sc_df["ticker"], textposition="top center",
            marker=dict(size=sc_df["beta"].abs() * 12 + 6, color=sc_df["return"],
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="Return %", thickness=10)),
            hovertemplate="<b>%{text}</b><br>RSI: %{x:.1f}<br>Return: %{y:.1f}%<extra></extra>",
        ))
        for x, col, lbl in [(70,"#f87171","Overbought"),(30,"#4ade80","Oversold")]:
            fig_sc.add_vline(x=x, line_dash="dash", line_color=col, line_width=1,
                             annotation_text=lbl, annotation_font_color=col)
        fig_sc.add_hline(y=bench_return or 0, line_dash="dash", line_color="#4c6ef5",
                         line_width=1, annotation_text="Benchmark")
        fig_sc.update_layout(**CHART_LAYOUT, height=380,
                             xaxis_title="RSI (14)", yaxis_title="Period Return %")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<div class="section-header">Beta vs Volatility (ATR%) - Risk Profile</div>',
                unsafe_allow_html=True)
    rk_rows = [{"ticker": pv["ticker"],
                "beta":    indicators.get(pv["ticker"],{}).get("beta"),
                "atr_pct": indicators.get(pv["ticker"],{}).get("atr_pct"),
                "return":  pv["period_return"] or 0}
               for pv in portfolio_values
               if indicators.get(pv["ticker"],{}).get("beta") is not None
               and indicators.get(pv["ticker"],{}).get("atr_pct") is not None]
    if rk_rows:
        rk_df = pd.DataFrame(rk_rows)
        fig_risk = go.Figure(go.Scatter(
            x=rk_df["beta"], y=rk_df["atr_pct"], mode="markers+text",
            text=rk_df["ticker"], textposition="top center",
            marker=dict(size=12, color=rk_df["return"], colorscale="RdYlGn",
                        showscale=True, colorbar=dict(title="Return %", thickness=10)),
            hovertemplate="<b>%{text}</b><br>Beta: %{x:.2f}<br>ATR%: %{y:.1f}%<extra></extra>",
        ))
        fig_risk.add_vline(x=1.0, line_dash="dash", line_color="#4c6ef5", line_width=1,
                           annotation_text="Beta=1")
        fig_risk.update_layout(**CHART_LAYOUT, height=360,
                               xaxis_title="Beta", yaxis_title="ATR %")
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<div class="section-header">Full Holdings - All Indicators</div>',
                unsafe_allow_html=True)
    full_rows = []
    for h in portfolio["holdings"]:
        t    = h["ticker"]
        info = current_info.get(t, {})
        cp   = info.get("current_price")
        pv   = next((p for p in portfolio_values if p["ticker"] == t), None)
        ind  = indicators.get(t, {})
        _, slbl = score_label(ind.get("composite", 0))
        full_rows.append({
            "Ticker": t, "Sector": h["sector"],
            "Price":  f"${cp:.2f}"  if cp else "N/A",
            "Day":    fmt_pct(info.get("day_change_pct"), 1),
            "Period": fmt_pct(pv["period_return"] if pv else None),
            "Total":  fmt_pct(pv["total_return"]  if pv else None),
            "RSI":    f"{ind.get('rsi'):.0f}"      if ind.get("rsi")      else "N/A",
            "MACD":   (ind.get("macd_label","-")   or "-").replace("_"," ").title(),
            "MA":     (ind.get("ma_cross","-")      or "-").replace("_"," ").title(),
            "ADL":    (ind.get("adl_label","-")     or "-").replace("_"," ").title(),
            "Beta":   f"{ind.get('beta'):.2f}"     if ind.get("beta")     else "N/A",
            "ATR%":   f"{ind.get('atr_pct'):.1f}%" if ind.get("atr_pct") else "N/A",
            "Vol":    f"{ind.get('vol_ratio'):.1f}x" if ind.get("vol_ratio") else "N/A",
            "Signal": slbl,
        })
    st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Sector Breakdown</div>', unsafe_allow_html=True)
    sector_map: dict = {}
    for pv in portfolio_values:
        s = pv["sector"]
        sector_map.setdefault(s, {"value": 0, "tickers": []})
        sector_map[s]["value"]   += pv["current_val"]
        sector_map[s]["tickers"].append(pv["ticker"])
    sec_rows = [
        {"Sector": k, "Value": f"${v['value']:,.0f}",
         "Weight": f"{v['value']/total_port_val*100:.1f}%",
         "Tickers": ", ".join(v["tickers"])}
        for k, v in sorted(sector_map.items(), key=lambda x: -x[1]["value"])
    ]
    st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 6 — MOMENTUM
# ------------------------------------------------------------------------------
with tab6:
    mcfg     = get_momentum_cfg(cfg)
    top_n    = int(mcfg["top_n"])
    universe = tuple(dict.fromkeys(
        [h["ticker"] for h in portfolio["holdings"]] +
        [w["ticker"] for w in watchlist]
    ))

    if not st.session_state.get("momentum_processed", False):
        st.markdown("")
        st.markdown(
            '<div style="text-align:center;padding:3rem 0">'
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.9rem;font-weight:600;'
            'color:#e2e8f0;margin-bottom:0.6rem">Momentum analysis ready</div>'
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#5c667a;'
            'margin-bottom:1.5rem">'
            f'Scores will be computed for {len(universe)} tickers when you press the button below.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _m_c1, _m_c2, _m_c3 = st.columns([1, 1, 1])
        with _m_c2:
            if st.button("Run Momentum Analysis", key="mom_process_btn", type="primary",
                         use_container_width=True):
                st.session_state["momentum_processed"] = True
                st.rerun()
    else:
        period_label  = "Monthly" if mcfg["use_monthly"] else "Weekly"
        weekday_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        # Build price / high / low DataFrames directly from the already-loaded
        # ohlcv_data — no second network call needed.
        _mom_px_dict = {t: ohlcv_data[t]["Close"]
                        for t in universe if t in ohlcv_data}
        _mom_hi_dict = {t: ohlcv_data[t]["High"]
                        for t in universe if t in ohlcv_data and "High" in ohlcv_data[t]}
        _mom_lo_dict = {t: ohlcv_data[t]["Low"]
                        for t in universe if t in ohlcv_data and "Low"  in ohlcv_data[t]}

        mom_prices = (pd.concat(_mom_px_dict, axis=1).dropna(axis=1, how="all").ffill()
                      if _mom_px_dict else pd.DataFrame())
        mom_highs  = (pd.concat(_mom_hi_dict, axis=1).dropna(axis=1, how="all").ffill()
                      if _mom_hi_dict else pd.DataFrame())
        mom_lows   = (pd.concat(_mom_lo_dict, axis=1).dropna(axis=1, how="all").ffill()
                      if _mom_lo_dict else pd.DataFrame())

        # VIX comes from ohlcv_data if it was fetched (use_vix_rank=True adds it)
        if mcfg["use_vix_rank"] and _vix_ticker in ohlcv_data:
            vix_series = ohlcv_data[_vix_ticker]["Close"].dropna()
        else:
            vix_series = pd.Series(dtype=float)
            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                if mcfg["use_monthly"]:
                    st.markdown(f"**Rebalance:** Monthly ({'End' if mcfg['month_end'] else 'Start'})")
                else:
                    st.markdown(f"**Rebalance:** Weekly ({weekday_names[mcfg['weekday']]})")
                st.markdown(f"**Momentum lookback:** {mcfg['mom_lookback']} periods")
                st.markdown(f"**Require positive:** {'yes' if mcfg['require_positive'] else 'no'}")
            with ec2:
                st.markdown(f"**Portfolio slots:** {mcfg['portfolio_slots']}")
                st.markdown(f"**Top candidates:** {mcfg['top_candidates']}")
                st.markdown(f"**Corr lookback:** {mcfg['corr_lb']} days")
                st.markdown(f"**Use correlation:** {'yes' if mcfg['use_correlation'] else 'no'}")
            with ec3:
                st.markdown(f"**BBI filter:** {'on (' + str(mcfg['bbi_lookback']) + 'd)' if mcfg['use_bbi'] else 'off'}")
                st.markdown(f"**ETS filter:** {'on' if mcfg['use_ets'] else 'off'}")
                st.markdown(f"**RSI period:** {mcfg['rsi_period']}")
                st.markdown(f"**Logistic prob:** {'on' if mcfg['use_logistic_prob'] else 'off'}")
            with ec4:
                st.markdown(f"**VIX rank gate:** {'on max ' + str(mcfg['max_vix_rank']) + '%' if mcfg['use_vix_rank'] else 'off'}")
                st.markdown(f"**Inv-vol weights:** {'yes' if mcfg['show_invvol'] else 'no'}")
                st.markdown(f"**Cash proxy:** {mcfg['cash_proxy']}")
                st.markdown(f"**Top N shown:** {top_n}")

        st.markdown("")

        with st.spinner("Loading price history for momentum scoring..."):
            mom_prices          = fetch_momentum_prices(universe, days=400,
                                                        use_yfinance=use_yfinance, data_dir=data_dir)
            mom_highs, mom_lows = fetch_momentum_hl(universe, days=400,
                                                    use_yfinance=use_yfinance, data_dir=data_dir)
            vix_series          = fetch_vix(days=400) if mcfg["use_vix_rank"] else pd.Series(dtype=float)

        if mom_prices.empty:
            st.warning("No price data available for the momentum universe.")
            st.stop()

        result      = score_candidates(mom_prices, mom_highs, mom_lows, vix_series, mcfg)
        ranked      = result["ranked"]
        selected    = result["selected"]
        iv_weights  = result["inv_vol_weights"]
        dec_date    = result["decision_date"]
        vix_rank    = result["vix_rank"]
        vix_blocked = result["vix_blocked"]

        if ranked.empty:
            st.info("Not enough history to compute momentum scores. Try reducing mom_lookback.")
            st.stop()

        vix_val = float(vix_series.iloc[-1]) if not vix_series.empty else None
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Decision Date", dec_date.strftime("%Y-%m-%d") if dec_date is not None else "N/A")
        with m2:
            st.metric("Rebalance Period", period_label)
        with m3:
            st.metric("Candidates Found", str(len(result["top_candidates"])))
        with m4:
            st.metric("Slots Selected", str(len([s for s in selected if s != mcfg["cash_proxy"]])))
        with m5:
            if mcfg["use_vix_rank"] and vix_rank is not None:
                st.metric("VIX Rank", f"{vix_rank:.1f}%",
                          delta="BLOCKED" if vix_blocked else f"Limit: {mcfg['max_vix_rank']}%")
            elif vix_val:
                st.metric("VIX", f"{vix_val:.2f}", delta="VIX rank gate off")
            else:
                st.metric("VIX", "N/A", delta="not fetched")

        if vix_blocked:
            st.warning(
                f"VIX rank ({vix_rank:.1f}%) exceeds max ({mcfg['max_vix_rank']}%). "
                f"Slots defaulting to {mcfg['cash_proxy']}."
            )

        st.markdown("")
        st.markdown('<div class="section-header">Selected Portfolio Slots</div>', unsafe_allow_html=True)

        non_cash_selected = [s for s in selected if s != mcfg["cash_proxy"]]
        if non_cash_selected:
            slot_cols = st.columns(min(len(selected), 4))
            for i, ticker in enumerate(selected):
                is_cash   = (ticker == mcfg["cash_proxy"])
                row_data  = ranked[ranked["ticker"] == ticker]
                mom_s     = float(row_data["mom_score"].iloc[0]) if not row_data.empty else None
                rsi_s     = row_data["rsi"].iloc[0]               if not row_data.empty else None
                iv_w      = iv_weights.get(ticker)
                col_color = "#5c667a" if is_cash else "#4ade80"
                with slot_cols[i % 4]:
                    if mom_s is not None and rsi_s is not None and iv_w is not None:
                        st.markdown(
                            f'<div class="alert-green" style="text-align:center;">'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                            f'color:#5c667a;text-transform:uppercase;letter-spacing:2px">Slot {i+1}</div>'
                            f'<div class="ticker-badge" style="color:{col_color};font-size:1.1rem;'
                            f'margin:6px 0">{ticker}</div>'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                            f'color:#86efac">Mom {mom_s:+.1%} &middot; RSI {rsi_s:.0f}</div>'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                            f'color:#5c667a;margin-top:4px">Inv-Vol wt: {iv_w:.1%}</div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="alert-blue" style="text-align:center;">'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                            f'color:#5c667a;text-transform:uppercase;letter-spacing:2px">Slot {i+1}</div>'
                            f'<div class="ticker-badge" style="color:#94a3b8;font-size:1.1rem;'
                            f'margin:6px 0">{ticker}</div>'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                            f'color:#5c667a">Cash proxy</div></div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                f'<div class="alert-yellow">All {mcfg["portfolio_slots"]} slots holding cash proxy '
                f'<b>{mcfg["cash_proxy"]}</b></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="section-header">Top {top_n} Momentum Candidates</div>',
                    unsafe_allow_html=True)

        use_bbi = bool(mcfg["use_bbi"])
        use_ets = bool(mcfg["use_ets"])
        use_lp  = bool(mcfg["use_logistic_prob"])

        def _slot_status(row) -> str:
            t = row["ticker"]
            if t in selected and t != mcfg["cash_proxy"]: return "SLOT"
            if row["in_top_cands"]:                        return "CANDIDATE"
            if row["mom_score"] is not None:               return "RANKED"
            return "-"

        disp = pd.DataFrame({"Ticker": ranked["ticker"]})
        disp["Status"]    = ranked.apply(_slot_status, axis=1)
        disp["Mom Score"] = ranked["mom_score"].apply(
            lambda x: f"{x:+.2%}" if x is not None and not pd.isna(x) else "N/A")
        disp["RSI"]       = ranked["rsi"].apply(
            lambda x: f"{x:.1f}" if x is not None else "N/A")
        if use_bbi:
            disp["BBI"]    = ranked["bbi"].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
            disp["BBI OK"] = ranked["passes_bbi"].map({True: "Y", False: "N", None: "-"})
        if use_ets:
            disp["ETS"]    = ranked["ets"].apply(lambda x: str(x) if x is not None else "N/A")
            disp["ETS OK"] = ranked["passes_ets"].map({True: "Y", False: "N", None: "-"})
        if use_lp:
            disp["LogProb"] = ranked["logprob"].apply(
                lambda x: f"{x:.3f}" if x is not None and not pd.isna(x) else "N/A")
            disp["RSR"]     = ranked["rsr"].apply(
                lambda x: f"{x:.3f}" if x is not None and not pd.isna(x) else "N/A")
        disp["Top-K"]    = ranked["in_top_cands"].map({True: "Y", False: "-"})
        disp["Selected"] = ranked["selected"].map({True: "Y", False: "-"})
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Momentum Scores</div>', unsafe_allow_html=True)
        bar_data = ranked.dropna(subset=["mom_score"]).copy()
        if not bar_data.empty:
            bar_colors = ["#4ade80" if r["selected"] else "#fbbf24" if r["in_top_cands"] else "#94a3b8"
                          for _, r in bar_data.iterrows()]
            fig_mbar = go.Figure(go.Bar(
                x=bar_data["ticker"], y=bar_data["mom_score"]*100, marker_color=bar_colors,
                text=[f"{v:+.1f}%" for v in bar_data["mom_score"]*100], textposition="outside",
                hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
            ))
            fig_mbar.add_hline(y=0, line_color="#4c6ef5", line_width=1)
            fig_mbar.update_layout(**CHART_LAYOUT, height=340,
                                   yaxis_title=f"{mcfg['mom_lookback']}-Period Momentum Return %")
            st.plotly_chart(fig_mbar, use_container_width=True)

        top_tickers = result["top_candidates"]
        if len(top_tickers) >= 2 and all(t in mom_prices.columns for t in top_tickers):
            st.markdown('<div class="section-header">Correlation Heatmap</div>',
                        unsafe_allow_html=True)
            corr_px = mom_prices[top_tickers].tail(mcfg["corr_lb"]+5).ffill().pct_change().dropna()
            if not corr_px.empty:
                corr_mat = corr_px.corr()
                fig_heat = go.Figure(go.Heatmap(
                    z=corr_mat.values, x=corr_mat.columns.tolist(), y=corr_mat.index.tolist(),
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" for v in r] for r in corr_mat.values],
                    texttemplate="%{text}", showscale=True, colorbar=dict(thickness=12),
                ))
                for t in selected:
                    if t in corr_mat.columns:
                        idx = corr_mat.columns.tolist().index(t)
                        fig_heat.add_shape(type="rect",
                            x0=idx-0.5, x1=idx+0.5, y0=-0.5, y1=len(corr_mat)-0.5,
                            line=dict(color="#4ade80", width=2))
                fig_heat.update_layout(
                    **{k:v for k,v in CHART_LAYOUT.items() if k not in ("xaxis","yaxis")},
                    height=max(300, len(top_tickers)*45),
                    xaxis=dict(gridcolor="#1e2330"),
                    yaxis=dict(gridcolor="#1e2330", autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("Green border = selected slot")


# ------------------------------------------------------------------------------
# TAB 7 — CONFIG
# ------------------------------------------------------------------------------
with tab7:
    cfg_subtab1, cfg_subtab2, cfg_subtab3, cfg_subtab4 = st.tabs([
        "Watchlist Builder", "Portfolio Builder", "Config Editor", "Data Status",
    ])

    # -- Watchlist Builder ----------------------------------------------------
    with cfg_subtab1:
        _cfg_path_wl = Path(__file__).parent / "portfolio_config.json"
        st.caption(
            "Enter comma-separated symbols and click **Apply to Config** to save them "
            "directly into the watchlist section of `portfolio_config.json`. "
            "No market data is fetched. Press **Refresh Data** when ready."
        )
        st.markdown("")
        wl_col1, wl_col2 = st.columns([1, 1])
        with wl_col1:
            st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)
            wl_symbols_raw = st.text_area(
                "Symbols", placeholder="AAPL, ABBV, ADBE, NVDA, META",
                height=120, key="cfg_wl_input", label_visibility="collapsed",
            )
            wl_tickers = [t.strip().upper() for t in wl_symbols_raw.split(",") if t.strip()]
            wl_reason  = st.text_input(
                "Reason / notes (applied to all)",
                placeholder="Sector rotation candidates",
                key="cfg_wl_reason",
            )
            wl_sector  = st.text_input(
                "Sector (applied to all, leave blank to keep existing)",
                placeholder="Technology",
                key="cfg_wl_sector",
            )
        with wl_col2:
            st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
            if wl_tickers:
                wl_entries = [
                    {
                        "ticker": t,
                        "name":   t,
                        "reason": wl_reason.strip() or "",
                        "sector": wl_sector.strip() or "",
                    }
                    for t in wl_tickers
                ]
                st.caption(f"{len(wl_tickers)} ticker(s) — review before applying")
                st.dataframe(
                    pd.DataFrame(wl_entries)[["ticker","reason","sector"]],
                    use_container_width=True, hide_index=True, height=200,
                )
                st.markdown("")
                _wl_replace = st.radio(
                    "Apply mode",
                    ["Replace entire watchlist", "Merge (add new, keep existing)"],
                    key="cfg_wl_mode",
                    horizontal=True,
                )
                if st.button("Apply to Config", key="cfg_wl_apply_btn",
                             type="primary", use_container_width=True):
                    try:
                        with open(_cfg_path_wl, encoding="utf-8") as _f:
                            _disk_cfg = json.load(_f)
                        import shutil as _shutil
                        _shutil.copy2(str(_cfg_path_wl), str(_cfg_path_wl) + ".bak")
                        if _wl_replace == "Replace entire watchlist":
                            _disk_cfg["watchlist"] = wl_entries
                        else:
                            existing_tickers = {w["ticker"] for w in _disk_cfg.get("watchlist", [])}
                            _disk_cfg["watchlist"] = (
                                _disk_cfg.get("watchlist", []) +
                                [e for e in wl_entries if e["ticker"] not in existing_tickers]
                            )
                        with open(_cfg_path_wl, "w", encoding="utf-8") as _f:
                            json.dump(_disk_cfg, _f, indent=2, ensure_ascii=False)
                        st.session_state.pop("cfg_session_override", None)
                        st.success(
                            f"Saved {len(_disk_cfg['watchlist'])} watchlist entry/entries "
                            f"to `portfolio_config.json`. Press **Refresh Data** to reload."
                        )
                    except Exception as _wl_err:
                        st.error(f"Save failed: {_wl_err}")
            else:
                st.info("Enter at least one symbol on the left.")

    # -- Portfolio Builder ----------------------------------------------------
    with cfg_subtab2:
        _cfg_path_port = Path(__file__).parent / "portfolio_config.json"
        st.caption(
            "Paste your brokerage holdings export and click **Apply to Config** to save "
            "them directly into the `portfolio.holdings` section of `portfolio_config.json`. "
            "No market data is fetched. Press **Refresh Data** when ready."
        )
        st.markdown("")
        with st.expander("Accepted input formats", expanded=False):
            st.markdown(
                "**Format A** - one field per line (6-line records after 6-line header):\n"
                "Account / Sec / Asset / Date Add / Price / Qty\n\n"
                "**Format B** - tab, comma, or multi-space delimited rows.\n\n"
                "Symbol rules: -T becomes .TO; any other suffix is stripped."
            )
        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            st.markdown('<div class="section-header">Paste Brokerage Data</div>',
                        unsafe_allow_html=True)
            port_raw_cfg = st.text_area(
                "Brokerage data", height=280,
                key="cfg_port_input", label_visibility="collapsed",
            )
        with p_col2:
            st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
            if port_raw_cfg.strip():
                holdings_parsed, parse_warnings = _cfg_parse_brokerage(port_raw_cfg)
                for w in parse_warnings:
                    st.warning(w)
                if holdings_parsed:
                    prev_df = pd.DataFrame([{
                        "Ticker": h["ticker"], "Sector": h["sector"],
                        "Shares": h["shares"], "Avg Cost": f'${h["avg_cost"]:,.4f}',
                    } for h in holdings_parsed])
                    st.caption(f"{len(holdings_parsed)} holding(s) — review before applying")
                    st.dataframe(prev_df, use_container_width=True, hide_index=True, height=160)
                    st.markdown("")
                    _port_replace = st.radio(
                        "Apply mode",
                        ["Replace entire holdings", "Merge (add new tickers, keep existing)"],
                        key="cfg_port_mode",
                        horizontal=True,
                    )
                    if st.button("Apply to Config", key="cfg_port_apply_btn",
                                 type="primary", use_container_width=True):
                        try:
                            with open(_cfg_path_port, encoding="utf-8") as _f:
                                _disk_cfg = json.load(_f)
                            import shutil as _shutil
                            _shutil.copy2(str(_cfg_path_port), str(_cfg_path_port) + ".bak")
                            if _port_replace == "Replace entire holdings":
                                _disk_cfg["portfolio"]["holdings"] = holdings_parsed
                            else:
                                existing_tickers = {
                                    h["ticker"]
                                    for h in _disk_cfg.get("portfolio", {}).get("holdings", [])
                                }
                                _disk_cfg["portfolio"]["holdings"] = (
                                    _disk_cfg["portfolio"].get("holdings", []) +
                                    [h for h in holdings_parsed if h["ticker"] not in existing_tickers]
                                )
                            with open(_cfg_path_port, "w", encoding="utf-8") as _f:
                                json.dump(_disk_cfg, _f, indent=2, ensure_ascii=False)
                            st.session_state.pop("cfg_session_override", None)
                            st.success(
                                f"Saved {len(_disk_cfg['portfolio']['holdings'])} holding(s) "
                                f"to `portfolio_config.json`. Press **Refresh Data** to reload."
                            )
                        except Exception as _port_err:
                            st.error(f"Save failed: {_port_err}")
                else:
                    st.error("No holdings could be parsed. "
                             "Expected: Account, Sec, Asset, Date Add, Price, Qty.")
            else:
                st.info("Paste brokerage data on the left.")

    # -- Config Editor --------------------------------------------------------
    with cfg_subtab3:
        _cfg_path = Path(__file__).parent / "portfolio_config.json"
        _has_session_cfg = st.session_state.get("cfg_session_override") is not None
        if _has_session_cfg:
            st.info(
                "Session override active - config applied in-memory, not written to disk. "
                "Click Save to File to persist, or Reset to File to discard."
            )
        _editor_src     = st.session_state.get("cfg_session_override") or _cfg_raw or cfg
        _editor_default = json.dumps(_editor_src, indent=2) if _editor_src else "{}"
        st.caption(
            "Edit portfolio_config.json directly. "
            "Apply loads into the session without touching the file. "
            "Save to File writes to disk (creates .bak backup). "
            "Reset to File discards session edits."
        )
        st.markdown("")
        ed_col, val_col = st.columns([3, 2])
        with ed_col:
            st.markdown('<div class="section-header">JSON Editor</div>', unsafe_allow_html=True)
            edited_json = st.text_area(
                "Config", value=_editor_default, height=560,
                key="cfg_editor_text", label_visibility="collapsed",
            )
        with val_col:
            st.markdown('<div class="section-header">Validation and Actions</div>',
                        unsafe_allow_html=True)
            _parsed_cfg = None
            _val_errors: list[str] = []
            try:
                _parsed_cfg = json.loads(edited_json)
                _val_errors = validate_config(_parsed_cfg)
                if _val_errors:
                    st.error(f"Schema errors ({len(_val_errors)}):")
                    for e in _val_errors:
                        st.markdown(
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
                            f'color:#f87171;margin:2px 0">- {e}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    _hc = len(_parsed_cfg.get("portfolio", {}).get("holdings", []))
                    _wc = len(_parsed_cfg.get("watchlist", []))
                    st.success(f"Valid - {_hc} holding(s), {_wc} watchlist ticker(s)")
            except json.JSONDecodeError as _je:
                st.error(f"Invalid JSON: {_je}")
            _can_apply = _parsed_cfg is not None and not _val_errors
            st.markdown("")
            if st.button("Apply (session only)", key="cfg_apply_btn", use_container_width=True,
                         disabled=not _can_apply,
                         help="Load into session without saving. Press Refresh Data afterwards."):
                st.session_state["cfg_session_override"] = _parsed_cfg
                st.session_state["data_loaded"] = False
                st.cache_data.clear()
                st.rerun()
            st.markdown("")
            if st.button("Save to File", key="cfg_save_btn", use_container_width=True,
                         disabled=not _can_apply,
                         help="Write to portfolio_config.json. Creates .bak backup first."):
                try:
                    import shutil as _shutil
                    if _cfg_path.exists():
                        _shutil.copy2(str(_cfg_path), str(_cfg_path) + ".bak")
                    with open(_cfg_path, "w", encoding="utf-8") as _f:
                        json.dump(_parsed_cfg, _f, indent=2, ensure_ascii=False)
                    st.session_state.pop("cfg_session_override", None)
                    st.session_state["data_loaded"] = False
                    st.cache_data.clear()
                    st.rerun()
                except Exception as _save_err:
                    st.error(f"Save failed: {_save_err}")
            st.markdown("")
            if st.button("Reset to File", key="cfg_reset_btn", use_container_width=True,
                         help="Discard session edits and reload from disk."):
                st.session_state.pop("cfg_session_override", None)
                st.session_state["data_loaded"] = False
                st.rerun()
            st.markdown("")
            if _parsed_cfg:
                with st.expander("Config summary", expanded=False):
                    _h = _parsed_cfg.get("portfolio", {}).get("holdings", [])
                    _w = _parsed_cfg.get("watchlist", [])
                    _s = _parsed_cfg.get("settings", {})
                    st.markdown(
                        f"**Name:** {_parsed_cfg.get('portfolio',{}).get('name','-')}  \n"
                        f"**Benchmark:** {_parsed_cfg.get('portfolio',{}).get('benchmark_name','-')}  \n"
                        f"**Holdings:** {len(_h)}  \n"
                        f"**Watchlist:** {len(_w)}  \n"
                        f"**Default lookback:** {_s.get('default_lookback_days','?')}D  \n"
                        f"**Threshold:** {_s.get('underperformance_threshold_pct','?')}%"
                    )
                    for h in _h:
                        st.markdown(
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                            f'color:#94a3b8">'
                            f'- {h.get("ticker","?")} {h.get("shares",0)} sh '
                            f'@ ${h.get("avg_cost",0):.2f}</div>',
                            unsafe_allow_html=True,
                        )

    # -- Data Status ----------------------------------------------------------
    with cfg_subtab4:
        st.caption(
            "Price data currently loaded in this session. "
            "Scope and mode are shown in the status bar below the header metrics."
        )
        st.markdown("")

        _port_ticker_set = {h["ticker"] for h in portfolio["holdings"]}
        _wl_ticker_set   = {w["ticker"] for w in watchlist}
        _bench_t         = portfolio["benchmark"]

        _status_rows = []
        for t in all_tickers:
            if t == _bench_t:
                group = "Benchmark"
            elif t in _port_ticker_set:
                group = "Portfolio"
            else:
                group = "Watchlist"

            df = ohlcv_data.get(t)
            if df is not None and not df.empty:
                latest_date  = df.index[-1].strftime("%Y-%m-%d")
                latest_close = f"${df['Close'].iloc[-1]:.2f}"
                days_avail   = len(df)
                fetched      = "Yes"
            else:
                latest_date  = "-"
                latest_close = "-"
                days_avail   = 0
                fetched      = "No"

            _status_rows.append({
                "Ticker":       t,
                "Group":        group,
                "Latest Date":  latest_date,
                "Latest Close": latest_close,
                "Days Loaded":  days_avail,
                "Fetched":      fetched,
            })

        st.dataframe(pd.DataFrame(_status_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Scope: **{_scope_label[_scope]}** · Mode: **{_mode_label}** · "
            f"{len([r for r in _status_rows if r['Fetched'] == 'Yes'])} of "
            f"{len(_status_rows)} tickers fetched this session."
        )


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
    f'color:#2d3748;text-align:center">'
    f'Portfolio Sentinel · RSI · MACD · BB · MA Cross · RSC · Beta · ATR · Volume · ADL '
    f'· Data via Yahoo Finance · {datetime.now().strftime("%Y-%m-%d %H:%M")} '
    f'· Not financial advice.</div>',
    unsafe_allow_html=True,
)
