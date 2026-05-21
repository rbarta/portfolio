import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Sentinel", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ─── CSS ──────────────────────────────────────────────────────────────────────
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
  .score-card  { display:inline-block; font-family:'DM Mono',monospace; font-size:0.85rem;
    font-weight:700; padding:3px 10px; border-radius:6px; margin-left:8px; }
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
  .stSlider>div{ color:#e2e8f0; }
  button[data-baseweb="tab"]{ font-family:'Syne',sans-serif; font-weight:600; color:#5c667a; }
  button[data-baseweb="tab"][aria-selected="true"]{ color:#7c9aff; border-bottom-color:#4c6ef5 !important; }
  ::-webkit-scrollbar{ width:6px; } ::-webkit-scrollbar-track{ background:#0d0f14; }
  ::-webkit-scrollbar-thumb{ background:#1e2330; border-radius:3px; }
  hr{ border-color:#1e2330; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f  = close.ewm(span=fast,   adjust=False).mean()
    ema_s  = close.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal,  adjust=False).mean()
    hist   = macd - sig
    return macd, sig, hist


def calc_bollinger(close: pd.Series, period=20, num_std=2):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    return sma + num_std * std, sma, sma - num_std * std   # upper, mid, lower


def calc_ma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    cp  = c.shift(1)
    tr  = pd.concat([(h - l), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calc_beta(stock_close: pd.Series, bench_close: pd.Series) -> float | None:
    sr  = stock_close.pct_change().dropna()
    br  = bench_close.pct_change().dropna()
    idx = sr.index.intersection(br.index)
    if len(idx) < 30:
        return None
    cov = np.cov(sr.loc[idx], br.loc[idx])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None


def calc_rsc(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series | None:
    idx = stock_close.index.intersection(bench_close.index)
    if len(idx) < 5:
        return None
    return stock_close.loc[idx] / bench_close.loc[idx]


def calc_volume_ratio(volume: pd.Series, period=20) -> float | None:
    if volume is None or len(volume) < period + 2:
        return None
    avg = volume.iloc[-(period + 1):-1].mean()
    return float(volume.iloc[-1] / avg) if avg > 0 else None


def detect_ma_cross(close: pd.Series) -> str:
    """Return: golden_cross | death_cross | bullish | bearish | insufficient"""
    ma50  = calc_ma(close, 50).dropna()
    ma200 = calc_ma(close, 200).dropna()
    if len(ma50) < 2 or len(ma200) < 2:
        return "insufficient"
    diff  = (ma50 - ma200).reindex(ma200.index).dropna()
    if len(diff) < 2:
        return "insufficient"
    cur   = diff.iloc[-1]
    prev  = diff.iloc[-2]
    if cur > 0 and prev <= 0:
        return "golden_cross"
    if cur < 0 and prev >= 0:
        return "death_cross"
    return "bullish" if cur > 0 else "bearish"


def macd_signal_label(hist: pd.Series) -> str:
    """Bullish/bearish/diverging from last two histogram bars."""
    if hist is None or len(hist.dropna()) < 2:
        return "neutral"
    v = hist.dropna()
    last, prev = float(v.iloc[-1]), float(v.iloc[-2])
    if last > 0 and last > prev:
        return "bullish"
    if last < 0 and last < prev:
        return "bearish"
    if last > 0:
        return "weakening_bull"
    return "weakening_bear"


def composite_score(rsi_val, macd_sig, ma_cross, vol_ratio, day_chg) -> int:
    """
    Returns integer score:
      -3 strong sell  |  -2 sell  |  -1 lean sell
       0 neutral
      +1 lean buy     |  +2 buy   |  +3 strong buy
    """
    s = 0
    # RSI
    if rsi_val is not None:
        if   rsi_val < 25:  s -= 2
        elif rsi_val < 40:  s -= 1
        elif rsi_val > 75:  s += 2
        elif rsi_val > 60:  s += 1
    # MACD
    if   macd_sig == "bearish":        s -= 1
    elif macd_sig == "weakening_bear": s -= 1
    elif macd_sig == "bullish":        s += 1
    elif macd_sig == "weakening_bull": s += 1
    # MA cross
    if   ma_cross == "death_cross": s -= 2
    elif ma_cross == "bearish":     s -= 1
    elif ma_cross == "golden_cross":s += 2
    elif ma_cross == "bullish":     s += 1
    # Volume divergence (high volume on down day = distribution)
    if vol_ratio is not None and vol_ratio > 1.5:
        if day_chg is not None and day_chg < 0:
            s -= 1
        elif day_chg is not None and day_chg > 0:
            s += 1
    return max(-4, min(4, s))


def score_label(s: int) -> tuple[str, str]:
    """(css_class, label)"""
    if s <= -3: return "score-strong-sell", "⛔ STRONG SELL"
    if s == -2: return "score-sell",        "🔴 SELL"
    if s == -1: return "score-sell",        "🟠 LEAN SELL"
    if s ==  0: return "score-neutral",     "⚪ NEUTRAL"
    if s ==  1: return "score-buy",         "🟡 LEAN BUY"
    if s ==  2: return "score-buy",         "🟢 BUY"
    return          "score-strong-buy",     "✅ STRONG BUY"


def indicator_badges_html(rsi_val, macd_sig, ma_cross, vol_ratio, beta_val, atr_pct) -> str:
    parts = []
    # RSI badge
    if rsi_val is not None:
        cls = "ind-bear" if rsi_val < 40 else ("ind-bull" if rsi_val > 60 else "ind-neut")
        over = " OB" if rsi_val > 70 else (" OS" if rsi_val < 30 else "")
        parts.append(f'<span class="ind-badge {cls}">RSI {rsi_val:.0f}{over}</span>')
    # MACD badge
    macd_cls = {"bullish":"ind-bull","golden_cross":"ind-bull","weakening_bull":"ind-warn",
                "bearish":"ind-bear","death_cross":"ind-bear","weakening_bear":"ind-warn"}.get(macd_sig,"ind-neut")
    parts.append(f'<span class="ind-badge {macd_cls}">MACD {macd_sig.replace("_"," ").upper()}</span>')
    # MA cross badge
    ma_cls = {"golden_cross":"ind-bull","bullish":"ind-bull",
              "death_cross":"ind-bear","bearish":"ind-bear","insufficient":"ind-neut"}.get(ma_cross,"ind-neut")
    ma_lbl = {"golden_cross":"✨ GOLDEN X","bullish":"50>200","death_cross":"💀 DEATH X",
              "bearish":"50<200","insufficient":"MA N/A"}.get(ma_cross,"—")
    parts.append(f'<span class="ind-badge {ma_cls}">{ma_lbl}</span>')
    # Volume badge
    if vol_ratio is not None:
        vcls = "ind-bull" if vol_ratio > 1.5 else ("ind-neut" if vol_ratio > 0.7 else "ind-warn")
        parts.append(f'<span class="ind-badge {vcls}">VOL {vol_ratio:.1f}x</span>')
    # Beta badge
    if beta_val is not None:
        bcls = "ind-warn" if abs(beta_val) > 1.5 else "ind-neut"
        parts.append(f'<span class="ind-badge {bcls}">β {beta_val:.2f}</span>')
    # ATR badge
    if atr_pct is not None:
        parts.append(f'<span class="ind-badge ind-neut">ATR {atr_pct:.1f}%</span>')
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_ohlcv(tickers: list, display_days: int) -> dict:
    """Fetch OHLCV. Always fetches 300+ days so 200-day MA always has data."""
    fetch_days = max(display_days + 10, 310)
    end   = datetime.today()
    start = end - timedelta(days=fetch_days)
    data  = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df.dropna(subset=["Close"])
        except Exception:
            pass
    return data


@st.cache_data(ttl=300)
def fetch_current_info(tickers: list) -> dict:
    info = {}
    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                cur  = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else cur
                info[ticker] = {"current_price": cur,
                                "day_change_pct": ((cur - prev) / prev) * 100}
        except Exception:
            info[ticker] = {"current_price": None, "day_change_pct": None}
    return info


def get_close(ohlcv: dict, ticker: str, days: int | None = None) -> pd.Series | None:
    df = ohlcv.get(ticker)
    if df is None or df.empty:
        return None
    s = df["Close"]
    if days:
        cutoff = datetime.today() - timedelta(days=days)
        s = s[s.index >= pd.Timestamp(cutoff)]
    return s if len(s) >= 2 else None


def calc_return(series: pd.Series) -> float | None:
    if series is None or len(series) < 2:
        return None
    return ((series.iloc[-1] / series.iloc[0]) - 1) * 100


def normalize(s: pd.Series) -> pd.Series:
    return (s / s.iloc[0]) * 100


def fmt_pct(v, d=2):
    if v is None: return "N/A"
    return f"+{v:.{d}f}%" if v >= 0 else f"{v:.{d}f}%"


def color_class(v):
    if v is None: return "neutral"
    return "positive" if v >= 0 else "negative"


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


REQUIRED_KEYS = {
    "portfolio": ["name", "benchmark", "benchmark_name", "holdings"],
    "watchlist": None,
    "settings":  ["lookback_periods", "default_lookback_days", "underperformance_threshold_pct"],
}
REQUIRED_HOLDING_KEYS = ["ticker", "name", "shares", "avg_cost"]
REQUIRED_WATCHLIST_KEYS = ["ticker", "name", "reason"]


def validate_config(cfg: dict) -> list[str]:
    """Return list of error strings; empty = valid."""
    errors = []
    for top_key, sub_keys in REQUIRED_KEYS.items():
        if top_key not in cfg:
            errors.append(f"Missing top-level key: '{top_key}'")
            continue
        if sub_keys:
            for sk in sub_keys:
                if sk not in cfg[top_key]:
                    errors.append(f"portfolio_config.{top_key} is missing '{sk}'")
    if "portfolio" in cfg and "holdings" in cfg["portfolio"]:
        for i, h in enumerate(cfg["portfolio"]["holdings"]):
            for k in REQUIRED_HOLDING_KEYS:
                if k not in h:
                    errors.append(f"holdings[{i}] ('{h.get('ticker','?')}') missing '{k}'")
    if "watchlist" in cfg:
        for i, w in enumerate(cfg["watchlist"]):
            for k in REQUIRED_WATCHLIST_KEYS:
                if k not in w:
                    errors.append(f"watchlist[{i}] ('{w.get('ticker','?')}') missing '{k}'")
    return errors


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0f14",
    font=dict(family="DM Mono", color="#94a3b8", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hovermode="x unified",
    xaxis=dict(gridcolor="#1e2330", showgrid=True),
    yaxis=dict(gridcolor="#1e2330", showgrid=True),
    margin=dict(l=0, r=0, t=10, b=0),
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # ── Drag-and-drop / file upload ───────────────────────────────────────────
    st.markdown("""
<div style='font-family:"DM Mono",monospace;font-size:0.7rem;color:#5c667a;
     margin-bottom:0.4rem;letter-spacing:1px;text-transform:uppercase'>
  Portfolio Config
</div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drop portfolio_config.json here",
        type=["json"],
        help="Drag & drop your portfolio_config.json, or click to browse",
        label_visibility="collapsed",
    )

    # Resolve config: uploaded file > default file on disk
    cfg = None
    config_source = None

    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            cfg = json.loads(raw)
            config_source = f"📄 {uploaded_file.name}"
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()
    else:
        default_cfg = Path(__file__).parent / "portfolio_config.json"
        if default_cfg.exists():
            cfg = load_config(str(default_cfg))
            config_source = "📁 portfolio_config.json (default)"
        else:
            st.markdown("""
<div style='background:#2b2007;border:1px solid #78350f;border-radius:8px;
     padding:0.8rem;font-family:"DM Mono",monospace;font-size:0.72rem;color:#fde68a;
     margin-top:0.5rem'>
  ⬆️ No default config found.<br>Drop your <b>portfolio_config.json</b> above to get started.
</div>""", unsafe_allow_html=True)
            st.stop()

    # ── Validation ────────────────────────────────────────────────────────────
    errors = validate_config(cfg)
    if errors:
        st.markdown(f"""
<div style='background:#2d1515;border:1px solid #7f1d1d;border-radius:8px;
     padding:0.8rem;font-family:"DM Mono",monospace;font-size:0.72rem;color:#fca5a5;
     margin-top:0.5rem'>
  ❌ <b>Config errors</b><br>{'<br>'.join(f'· {e}' for e in errors)}
</div>""", unsafe_allow_html=True)
        st.stop()

    # ── Config loaded OK ──────────────────────────────────────────────────────
    st.markdown(f"""
<div style='background:#0f2d1a;border:1px solid #14532d;border-radius:6px;
     padding:0.4rem 0.7rem;font-family:"DM Mono",monospace;font-size:0.68rem;
     color:#86efac;margin-bottom:0.5rem'>
  ✅ {config_source}
</div>""", unsafe_allow_html=True)

    settings  = cfg["settings"]
    portfolio = cfg["portfolio"]
    watchlist = cfg["watchlist"]

    period_opts = {f"{d}D": d for d in settings["lookback_periods"]}
    sel_period  = st.selectbox("Lookback period", list(period_opts.keys()),
                               index=list(period_opts.values()).index(settings["default_lookback_days"]))
    lookback_days = period_opts[sel_period]

    threshold = st.slider("Underperformance threshold (vs benchmark, %)",
                           min_value=-20, max_value=0,
                           value=int(settings["underperformance_threshold_pct"]), step=1)

    st.markdown("---")
    st.markdown(f"**Portfolio:** {portfolio['name']}")
    st.markdown(f"**Benchmark:** {portfolio['benchmark_name']}")
    st.markdown(f"**Holdings:** {len(portfolio['holdings'])} stocks")
    st.markdown(f"**Watchlist:** {len(watchlist)} stocks")
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.markdown("**Indicator Legend**")
    st.markdown("""
<div style='font-family:"DM Mono",monospace;font-size:0.68rem;color:#5c667a;line-height:1.8'>
RSI &lt;30 = Oversold · &gt;70 = Overbought<br>
MACD histogram direction = momentum<br>
Golden Cross = 50MA crosses above 200MA<br>
Death Cross = 50MA crosses below 200MA<br>
VOL = last day vs 20-day avg volume<br>
β = beta vs benchmark<br>
ATR% = avg true range as % of price
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
all_tickers = ([h["ticker"] for h in portfolio["holdings"]] +
               [w["ticker"] for w in watchlist] +
               [portfolio["benchmark"]])

with st.spinner("Fetching market data & computing indicators…"):
    ohlcv_data   = fetch_ohlcv(all_tickers, lookback_days)
    current_info = fetch_current_info(all_tickers)

bench_ticker = portfolio["benchmark"]
bench_full   = get_close(ohlcv_data, bench_ticker)              # full history
bench_period = get_close(ohlcv_data, bench_ticker, lookback_days)
bench_return = calc_return(bench_period)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE INDICATORS FOR EVERY TICKER
# ══════════════════════════════════════════════════════════════════════════════
indicators = {}
for h in portfolio["holdings"] + watchlist:
    t   = h["ticker"]
    df  = ohlcv_data.get(t)
    if df is None or df.empty:
        indicators[t] = {}
        continue
    close  = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None

    rsi_s          = calc_rsi(close)
    macd_l, macd_sig_l, macd_hist = calc_macd(close)
    bb_upper, bb_mid, bb_lower    = calc_bollinger(close)
    ma50               = calc_ma(close, 50)
    ma200              = calc_ma(close, 200)
    atr_s              = calc_atr(df) if all(c in df.columns for c in ["High","Low"]) else None
    beta_val           = calc_beta(close, bench_full) if bench_full is not None else None
    rsc_s              = calc_rsc(close, bench_full)  if bench_full is not None else None
    vol_ratio          = calc_volume_ratio(volume)    if volume is not None else None
    ma_cross           = detect_ma_cross(close)
    macd_sig_label_val = macd_signal_label(macd_hist)

    rsi_now  = float(rsi_s.dropna().iloc[-1])  if rsi_s  is not None and not rsi_s.dropna().empty  else None
    atr_now  = float(atr_s.dropna().iloc[-1])  if atr_s  is not None and not atr_s.dropna().empty  else None
    price_now = float(close.iloc[-1])
    atr_pct  = (atr_now / price_now * 100) if (atr_now and price_now) else None

    day_chg  = (current_info.get(t) or {}).get("day_change_pct")
    comp     = composite_score(rsi_now, macd_sig_label_val, ma_cross, vol_ratio, day_chg)

    indicators[t] = {
        "rsi_series": rsi_s, "rsi": rsi_now,
        "macd_line": macd_l, "macd_signal": macd_sig_l, "macd_hist": macd_hist,
        "macd_label": macd_sig_label_val,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "ma50": ma50, "ma200": ma200,
        "atr": atr_now, "atr_pct": atr_pct,
        "beta": beta_val,
        "rsc": rsc_s,
        "vol_ratio": vol_ratio,
        "ma_cross": ma_cross,
        "composite": comp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO METRICS
# ══════════════════════════════════════════════════════════════════════════════
portfolio_values = []
for h in portfolio["holdings"]:
    t    = h["ticker"]
    info = current_info.get(t, {})
    cp   = info.get("current_price")
    if cp:
        per_ret = calc_return(get_close(ohlcv_data, t, lookback_days))
        tot_ret = ((cp - h["avg_cost"]) / h["avg_cost"]) * 100
        portfolio_values.append({
            "ticker": t, "name": h["name"], "sector": h.get("sector","—"),
            "shares": h["shares"], "avg_cost": h["avg_cost"],
            "current_val": cp * h["shares"], "cost_basis": h["avg_cost"] * h["shares"],
            "period_return": per_ret, "total_return": tot_ret,
        })

total_port_val  = sum(p["current_val"]  for p in portfolio_values)
total_cost      = sum(p["cost_basis"]   for p in portfolio_values)
total_gain_pct  = ((total_port_val - total_cost) / total_cost * 100) if total_cost else 0
valid_rets      = [p["period_return"] for p in portfolio_values if p["period_return"] is not None]
avg_port_ret    = float(np.mean(valid_rets)) if valid_rets else None

underperformers = [
    pv["ticker"] for pv in portfolio_values
    if pv["period_return"] is not None and bench_return is not None
    and (pv["period_return"] - bench_return) < threshold
]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sentinel-title">Portfolio Sentinel</div>', unsafe_allow_html=True)
st.markdown('<div class="sentinel-sub">Performance · Technicals · Rebalancing</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
diff_vs_bench = ((avg_port_ret - bench_return) if (avg_port_ret and bench_return) else None)
with c1: st.metric("Portfolio Value",   f"${total_port_val:,.0f}",        delta=f"{total_gain_pct:+.2f}% total")
with c2: st.metric(portfolio["benchmark_name"], fmt_pct(bench_return),    delta=f"{lookback_days}D period")
with c3: st.metric("Avg Portfolio Ret", fmt_pct(avg_port_ret),            delta=f"{fmt_pct(diff_vs_bench)} vs BM")
with c4: st.metric("Underperformers",   str(len(underperformers)),         delta=f"of {len(portfolio['holdings'])} holdings")
# Count stocks with bearish composite score
bearish_count = sum(1 for h in portfolio["holdings"]
                    if indicators.get(h["ticker"], {}).get("composite", 0) < -1)
with c5: st.metric("Bearish Tech Signal", str(bearish_count),             delta="composite score < -1")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Performance", "🚨 Flags & Actions", "📡 Technicals", "📈 Watchlist", "📊 Analytics"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-header">Normalised Performance vs Benchmark</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if bench_period is not None:
            fig.add_trace(go.Scatter(x=bench_period.index, y=normalize(bench_period),
                name=portfolio["benchmark_name"],
                line=dict(color="#4c6ef5", width=2.5, dash="dot")))

        cscale = px.colors.qualitative.Plotly
        for i, h in enumerate(portfolio["holdings"]):
            s = get_close(ohlcv_data, h["ticker"], lookback_days)
            if s is None: continue
            ret  = calc_return(s)
            diff = (ret - bench_return) if (ret and bench_return) else None
            is_under = diff is not None and diff < threshold
            fig.add_trace(go.Scatter(
                x=s.index, y=normalize(s), name=h["ticker"],
                line=dict(color="#f87171" if is_under else cscale[i % len(cscale)],
                          width=2.2 if is_under else 1.5),
                opacity=1.0 if is_under else 0.7,
                hovertemplate=f"%{{y:.1f}}<extra>{h['ticker']}</extra>"))

        fig.update_layout(**CHART_LAYOUT, height=420,
                          yaxis_title="Indexed (base=100)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Holdings — Sorted by Period Return</div>', unsafe_allow_html=True)
        for pv in sorted(portfolio_values, key=lambda x: (x["period_return"] or -999)):
            ret   = pv["period_return"]
            diff  = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            comp  = indicators.get(pv["ticker"], {}).get("composite", 0)
            is_un = diff is not None and diff < threshold
            flag_cls = "flag-red" if is_un else ("flag-yellow" if (diff is not None and diff < 0) else "flag-green")
            scls, slbl = score_label(comp)
            st.markdown(f"""
            <div style="background:#13161e;border:1px solid #1e2330;border-radius:10px;
                        padding:0.7rem 1rem;margin-bottom:0.4rem;
                        border-left:3px solid {'#f87171' if is_un else ('#fbbf24' if (diff and diff<0) else '#4ade80')}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span class="ticker-badge">{pv['ticker']}</span>
                <span class="{color_class(ret)}" style="font-weight:700;font-size:0.9rem">{fmt_pct(ret)}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:4px">
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5c667a">
                  vs BM: <span class="{color_class(diff)}">{fmt_pct(diff)}</span>
                </span>
                <span class="score-card {scls}" style="font-size:0.62rem;padding:1px 6px">{slbl}</span>
              </div>
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — FLAGS & ACTIONS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    def build_flag_card(pv, color_key, ind):
        t    = pv["ticker"]
        ret  = pv["period_return"]
        diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
        cp   = (current_info.get(t) or {}).get("current_price")
        comp = ind.get("composite", 0)
        scls, slbl = score_label(comp)
        badges = indicator_badges_html(
            ind.get("rsi"), ind.get("macd_label"), ind.get("ma_cross"),
            ind.get("vol_ratio"), ind.get("beta"), ind.get("atr_pct"))
        color_map = {"red":"#fca5a5","yellow":"#fde68a","blue":"#93c5fd","green":"#86efac"}
        tc = color_map.get(color_key,"#94a3b8")
        alert_cls = f"alert-{color_key}"
        return f"""
        <div class="{alert_cls}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <span class="ticker-badge" style="color:{tc}">{t}</span>
              <span style="font-size:0.8rem;color:{tc};margin-left:8px;font-weight:600">{pv['name']}</span>
              <span class="score-card {scls}" style="font-size:0.65rem">{slbl}</span>
            </div>
            <span style="font-family:'DM Mono',monospace;font-size:0.85rem;font-weight:700"
                  class="{color_class(ret)}">{fmt_pct(ret)}</span>
          </div>
          <div style="margin-top:0.35rem;font-family:'DM Mono',monospace;font-size:0.72rem;color:#9ca3af">
            vs BM: <span class="{color_class(diff)}">{fmt_pct(diff)}</span>
            &nbsp;·&nbsp; Price: {'$'+f'{cp:.2f}' if cp else 'N/A'}
            &nbsp;·&nbsp; {pv['sector']}
            &nbsp;·&nbsp; Total P&L: <span class="{color_class(pv['total_return'])}">{fmt_pct(pv['total_return'])}</span>
          </div>
          <div style="margin-top:0.4rem">{badges}</div>
        </div>"""

    with col_a:
        st.markdown('<div class="section-header">🔴 Remove — Underperforming Holdings</div>', unsafe_allow_html=True)
        red_flags, yellow_flags = [], []
        for pv in portfolio_values:
            ret  = pv["period_return"]
            diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            if diff is None: continue
            if diff < threshold:   red_flags.append(pv)
            elif diff < 0:         yellow_flags.append(pv)

        if not red_flags:
            st.markdown('<div class="alert-green"><b>✅ No critical underperformers</b></div>', unsafe_allow_html=True)
        else:
            for pv in sorted(red_flags, key=lambda x: x["period_return"] or 0):
                st.markdown(build_flag_card(pv, "red", indicators.get(pv["ticker"], {})), unsafe_allow_html=True)

        st.markdown('<div class="section-header">🟡 Watch — Slightly Lagging</div>', unsafe_allow_html=True)
        if not yellow_flags:
            st.markdown('<div class="alert-green"><b>✅ No lagging holdings</b></div>', unsafe_allow_html=True)
        else:
            for pv in sorted(yellow_flags, key=lambda x: x["period_return"] or 0):
                st.markdown(build_flag_card(pv, "yellow", indicators.get(pv["ticker"], {})), unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">🟢 Strong Performers</div>', unsafe_allow_html=True)
        greens = [pv for pv in portfolio_values
                  if (pv["period_return"] is not None and bench_return is not None
                      and pv["period_return"] - bench_return >= 0)]
        if greens:
            for pv in sorted(greens, key=lambda x: -(x["period_return"] or 0))[:8]:
                st.markdown(build_flag_card(pv, "green", indicators.get(pv["ticker"], {})), unsafe_allow_html=True)

        # Action summary table
        st.markdown('<div class="section-header">📋 Action Summary Table</div>', unsafe_allow_html=True)
        rows = []
        for pv in portfolio_values:
            ret  = pv["period_return"]
            diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            ind  = indicators.get(pv["ticker"], {})
            comp = ind.get("composite", 0)
            if diff is None:      perf_action = "Insufficient Data"
            elif diff < threshold:perf_action = "⛔ REMOVE"
            elif diff < 0:        perf_action = "⚠️ MONITOR"
            else:                 perf_action = "✅ HOLD"
            _, tech_action = score_label(comp)
            rows.append({
                "Ticker": pv["ticker"], "Period Ret": fmt_pct(ret),
                "vs BM": fmt_pct(diff), "RSI": f"{ind.get('rsi'):.0f}" if ind.get('rsi') else "N/A",
                "MA Cross": ind.get("ma_cross","—").replace("_"," ").upper(),
                "β": f"{ind.get('beta'):.2f}" if ind.get('beta') else "N/A",
                "Perf Signal": perf_action, "Tech Signal": tech_action,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — TECHNICALS (deep dive)
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    all_tickers_ui = (
        [h["ticker"] for h in portfolio["holdings"]] +
        [w["ticker"] for w in watchlist]
    )
    ticker_names = {h["ticker"]: h["name"] for h in portfolio["holdings"] + watchlist}

    tc1, tc2 = st.columns([1, 3])
    with tc1:
        sel_ticker = st.selectbox("Select stock",
                                  all_tickers_ui,
                                  format_func=lambda t: f"{t} — {ticker_names.get(t,'')}")
        chart_period = st.selectbox("Chart period", ["3M","6M","1Y","All"], index=1)
    with tc2:
        ind = indicators.get(sel_ticker, {})
        comp = ind.get("composite", 0)
        scls, slbl = score_label(comp)
        rsi_v   = ind.get("rsi")
        beta_v  = ind.get("beta")
        atr_pct = ind.get("atr_pct")
        vol_r   = ind.get("vol_ratio")

        b1,b2,b3,b4,b5,b6 = st.columns(6)
        with b1: st.metric("RSI (14)", f"{rsi_v:.1f}" if rsi_v else "N/A",
                            delta="Overbought" if rsi_v and rsi_v>70 else ("Oversold" if rsi_v and rsi_v<30 else "Neutral"))
        with b2: st.metric("MACD Signal", ind.get("macd_label","N/A").replace("_"," ").title())
        with b3: st.metric("MA Cross", ind.get("ma_cross","N/A").replace("_"," ").title())
        with b4: st.metric("Beta (β)", f"{beta_v:.2f}" if beta_v else "N/A")
        with b5: st.metric("ATR%", f"{atr_pct:.1f}%" if atr_pct else "N/A")
        with b6: st.metric("Vol Ratio", f"{vol_r:.2f}x" if vol_r else "N/A")

    # Chart period lookup
    cp_days = {"3M": 90, "6M": 180, "1Y": 365, "All": 9999}[chart_period]
    df_full = ohlcv_data.get(sel_ticker)

    if df_full is not None and not df_full.empty:
        if cp_days < 9999:
            cutoff = pd.Timestamp(datetime.today() - timedelta(days=cp_days))
            df_chart = df_full[df_full.index >= cutoff]
        else:
            df_chart = df_full

        close = df_full["Close"]   # full series for indicator calcs
        c_idx = df_chart.index     # display window index

        rsi_s     = ind.get("rsi_series")
        macd_l    = ind.get("macd_line")
        macd_sl   = ind.get("macd_signal")
        macd_hs   = ind.get("macd_hist")
        bb_up     = ind.get("bb_upper")
        bb_mi     = ind.get("bb_mid")
        bb_lo     = ind.get("bb_lower")
        ma50_s    = ind.get("ma50")
        ma200_s   = ind.get("ma200")
        rsc_s     = ind.get("rsc")

        # ── 3-panel chart: Price+BB+MA | MACD | RSI ──
        fig_tech = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.03,
            subplot_titles=("Price · Bollinger Bands · Moving Averages", "MACD", "RSI (14)"),
        )

        # Panel 1: Price
        ohlc_chart = df_chart
        fig_tech.add_trace(go.Candlestick(
            x=ohlc_chart.index,
            open=ohlc_chart["Open"], high=ohlc_chart["High"],
            low=ohlc_chart["Low"],  close=ohlc_chart["Close"],
            name="Price",
            increasing_line_color="#4ade80", decreasing_line_color="#f87171",
            increasing_fillcolor="#4ade80",  decreasing_fillcolor="#f87171",
        ), row=1, col=1)

        # Bollinger Bands
        for s, name, color, fill in [
            (bb_up.reindex(c_idx), "BB Upper", "#60a5fa", "tonexty"),
            (bb_mi.reindex(c_idx), "BB Mid",   "#93c5fd", None),
            (bb_lo.reindex(c_idx), "BB Lower", "#60a5fa", None),
        ]:
            if s is not None:
                kw = dict(fill=fill, fillcolor="rgba(96,165,250,0.06)") if fill else {}
                fig_tech.add_trace(go.Scatter(x=s.index, y=s, name=name,
                    line=dict(color=color, width=1, dash="dot"), **kw), row=1, col=1)

        # MAs
        for ma_s, lbl, col in [(ma50_s, "MA 50", "#fbbf24"), (ma200_s, "MA 200", "#f472b6")]:
            if ma_s is not None:
                ms = ma_s.reindex(c_idx).dropna()
                if not ms.empty:
                    fig_tech.add_trace(go.Scatter(x=ms.index, y=ms, name=lbl,
                        line=dict(color=col, width=1.5)), row=1, col=1)

        # Panel 2: MACD
        if macd_l is not None:
            ml = macd_l.reindex(c_idx).dropna()
            ms = macd_sl.reindex(c_idx).dropna()
            mh = macd_hs.reindex(c_idx).dropna()
            colors_hist = ["#4ade80" if v >= 0 else "#f87171" for v in mh]
            fig_tech.add_trace(go.Bar(x=mh.index, y=mh, name="Histogram",
                marker_color=colors_hist, opacity=0.7), row=2, col=1)
            fig_tech.add_trace(go.Scatter(x=ml.index, y=ml, name="MACD",
                line=dict(color="#60a5fa", width=1.5)), row=2, col=1)
            fig_tech.add_trace(go.Scatter(x=ms.index, y=ms, name="Signal",
                line=dict(color="#f472b6", width=1.5)), row=2, col=1)

        # Panel 3: RSI
        if rsi_s is not None:
            rs = rsi_s.reindex(c_idx).dropna()
            if not rs.empty:
                fig_tech.add_trace(go.Scatter(x=rs.index, y=rs, name="RSI",
                    line=dict(color="#a78bfa", width=1.8)), row=3, col=1)
                fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.08)",
                    line_width=0, row=3, col=1)
                fig_tech.add_hrect(y0=0, y1=30, fillcolor="rgba(74,222,128,0.08)",
                    line_width=0, row=3, col=1)
                fig_tech.add_hline(y=70, line_dash="dash", line_color="#f87171",
                    line_width=1, row=3, col=1)
                fig_tech.add_hline(y=30, line_dash="dash", line_color="#4ade80",
                    line_width=1, row=3, col=1)

        fig_tech.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis","yaxis")},
            height=680, showlegend=True,
            xaxis_rangeslider_visible=False,
        )
        for i in range(1, 4):
            fig_tech.update_xaxes(gridcolor="#1e2330", row=i, col=1)
            fig_tech.update_yaxes(gridcolor="#1e2330", row=i, col=1)

        st.plotly_chart(fig_tech, use_container_width=True)

        # ── RSC chart ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Relative Strength Comparative (RSC) vs Benchmark</div>',
                    unsafe_allow_html=True)
        if rsc_s is not None and not rsc_s.empty:
            rsc_d = rsc_s.reindex(c_idx).dropna()
            rsc_norm = normalize(rsc_d) if len(rsc_d) > 1 else rsc_d
            rising   = rsc_norm.iloc[-1] > rsc_norm.iloc[0] if len(rsc_norm) > 1 else None
            rsc_color = "#4ade80" if rising else "#f87171"

            fig_rsc = go.Figure()
            fig_rsc.add_trace(go.Scatter(
                x=rsc_norm.index, y=rsc_norm,
                name=f"{sel_ticker} RSC", fill="tozeroy",
                fillcolor=f"rgba({'74,222,128' if rising else '248,113,113'},0.08)",
                line=dict(color=rsc_color, width=2)))
            fig_rsc.add_hline(y=100, line_dash="dash", line_color="#4c6ef5", line_width=1)
            fig_rsc.update_layout(**CHART_LAYOUT, height=200,
                                  yaxis_title="RSC (indexed)", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_rsc, use_container_width=True)

            trend = "📈 Rising — outperforming benchmark" if rising else "📉 Falling — underperforming benchmark"
            color = "#4ade80" if rising else "#f87171"
            st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:{color}">{trend}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Insufficient data for RSC chart.")

        # ── Volume chart ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Volume vs 20-Day Average</div>', unsafe_allow_html=True)
        if "Volume" in df_chart.columns:
            vol_s  = df_chart["Volume"]
            vol_ma = vol_s.rolling(20).mean()
            vol_colors = ["#4ade80" if c >= o else "#f87171"
                          for c, o in zip(df_chart["Close"], df_chart["Open"])]
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=vol_s.index, y=vol_s, name="Volume",
                marker_color=vol_colors, opacity=0.7))
            fig_vol.add_trace(go.Scatter(x=vol_ma.index, y=vol_ma, name="20D Avg Volume",
                line=dict(color="#fbbf24", width=2)))
            fig_vol.update_layout(**CHART_LAYOUT, height=180, yaxis_title="Volume",
                                  xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning(f"No OHLCV data available for {sel_ticker}.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — WATCHLIST
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    wl_col1, wl_col2 = st.columns([1.4, 1])

    with wl_col1:
        st.markdown('<div class="section-header">Watchlist Performance vs Benchmark</div>', unsafe_allow_html=True)
        fig_wl = go.Figure()
        if bench_period is not None:
            fig_wl.add_trace(go.Scatter(x=bench_period.index, y=normalize(bench_period),
                name=portfolio["benchmark_name"],
                line=dict(color="#4c6ef5", width=2.5, dash="dot")))
        wl_colors = ["#a78bfa","#34d399","#fb923c","#f472b6","#38bdf8"]
        for i, w in enumerate(watchlist):
            s = get_close(ohlcv_data, w["ticker"], lookback_days)
            if s is not None and len(s) > 1:
                fig_wl.add_trace(go.Scatter(x=s.index, y=normalize(s), name=w["ticker"],
                    line=dict(color=wl_colors[i % len(wl_colors)], width=2)))
        fig_wl.update_layout(**CHART_LAYOUT, height=380, yaxis_title="Indexed (base=100)")
        st.plotly_chart(fig_wl, use_container_width=True)

    with wl_col2:
        st.markdown('<div class="section-header">Watchlist Detail + Indicators</div>', unsafe_allow_html=True)
        for w in watchlist:
            t    = w["ticker"]
            s    = get_close(ohlcv_data, t, lookback_days)
            ret  = calc_return(s)
            info = current_info.get(t, {})
            cp   = info.get("current_price")
            day  = info.get("day_change_pct")
            diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            ind  = indicators.get(t, {})
            comp = ind.get("composite", 0)
            scls, slbl = score_label(comp)
            badges = indicator_badges_html(
                ind.get("rsi"), ind.get("macd_label"), ind.get("ma_cross"),
                ind.get("vol_ratio"), ind.get("beta"), ind.get("atr_pct"))
            beats = diff is not None and diff >= 0
            add_sig = "✅ Strong Add" if (diff is not None and diff > 5 and comp > 0) else \
                      "🟡 Consider" if (diff is not None and diff >= 0) else "⚠️ Wait"
            st.markdown(f"""
            <div class="alert-blue" style="margin-bottom:0.6rem">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span class="ticker-badge" style="color:#93c5fd">{t}</span>
                  <span style="font-size:0.78rem;color:#93c5fd;margin-left:8px;font-weight:600">{w['name']}</span>
                  <span class="score-card {scls}" style="font-size:0.62rem">{slbl}</span>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:0.85rem;font-weight:700"
                      class="{color_class(ret)}">{fmt_pct(ret)}</span>
              </div>
              <div style="margin-top:0.3rem;font-family:'DM Mono',monospace;font-size:0.7rem;color:#9ca3af">
                {w['sector']} · {'$'+f'{cp:.2f}' if cp else 'N/A'}
                · Day: <span class="{color_class(day)}">{fmt_pct(day,1)}</span>
                · {'<b style="color:#4ade80">Beats BM</b>' if beats else '<span style="color:#f87171">Lags BM</span>'}
                · {add_sig}
              </div>
              <div style="margin-top:0.35rem;font-size:0.74rem;color:#60a5fa;font-style:italic">
                💡 {w['reason']}
              </div>
              <div style="margin-top:0.4rem">{badges}</div>
            </div>""", unsafe_allow_html=True)

    # Watchlist comparison table
    st.markdown('<div class="section-header">Watchlist Indicator Summary</div>', unsafe_allow_html=True)
    wl_rows = []
    for w in watchlist:
        t    = w["ticker"]
        s    = get_close(ohlcv_data, t, lookback_days)
        ret  = calc_return(s)
        info = current_info.get(t, {})
        diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
        ind  = indicators.get(t, {})
        comp = ind.get("composite", 0)
        _, slbl = score_label(comp)
        wl_rows.append({
            "Ticker": t, "Name": w["name"], "Sector": w["sector"],
            f"{lookback_days}D Ret": fmt_pct(ret), "vs BM": fmt_pct(diff),
            "RSI": f"{ind.get('rsi'):.0f}" if ind.get('rsi') else "N/A",
            "MACD": ind.get("macd_label","—").replace("_"," ").title(),
            "MA Cross": ind.get("ma_cross","—").replace("_"," ").title(),
            "β": f"{ind.get('beta'):.2f}" if ind.get('beta') else "N/A",
            "ATR%": f"{ind.get('atr_pct'):.1f}%" if ind.get('atr_pct') else "N/A",
            "Signal": slbl,
        })
    st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown('<div class="section-header">Returns vs Benchmark</div>', unsafe_allow_html=True)
        bar_data = []
        for pv in portfolio_values:
            ret  = pv["period_return"]
            diff = (ret - bench_return) if (ret is not None and bench_return is not None) else None
            bar_data.append({"Ticker": pv["ticker"], "diff": diff or 0})
        bar_df = pd.DataFrame(bar_data).sort_values("diff")
        colors = ["#f87171" if v < threshold else ("#fbbf24" if v < 0 else "#4ade80") for v in bar_df["diff"]]
        fig_bar = go.Figure(go.Bar(x=bar_df["Ticker"], y=bar_df["diff"],
            marker_color=colors, hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        fig_bar.add_hline(y=threshold, line_dash="dash", line_color="#f87171",
                          annotation_text=f"Threshold ({threshold}%)")
        fig_bar.add_hline(y=0, line_color="#4c6ef5", line_width=1)
        fig_bar.update_layout(**CHART_LAYOUT, height=340, yaxis_title="% vs Benchmark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with a2:
        st.markdown('<div class="section-header">Portfolio Allocation</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=[pv["ticker"] for pv in portfolio_values],
            values=[pv["current_val"] for pv in portfolio_values],
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Bold, line=dict(color="#0d0f14", width=2)),
            textfont=dict(family="DM Mono", size=11),
            hovertemplate="%{label}: $%{value:,.0f} (%{percent})<extra></extra>"))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="DM Mono", color="#94a3b8"),
                              legend=dict(bgcolor="rgba(0,0,0,0)"),
                              margin=dict(l=0,r=0,t=0,b=0), height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    # RSI scatter — period return vs RSI (show risk/reward quadrants)
    st.markdown('<div class="section-header">RSI vs Period Return — Quadrant View</div>', unsafe_allow_html=True)
    scatter_data = []
    for pv in portfolio_values:
        ind = indicators.get(pv["ticker"], {})
        rsi_v = ind.get("rsi")
        if rsi_v and pv["period_return"] is not None:
            scatter_data.append({
                "ticker": pv["ticker"], "rsi": rsi_v,
                "return": pv["period_return"],
                "beta": ind.get("beta") or 1.0,
                "sector": pv["sector"],
            })
    if scatter_data:
        sc_df = pd.DataFrame(scatter_data)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sc_df["rsi"], y=sc_df["return"],
            mode="markers+text",
            text=sc_df["ticker"], textposition="top center",
            marker=dict(size=sc_df["beta"].abs() * 12 + 6,
                        color=sc_df["return"],
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="Return %", thickness=10)),
            hovertemplate="<b>%{text}</b><br>RSI: %{x:.1f}<br>Return: %{y:.1f}%<extra></extra>"))
        fig_sc.add_vline(x=70, line_dash="dash", line_color="#f87171", line_width=1,
                         annotation_text="Overbought", annotation_font_color="#f87171")
        fig_sc.add_vline(x=30, line_dash="dash", line_color="#4ade80", line_width=1,
                         annotation_text="Oversold", annotation_font_color="#4ade80")
        fig_sc.add_hline(y=bench_return or 0, line_dash="dash", line_color="#4c6ef5",
                         line_width=1, annotation_text="Benchmark")
        fig_sc.update_layout(**CHART_LAYOUT, height=380,
                             xaxis_title="RSI (14)", yaxis_title="Period Return %")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Beta vs ATR scatter
    st.markdown('<div class="section-header">Beta vs Volatility (ATR%) — Risk Profile</div>', unsafe_allow_html=True)
    risk_data = []
    for pv in portfolio_values:
        ind = indicators.get(pv["ticker"], {})
        b   = ind.get("beta")
        a   = ind.get("atr_pct")
        if b is not None and a is not None:
            risk_data.append({"ticker": pv["ticker"], "beta": b, "atr_pct": a,
                               "return": pv["period_return"] or 0, "sector": pv["sector"]})
    if risk_data:
        rk_df = pd.DataFrame(risk_data)
        fig_risk = go.Figure(go.Scatter(
            x=rk_df["beta"], y=rk_df["atr_pct"],
            mode="markers+text",
            text=rk_df["ticker"], textposition="top center",
            marker=dict(size=12, color=rk_df["return"], colorscale="RdYlGn",
                        showscale=True, colorbar=dict(title="Return %", thickness=10)),
            hovertemplate="<b>%{text}</b><br>β: %{x:.2f}<br>ATR%: %{y:.1f}%<extra></extra>"))
        fig_risk.add_vline(x=1.0, line_dash="dash", line_color="#4c6ef5", line_width=1,
                           annotation_text="β=1")
        fig_risk.update_layout(**CHART_LAYOUT, height=360,
                               xaxis_title="Beta (β)", yaxis_title="ATR %")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Full holdings table with all indicators
    st.markdown('<div class="section-header">Full Holdings — All Indicators</div>', unsafe_allow_html=True)
    full_rows = []
    for h in portfolio["holdings"]:
        t    = h["ticker"]
        info = current_info.get(t, {})
        cp   = info.get("current_price")
        day  = info.get("day_change_pct")
        pv   = next((p for p in portfolio_values if p["ticker"] == t), None)
        ind  = indicators.get(t, {})
        _, slbl = score_label(ind.get("composite", 0))
        full_rows.append({
            "Ticker": t, "Sector": h["sector"],
            "Price": f"${cp:.2f}"   if cp  else "N/A",
            "Day":   fmt_pct(day,1),
            "Period":fmt_pct(pv["period_return"] if pv else None),
            "Total": fmt_pct(pv["total_return"]  if pv else None),
            "RSI":   f"{ind.get('rsi'):.0f}" if ind.get("rsi") else "N/A",
            "MACD":  ind.get("macd_label","—").replace("_"," ").title(),
            "MA":    ind.get("ma_cross","—").replace("_"," ").title(),
            "β":     f"{ind.get('beta'):.2f}"    if ind.get("beta")    else "N/A",
            "ATR%":  f"{ind.get('atr_pct'):.1f}%" if ind.get("atr_pct") else "N/A",
            "Vol×":  f"{ind.get('vol_ratio'):.1f}×" if ind.get("vol_ratio") else "N/A",
            "Signal":slbl,
        })
    st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)

    # Sector breakdown
    st.markdown('<div class="section-header">Sector Breakdown</div>', unsafe_allow_html=True)
    sector_map = {}
    for pv in portfolio_values:
        s = pv["sector"]
        sector_map.setdefault(s, {"value":0,"tickers":[]})
        sector_map[s]["value"]   += pv["current_val"]
        sector_map[s]["tickers"].append(pv["ticker"])
    sec_rows = [{"Sector":k,"Value":f"${v['value']:,.0f}",
                 "Weight":f"{v['value']/total_port_val*100:.1f}%",
                 "Tickers":", ".join(v["tickers"])}
                for k,v in sorted(sector_map.items(), key=lambda x:-x[1]["value"])]
    st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#2d3748;text-align:center">'
    f'Portfolio Sentinel · Indicators: RSI · MACD · Bollinger Bands · MA Cross · RSC · Beta · ATR · Volume '
    f'· Data via Yahoo Finance · {datetime.now().strftime("%Y-%m-%d %H:%M")} '
    f'· For informational purposes only — not financial advice.'
    f'</div>', unsafe_allow_html=True)
