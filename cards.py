"""
cards.py
────────
HTML card / badge rendering for flag cards and watchlist cards.
All visual layout is here; no business logic.

The `enabled_badges` parameter is a set of indicator keys (strings)
that controls which badge pills appear on each card.
The caller (sidebar) owns that set — cards.py never touches st.session_state.
"""
from __future__ import annotations
from indicators import badge_html


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fmt_pct(v: float | None, d: int = 2) -> str:
    if v is None: return "N/A"
    return f"+{v:.{d}f}%" if v >= 0 else f"{v:.{d}f}%"


def _color_cls(v: float | None) -> str:
    if v is None: return "neutral"
    return "positive" if v >= 0 else "negative"


def _score_label(score: int) -> tuple[str, str]:
    """(css_class, label)"""
    if score <= -3: return "score-strong-sell", "⛔ STRONG SELL"
    if score == -2: return "score-sell",         "🔴 SELL"
    if score == -1: return "score-sell",         "🟠 LEAN SELL"
    if score ==  0: return "score-neutral",      "⚪ NEUTRAL"
    if score ==  1: return "score-buy",          "🟡 LEAN BUY"
    if score ==  2: return "score-buy",          "🟢 BUY"
    return               "score-strong-buy",     "✅ STRONG BUY"


# ── Flag card (used in Tab 2) ─────────────────────────────────────────────────

def flag_card_html(
    ticker:         str,
    name:           str,
    sector:         str,
    period_return:  float | None,
    bench_return:   float | None,
    total_return:   float | None,
    current_price:  float | None,
    ind:            dict,
    alert_color:    str,        # "red" | "yellow" | "green" | "blue"
    enabled_badges: set[str],
) -> str:
    diff       = ((period_return - bench_return)
                  if period_return is not None and bench_return is not None
                  else None)
    comp       = ind.get("composite", 0)
    scls, slbl = _score_label(comp)
    badges     = badge_html(ind, enabled_badges)
    tc_map     = {"red": "#fca5a5", "yellow": "#fde68a",
                  "green": "#86efac", "blue": "#93c5fd"}
    tc         = tc_map.get(alert_color, "#94a3b8")
    price_str  = f"${current_price:.2f}" if current_price else "N/A"

    return f"""
<div class="alert-{alert_color}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="ticker-badge" style="color:{tc}">{ticker}</span>
      <span style="font-size:0.8rem;color:{tc};margin-left:8px;font-weight:600">{name}</span>
      <span class="score-card {scls}" style="font-size:0.65rem">{slbl}</span>
    </div>
    <span style="font-family:'DM Mono',monospace;font-size:0.85rem;font-weight:700"
          class="{_color_cls(period_return)}">{_fmt_pct(period_return)}</span>
  </div>
  <div style="margin-top:0.35rem;font-family:'DM Mono',monospace;font-size:0.72rem;color:#9ca3af">
    vs BM: <span class="{_color_cls(diff)}">{_fmt_pct(diff)}</span>
    &nbsp;·&nbsp; Price: {price_str}
    &nbsp;·&nbsp; {sector}
    &nbsp;·&nbsp; Total P&L: <span class="{_color_cls(total_return)}">{_fmt_pct(total_return)}</span>
  </div>
  {f'<div style="margin-top:0.4rem">{badges}</div>' if badges else ''}
</div>"""


# ── Holdings glance card (used in Tab 1 right column) ─────────────────────────

def holdings_glance_card_html(
    ticker:        str,
    period_return: float | None,
    bench_return:  float | None,
    threshold:     float,
    comp_score:    int,
) -> str:
    diff    = ((period_return - bench_return)
               if period_return is not None and bench_return is not None else None)
    is_un   = diff is not None and diff < threshold
    border  = ("#f87171" if is_un
               else ("#fbbf24" if (diff is not None and diff < 0) else "#4ade80"))
    scls, slbl = _score_label(comp_score)

    return f"""
<div style="background:#13161e;border:1px solid #1e2330;border-radius:10px;
            padding:0.7rem 1rem;margin-bottom:0.4rem;border-left:3px solid {border}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span class="ticker-badge">{ticker}</span>
    <span class="{_color_cls(period_return)}" style="font-weight:700;font-size:0.9rem">
      {_fmt_pct(period_return)}
    </span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:4px">
    <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5c667a">
      vs BM: <span class="{_color_cls(diff)}">{_fmt_pct(diff)}</span>
    </span>
    <span class="score-card {scls}" style="font-size:0.62rem;padding:1px 6px">{slbl}</span>
  </div>
</div>"""


# ── Watchlist card (used in Tab 4) ────────────────────────────────────────────

def watchlist_card_html(
    ticker:         str,
    name:           str,
    sector:         str,
    reason:         str,
    period_return:  float | None,
    bench_return:   float | None,
    current_price:  float | None,
    day_change_pct: float | None,
    ind:            dict,
    enabled_badges: set[str],
) -> str:
    diff       = ((period_return - bench_return)
                  if period_return is not None and bench_return is not None else None)
    beats      = diff is not None and diff >= 0
    comp       = ind.get("composite", 0)
    scls, slbl = _score_label(comp)
    badges     = badge_html(ind, enabled_badges)

    add_sig = (
        "✅ Strong Add"  if (diff is not None and diff > 5 and comp > 0) else
        "🟡 Consider"    if (diff is not None and diff >= 0) else
        "⚠️ Wait"
    )

    bm_html = ('<b style="color:#4ade80">Beats BM</b>'
               if beats else '<span style="color:#f87171">Lags BM</span>')
    price_str = f"${current_price:.2f}" if current_price else "N/A"
    day_str   = _fmt_pct(day_change_pct, 1)

    return f"""
<div class="alert-blue" style="margin-bottom:0.6rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="ticker-badge" style="color:#93c5fd">{ticker}</span>
      <span style="font-size:0.78rem;color:#93c5fd;margin-left:8px;font-weight:600">{name}</span>
      <span class="score-card {scls}" style="font-size:0.62rem">{slbl}</span>
    </div>
    <span style="font-family:'DM Mono',monospace;font-size:0.85rem;font-weight:700"
          class="{_color_cls(period_return)}">{_fmt_pct(period_return)}</span>
  </div>
  <div style="margin-top:0.3rem;font-family:'DM Mono',monospace;font-size:0.7rem;color:#9ca3af">
    {sector} · {price_str}
    · Day: <span class="{_color_cls(day_change_pct)}">{day_str}</span>
    · {bm_html} · {add_sig}
  </div>
  <div style="margin-top:0.35rem;font-size:0.74rem;color:#60a5fa;font-style:italic">
    💡 {reason}
  </div>
  {f'<div style="margin-top:0.4rem">{badges}</div>' if badges else ''}
</div>"""


# ── Inline status banners ─────────────────────────────────────────────────────

def config_ok_banner(source: str) -> str:
    return (
        f'<div style="background:#0f2d1a;border:1px solid #14532d;border-radius:6px;'
        f'padding:0.4rem 0.7rem;font-family:\'DM Mono\',monospace;font-size:0.68rem;'
        f'color:#86efac;margin-bottom:0.5rem">✅ {source}</div>'
    )


def config_error_banner(errors: list[str]) -> str:
    lines = "<br>".join(f"· {e}" for e in errors)
    return (
        f'<div style="background:#2d1515;border:1px solid #7f1d1d;border-radius:8px;'
        f'padding:0.8rem;font-family:\'DM Mono\',monospace;font-size:0.72rem;'
        f'color:#fca5a5;margin-top:0.5rem">❌ <b>Config errors</b><br>{lines}</div>'
    )


def config_missing_banner() -> str:
    return (
        '<div style="background:#2b2007;border:1px solid #78350f;border-radius:8px;'
        'padding:0.8rem;font-family:\'DM Mono\',monospace;font-size:0.72rem;'
        'color:#fde68a;margin-top:0.5rem">'
        '⬆️ No default config found.<br>'
        'Drop your <b>portfolio_config.json</b> above to get started.</div>'
    )
