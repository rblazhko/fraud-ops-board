"""Fraud Ops Board — Streamlit view over synthetic features + rule score.

Run: `make board`  or  `streamlit run board/app.py`
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from board.score import WEIGHTS, precision_at_k, score_frame

ROOT = Path(__file__).resolve().parents[1]
WIDE = ROOT / "data" / "features_wide.parquet"

# Calm ops-desk palette (cool slate, steel accent — not purple / neon)
COLORS = {
    "bg": "#e8ecf0",
    "panel": "#f7f8fa",
    "ink": "#1c2430",
    "muted": "#5a6878",
    "line": "#c5ced8",
    "accent": "#2f5d7a",
    "vip": "#2f6b57",
    "mass": "#6b7280",
    "flag": "#a8443a",
    "series": "#2f5d7a",
    "rate": "#a8443a",
}

st.set_page_config(
    page_title="Fraud Ops Board",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
            color: {COLORS["ink"]};
        }}
        .stApp {{
            background:
                linear-gradient(180deg, #dfe5eb 0%, {COLORS["bg"]} 28%, #edf0f3 100%);
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }}
        h1 {{
            font-weight: 600 !important;
            letter-spacing: -0.02em;
            font-size: 1.65rem !important;
            margin-bottom: 0.15rem !important;
        }}
        .ops-sub {{
            color: {COLORS["muted"]};
            font-size: 0.92rem;
            margin-bottom: 1.1rem;
        }}
        div[data-testid="stMetric"] {{
            background: {COLORS["panel"]};
            border: 1px solid {COLORS["line"]};
            border-radius: 4px;
            padding: 0.65rem 0.85rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {COLORS["muted"]} !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 1.35rem !important;
            font-weight: 500;
        }}
        section[data-testid="stSidebar"] {{
            background: {COLORS["panel"]};
            border-right: 1px solid {COLORS["line"]};
        }}
        section[data-testid="stSidebar"] h2 {{
            font-size: 0.95rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS["muted"]} !important;
        }}
        .ops-section {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["muted"]};
            margin: 1.35rem 0 0.55rem 0;
            border-bottom: 1px solid {COLORS["line"]};
            padding-bottom: 0.35rem;
        }}
        .ops-note {{
            color: {COLORS["muted"]};
            font-size: 0.82rem;
            margin: 0.2rem 0 0.8rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    if not WIDE.exists():
        raise FileNotFoundError(
            f"Missing {WIDE}. Run `make data && make features` first."
        )
    raw = pd.read_parquet(WIDE)
    raw["ts"] = pd.to_datetime(raw["ts"])
    return score_frame(raw)


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{100 * x:.2f}%"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def filter_frame(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    segment: str,
    min_score: float,
) -> pd.DataFrame:
    # inclusive calendar end-of-day
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (df["ts"] >= pd.Timestamp(start)) & (df["ts"] <= end_ts)
    if segment != "all":
        mask &= df["segment"] == segment
    mask &= df["risk_score"] >= min_score
    return df.loc[mask].copy()


def segment_compare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmp = df.assign(_flagged=df["risk_score"] >= 0.35)
    g = tmp.groupby("segment", observed=True)
    out = pd.DataFrame(
        {
            "txs": g.size(),
            "fraud_rate": g["label_fraud"].mean(),
            "avg_amount": g["amount"].mean(),
            "flagged_share": g["_flagged"].mean(),
            "alert_share": g.size() / len(df),
            "avg_score": g["risk_score"].mean(),
        }
    ).reset_index()
    return out.sort_values("segment")


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "volume", "fraud_rate", "flagged_rate"])
    tmp = df.copy()
    tmp["day"] = tmp["ts"].dt.floor("D")
    flagged = tmp["risk_score"] >= 0.35
    g = tmp.groupby("day", observed=True)
    return pd.DataFrame(
        {
            "day": g.size().index,
            "volume": g.size().values,
            "fraud_rate": g["label_fraud"].mean().values,
            "flagged_rate": flagged.groupby(tmp["day"]).mean().values,
        }
    )


def render_kpis(df: pd.DataFrame, full_window: pd.DataFrame) -> None:
    n = len(df)
    fraud_rate = df["label_fraud"].mean() if n else float("nan")
    vip_n = int((df["segment"] == "vip").sum()) if n else 0
    mass_n = int((df["segment"] == "mass").sum()) if n else 0
    vip_share = vip_n / n if n else float("nan")
    p50 = precision_at_k(df, k=50) if n else float("nan")
    alert_load = int((df["risk_score"] >= 0.35).sum()) if n else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Volume", f"{n:,}")
    c2.metric("Fraud rate", _fmt_pct(fraud_rate))
    c3.metric("VIP share", _fmt_pct(vip_share) if n else "—")
    c4.metric("Precision@50", _fmt_pct(p50))
    c5.metric("Alert load", f"{alert_load:,}")
    st.caption(
        f"Window in view: {full_window['ts'].min().date()} → {full_window['ts'].max().date()} "
        f"· score ≥ 0.35 counted as flagged · weights "
        f"v={WEIGHTS['velocity']:.2f} d={WEIGHTS['device']:.2f} "
        f"a={WEIGHTS['amount']:.2f} g={WEIGHTS['geo']:.2f}"
    )


def chart_volume_rates(daily: pd.DataFrame) -> None:
    if daily.empty:
        st.info("No rows for current filters.")
        return
    rates = daily.melt(
        id_vars=["day"],
        value_vars=["fraud_rate", "flagged_rate"],
        var_name="metric",
        value_name="rate",
    )
    rates["metric"] = rates["metric"].map(
        {"fraud_rate": "Fraud", "flagged_rate": "Flagged"}
    )
    bars = (
        alt.Chart(daily)
        .mark_bar(color=COLORS["series"], opacity=0.5)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("volume:Q", title="Volume"),
            tooltip=["day:T", "volume:Q"],
        )
    )
    lines = (
        alt.Chart(rates)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("rate:Q", title="Rate", axis=alt.Axis(format="%")),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["Fraud", "Flagged"],
                    range=[COLORS["rate"], COLORS["accent"]],
                ),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
            tooltip=[
                "day:T",
                "metric:N",
                alt.Tooltip("rate:Q", format=".2%"),
            ],
        )
    )
    layered = (
        alt.layer(bars, lines)
        .resolve_scale(y="independent")
        .properties(height=240)
        .configure_axis(labelColor=COLORS["muted"], titleColor=COLORS["muted"])
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(layered, use_container_width=True)


def chart_score_hist(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows for current filters.")
        return
    hist = (
        alt.Chart(df)
        .mark_bar(color=COLORS["accent"], opacity=0.75)
        .encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=28), title="Risk score"),
            y=alt.Y("count()", title="Txs"),
            tooltip=[alt.Tooltip("count()", title="txs")],
        )
        .properties(height=240)
        .configure_axis(labelColor=COLORS["muted"], titleColor=COLORS["muted"])
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(hist, use_container_width=True)


def render_queue(df: pd.DataFrame) -> None:
    cols = [
        "tx_id",
        "ts",
        "segment",
        "amount",
        "tx_cnt_1h",
        "tx_cnt_24h",
        "shared_flag",
        "device_degree_to_date",
        "device_users_24h",
        "amount_z_user",
        "geo_mismatch",
        "risk_score",
        "label_fraud",
    ]
    show = df.nlargest(min(200, len(df)), "risk_score")[cols].copy()
    show = show.rename(
        columns={
            "tx_cnt_1h": "vel_1h",
            "tx_cnt_24h": "vel_24h",
            "shared_flag": "shared",
            "device_degree_to_date": "dev_degree",
            "device_users_24h": "dev_users_24h",
            "amount_z_user": "amt_z",
            "geo_mismatch": "geo_mm",
            "risk_score": "score",
            "label_fraud": "fraud",
        }
    )
    show["ts"] = show["ts"].dt.strftime("%Y-%m-%d %H:%M")
    show["amount"] = show["amount"].round(2)
    show["score"] = show["score"].round(3)
    show["amt_z"] = show["amt_z"].round(2)
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "score": st.column_config.NumberColumn(format="%.3f"),
            "amount": st.column_config.NumberColumn(format="%.2f"),
            "fraud": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    _inject_css()
    st.title("Fraud Ops Board")
    st.markdown(
        '<p class="ops-sub">Synthetic portfolio · rule score over SQL features '
        "(velocity, device reuse, amount z, geo). Not a production model.</p>",
        unsafe_allow_html=True,
    )

    try:
        scored = load_scored()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    min_d = scored["ts"].min().date()
    max_d = scored["ts"].max().date()

    with st.sidebar:
        st.header("Filters")
        date_range = st.date_input(
            "Date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )
        segment = st.selectbox("Segment", options=["all", "vip", "mass"], index=0)
        min_score = st.slider("Min score", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        st.markdown(
            f'<p class="ops-note">Flagged threshold fixed at 0.35 for '
            f"alert load / flagged rate. Data: <code>{WIDE.name}</code></p>",
            unsafe_allow_html=True,
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    view = filter_frame(scored, start, end, segment, min_score)

    st.markdown('<div class="ops-section">Headline</div>', unsafe_allow_html=True)
    render_kpis(view, scored)

    st.markdown('<div class="ops-section">VIP vs Mass</div>', unsafe_allow_html=True)
    cmp_df = segment_compare(view)
    if cmp_df.empty:
        st.info("No rows for current filters.")
    else:
        display = cmp_df.copy()
        display["fraud_rate"] = display["fraud_rate"].map(_fmt_pct)
        display["flagged_share"] = display["flagged_share"].map(_fmt_pct)
        display["alert_share"] = display["alert_share"].map(_fmt_pct)
        display["avg_amount"] = display["avg_amount"].map(_fmt_num)
        display["avg_score"] = display["avg_score"].map(lambda x: f"{x:.3f}")
        display = display.rename(
            columns={
                "segment": "Segment",
                "txs": "Txs",
                "fraud_rate": "Fraud rate",
                "avg_amount": "Avg amount",
                "flagged_share": "Flagged share",
                "alert_share": "Volume share",
                "avg_score": "Avg score",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown('<div class="ops-section">Trends</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.caption("Daily volume · fraud rate (solid) · flagged rate (dashed)")
        chart_volume_rates(daily_series(view))
    with right:
        st.caption("Score distribution")
        chart_score_hist(view)

    st.markdown('<div class="ops-section">Queue</div>', unsafe_allow_html=True)
    st.caption("Top risky txs by rule score (max 200). Sortable columns.")
    if view.empty:
        st.info("No rows for current filters.")
    else:
        render_queue(view)


main()
