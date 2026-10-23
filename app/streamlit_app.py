import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from aqs import awareness as A, features as F, stats as S
from aqs.config import CITIES, HEADLINE_TERMS, INTERIM, PROCESSED, WHO_24H_GUIDELINE, default_geo_level

SURFACE, INK, INK2, MUTED, GRID, AXIS, EMPTY = "#ffffff", "#111111", "#4a4a47", "#8a8984", "#ececea", "#cfcec8", "#f3f3f1"
COLOR = {city: cfg["color"] for city, cfg in CITIES.items()}
NAME = {city: cfg["name"] for city, cfg in CITIES.items()}
ATTENTION = "attention index (all terms combined)"
BADGE = {"Reliable proxy": "b-good", "Real, but mostly seasonal": "b-mid", "Not informative": "b-none"}

st.set_page_config(page_title="Air quality vs search", page_icon="🌫️", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display: none !important;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
html, body, [class*="css"] {letter-spacing: -0.005em;}
h1 {font-weight: 700 !important; letter-spacing: -0.03em !important; font-size: 2.35rem !important; line-height: 1.12 !important; margin-bottom: 0.2rem !important;}
h2, h3 {font-weight: 650 !important; letter-spacing: -0.02em !important;}
section[data-testid="stSidebar"] {background: #f6f6f4; border-right: 1px solid #e6e6e2;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.6rem;}
section[data-testid="stSidebar"] [role="radiogroup"] {gap: 2px;}
section[data-testid="stSidebar"] [role="radiogroup"] label {padding: 8px 12px; border-radius: 8px; width: 100%; margin: 0;}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:not(:has([data-testid="stMarkdownContainer"])):not([data-testid="stMarkdownContainer"]) {display: none !important;}
section[data-testid="stSidebar"] [role="radiogroup"] label > div > div:not(:has([data-testid="stMarkdownContainer"])):not([data-testid="stMarkdownContainer"]) {display: none !important;}
section[data-testid="stSidebar"] [role="radiogroup"] label input {position: absolute; opacity: 0; pointer-events: none;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: #ececea;}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"], section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: linear-gradient(90deg, #2a78d6, #6c4fd8);}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p, section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color: #ffffff !important; font-weight: 600;}
.brand {font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; color: #111;}
.brand-sub {color: #8a8984; font-size: 0.82rem; margin-top: 2px; margin-bottom: 18px;}
.side-label {text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; color: #8a8984; font-weight: 600; margin: 18px 0 6px 0;}
.eyebrow {text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; color: #2a78d6; font-weight: 700; margin-bottom: 6px;}
.heroband {background: linear-gradient(120deg, #eef4fd 0%, #f4effd 45%, #fdf3ea 100%); border: 1px solid #e8e6f3; border-radius: 18px; padding: 26px 30px 6px 30px; margin-bottom: 24px;}
.heroband .lede {margin-bottom: 20px;}
.chips {display: flex; gap: 8px; flex-wrap: wrap; margin: -6px 0 20px 0;}
.chip {display: inline-flex; align-items: center; gap: 6px; background: #fff; border: 1px solid #e6e6e2; border-radius: 999px; padding: 4px 12px 4px 10px; font-size: 0.82rem; color: #333; font-weight: 600;}
.badge {display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 700; margin: 12px 0 10px 0;}
.b-good {background: #e3f5ec; color: #12784f;}
.b-mid {background: #fdf0dc; color: #a35d00;}
.b-none {background: #efefed; color: #6b6a66;}
.lede {color: #4a4a47; font-size: 1.05rem; line-height: 1.55; max-width: 820px; margin: 6px 0 26px 0;}
.section {margin: 34px 0 8px 0;}
.section .eyebrow {margin-bottom: 2px;}
.section h3 {margin: 0 !important; padding: 0 !important; font-size: 1.3rem !important;}
.section p {color: #6b6a66; font-size: 0.9rem; margin: 4px 0 0 0; max-width: 900px;}
.card {background: #fff; border: 1px solid #e6e6e2; border-radius: 14px; padding: 16px 18px 14px 18px; height: 100%; border-top-width: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);}
.card .city {display: flex; align-items: center; gap: 8px; font-weight: 600; color: #111; font-size: 0.95rem;}
.card .dot {width: 9px; height: 9px; border-radius: 50%; display: inline-block;}
.card .big {font-size: 2.3rem; font-weight: 750; letter-spacing: -0.04em; line-height: 1; margin: 2px 0 2px 0; font-variant-numeric: tabular-nums;}
.card .bigk {color: #8a8984; font-size: 0.78rem; margin-bottom: 8px;}
.card .row {display: flex; justify-content: space-between; align-items: baseline; border-top: 1px solid #f0f0ed; padding: 8px 0 2px 0;}
.card .k {color: #8a8984; font-size: 0.8rem;}
.card .v {color: #111; font-weight: 650; font-size: 1.05rem; font-variant-numeric: tabular-nums;}
.tile {background: #fff; border: 1px solid #e6e6e2; border-radius: 14px; padding: 16px 18px; border-left-width: 5px;}
.tile .k {color: #8a8984; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;}
.tile .v {color: #111; font-weight: 700; font-size: 1.9rem; letter-spacing: -0.03em; margin-top: 4px; font-variant-numeric: tabular-nums;}
.tile .s {color: #6b6a66; font-size: 0.82rem; margin-top: 2px;}
.finding {border-left: 3px solid #6c4fd8; background: #fff; border-radius: 0 10px 10px 0; padding-top: 8px !important; padding-bottom: 8px !important; padding: 2px 0 2px 14px; margin: 10px 0; color: #333; line-height: 1.5;}
div[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #ececea; border-radius: 12px; padding: 6px 4px 0 4px;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e6e2; border-radius: 12px; padding: 12px 14px;}
div[data-testid="stMetricLabel"] p {color: #8a8984 !important; font-size: 0.8rem !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow, title, lede=None):
    chips = "".join(f"<span class='chip'><span class='dot' style='width:8px;height:8px;border-radius:50%;background:{COLOR[c]}'></span>{NAME[c]}</span>" for c in CITIES)
    lede_html = f"<div class='lede'>{lede}</div>" if lede else ""
    st.markdown(f"<div class='heroband'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1>{lede_html}<div class='chips'>{chips}</div></div>",
                unsafe_allow_html=True)


def section(eyebrow, title, note=None):
    note_html = f"<p>{note}</p>" if note else ""
    st.markdown(f"<div class='section'><div class='eyebrow'>{eyebrow}</div><h3>{title}</h3>{note_html}</div>",
                unsafe_allow_html=True)


def tile(col, label, value, sub="", color="#111111"):
    col.markdown(f"<div class='tile' style='border-left-color:{color}'><div class='k'>{label}</div><div class='v' style='color:{color}'>{value}</div><div class='s'>{sub}</div></div>",
                 unsafe_allow_html=True)


@st.cache_data
def load():
    return F.load_long(), F.load_pm25()


@st.cache_data
def scorecard():
    return A.scorecard(load()[0])


@st.cache_data
def correlations():
    return S.correlation_table(load()[0])


@st.cache_data
def verdicts():
    v = pd.read_csv(PROCESSED / "verdicts.csv")
    return v[v["role"] == "attention"].set_index("city")


def health_summary():
    t = correlations()
    health = t[t["role"] == "health"]
    worst = health.assign(drop=health["raw_r"] - health["deseason_r"]).sort_values("drop").iloc[-1]
    return {"series": len(health), "significant": int(health["significant"].sum()), "city": NAME[worst["city"]],
            "label": worst["series"].split(" (")[0], "raw": worst["raw_r"], "deseason": worst["deseason_r"]}


@st.cache_data
def series_options(city):
    summ = F.series_summary(load()[0])
    keep = summ[(summ["city"] == city) & summ["dense"]]
    return {ATTENTION: None, **{r["label"]: (r["term"], r["geo_level"]) for _, r in keep.iterrows()}}


def get_pair(city, choice):
    long = load()[0]
    key = series_options(city)[choice]
    return F.attention_pair(long, city) if key is None else F.pair(long, city, *key)


def verdict_label(row):
    if not row["correlation_survives"]:
        return "Not informative"
    return "Reliable proxy" if row["r2_gain_from_pm25"] > 0.05 else "Real, but mostly seasonal"


def style(fig, height=340, legend=False):
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, showlegend=legend,
        font=dict(family="Source Sans Pro, system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK2),
        margin=dict(l=12, r=12, t=24, b=12), hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def timeline_chart(d, city, search_label, pm_label):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=d.index, y=d["pm25_mean"], mode="lines", line=dict(color=INK2, width=2),
                             name=pm_label, hovertemplate="%{x|%d %b %Y}<br>" + pm_label + ": %{y:.1f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["search"], mode="lines", line=dict(color=COLOR[city], width=2),
                             name=search_label, hovertemplate="%{x|%d %b %Y}<br>search: %{y:.1f}<extra></extra>"), 2, 1)
    fig.update_yaxes(title_text=pm_label, row=1, col=1)
    fig.update_yaxes(title_text="search", row=2, col=1)
    return style(fig, 420)


def scatter_chart(d, city, pm_label):
    d = d.dropna()
    fig = go.Figure(go.Scatter(
        x=d["pm25_mean"], y=d["search"], mode="markers", text=d.index.strftime("%d %b %Y"),
        marker=dict(color=COLOR[city], size=8, opacity=0.6, line=dict(color=SURFACE, width=1)),
        hovertemplate="%{text}<br>" + pm_label + ": %{x:.1f}<br>search: %{y:.1f}<extra></extra>"))
    if len(d) > 2:
        slope, intercept = np.polyfit(d["pm25_mean"], d["search"], 1)
        xs = np.array([d["pm25_mean"].min(), d["pm25_mean"].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", line=dict(color=INK, width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text=pm_label)
    fig.update_yaxes(title_text="search")
    return style(fig, 420)


def lag_chart(profile, city, chosen):
    fig = go.Figure(go.Bar(
        x=profile.index, y=profile.values, width=0.6,
        marker_color=[COLOR[city] if k == chosen else AXIS for k in profile.index],
        hovertemplate="lag %{x} weeks<br>correlation %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=INK2, line_width=1)
    fig.update_xaxes(title_text="← search first · lag in weeks · search after →", dtick=2)
    fig.update_yaxes(title_text="correlation")
    return style(fig, 300)


def silent_chart():
    long = load()[0]
    fig = make_subplots(rows=len(CITIES), cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for i, city in enumerate(CITIES, start=1):
        df, _ = A.silent_weeks(F.attention_pair(long, city), city)
        weeks = pd.date_range(df.index.min(), df.index.max(), freq=F.WEEK)
        df = df.reindex(weeks)
        state = np.where(df["search"].isna(), "no data",
                         np.where(df["silent"].fillna(False), "polluted and silent",
                                  np.where(df["high"].fillna(False), "polluted and noticed", "ordinary week")))
        colors = pd.Series(state).map({"no data": SURFACE, "ordinary week": EMPTY,
                                       "polluted and noticed": COLOR[city], "polluted and silent": INK})
        fig.add_trace(go.Bar(
            x=weeks, y=np.ones(len(weeks)), marker_color=colors, marker_line_width=0, width=6.2 * 86400000,
            customdata=np.column_stack([state, df["pm25_mean"].round(1).astype(str)]),
            hovertemplate="%{x|%d %b %Y}<br>%{customdata[0]}<br>PM2.5 %{customdata[1]} µg/m³<extra>" + NAME[city] + "</extra>"), i, 1)
        fig.update_yaxes(showticklabels=False, showgrid=False, title_text=NAME[city], row=i, col=1, range=[0, 1])
    fig.update_layout(bargap=0)
    return style(fig, 100 * len(CITIES))


def interval_chart(res, city):
    labels = {"expected": "seasonal pollution", "surprise": "unexpected pollution"}
    fig = go.Figure()
    for name in ["expected", "surprise"]:
        r = res.loc[name]
        fig.add_trace(go.Scatter(
            x=[r["effect"]], y=[labels[name]], mode="markers",
            marker=dict(size=14, color=COLOR[city] if name == "surprise" else SURFACE, line=dict(color=COLOR[city], width=2.5)),
            error_x=dict(type="data", symmetric=False, array=[r["high"] - r["effect"]], arrayminus=[r["effect"] - r["low"]],
                         color=COLOR[city], thickness=2, width=0),
            hovertemplate=f"{labels[name]}<br>effect %{{x:.2f}} SD per 10 µg/m³<br>95% interval {r['low']:.2f} to {r['high']:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title_text="change in attention (SD) per 10 µg/m³")
    return style(fig, 230)


def episode_chart(res, city):
    c = res["search"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.index, y=c["high"], mode="lines", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c.index, y=c["low"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(137,135,129,0.18)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c.index, y=c["mean"], mode="lines+markers", line=dict(color=COLOR[city], width=2.5),
                             marker=dict(size=7), hovertemplate="week %{x}<br>attention change %{y:.2f} SD<extra></extra>"))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title_text="weeks from the start of a pollution spike", dtick=2)
    fig.update_yaxes(title_text="attention change (SD)")
    return style(fig, 320)


def dose_chart(d, th, city):
    d = d.dropna()
    med = d.groupby(pd.qcut(d["pm25_mean"], 8), observed=True).median()
    xs = np.linspace(d["pm25_mean"].min(), d["pm25_mean"].max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["pm25_mean"], y=d["search"], mode="markers", text=d.index.strftime("%d %b %Y"),
                             marker=dict(color=COLOR[city], size=6, opacity=0.3),
                             hovertemplate="%{text}<br>PM2.5 %{x:.1f}<br>attention %{y:.2f} SD<extra></extra>"))
    fig.add_trace(go.Scatter(x=med["pm25_mean"], y=med["search"], mode="markers",
                             marker=dict(color=COLOR[city], size=12, line=dict(color=SURFACE, width=2)),
                             hovertemplate="median of band<br>PM2.5 %{x:.1f}<br>attention %{y:.2f} SD<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=th["base"] + th["slope_per_10"] / 10 * np.clip(xs - th["threshold"], 0, None),
                             mode="lines", line=dict(color=INK, width=2), hoverinfo="skip"))
    fig.add_vline(x=WHO_24H_GUIDELINE, line_color=INK2, line_width=1, annotation_text="WHO 15", annotation_position="top")
    fig.update_xaxes(title_text="weekly PM2.5 (µg/m³)")
    fig.update_yaxes(title_text="attention (SD)", range=[d["search"].min() - 0.3, d["search"].quantile(0.97) + 0.6])
    return style(fig, 320)


def coverage_chart():
    pm25 = load()[1]
    weeks = pd.date_range(pm25["week"].min(), pm25["week"].max(), freq=F.WEEK)
    z = [pm25[pm25["city"] == c].set_index("week")["n_days"].reindex(weeks).fillna(0).to_numpy() for c in CITIES]
    fig = go.Figure(go.Heatmap(
        z=z, x=weeks, y=[NAME[c] for c in CITIES], zmin=0, zmax=7, xgap=1, ygap=6,
        colorscale=[[0, "#eef4fd"], [0.4, "#6da7ec"], [0.75, "#256abf"], [1, "#0d366b"]],
        colorbar=dict(title="days", thickness=10, len=0.8),
        hovertemplate="%{y}, week of %{x|%d %b %Y}<br>%{z} days of PM2.5 data<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style(fig, 60 * (len(CITIES) + 1))


def zero_chart(city):
    summ = F.series_summary(load()[0])
    d = summ[summ["city"] == city].sort_values("zero_share", ascending=False)
    fig = go.Figure(go.Bar(
        x=d["zero_share"] * 100, y=d["label"], orientation="h", width=0.6,
        marker_color=[COLOR[city] if ok else AXIS for ok in d["dense"]],
        hovertemplate="%{y}<br>%{x:.0f}% of weeks are zero<extra></extra>"))
    fig.add_vline(x=50, line_color=INK2, line_width=1)
    fig.update_xaxes(title_text="% of weeks with zero search interest", range=[0, 100])
    return style(fig, 60 + 30 * len(d))


def page_overview():
    hero("PM2.5 × Google Trends · 4 cities · 2022–2024", "Does bad air show up in what people search for?",
         "Weekly PM2.5 from reference monitors against Google search interest in Bangkok, Kraków, Melbourne and Lahore. "
         "The question is not only whether the two move together, but <b>when the public notices bad air, and when it stops noticing</b>.")
    v, card = verdicts(), scorecard()
    for col, city in zip(st.columns(len(CITIES), gap="small"), CITIES):
        col.markdown(
            f"<div class='card' style='border-top-color:{COLOR[city]}'><div class='city'><span class='dot' style='background:{COLOR[city]}'></span>{NAME[city]}</div>"
            f"<span class='badge {BADGE[verdict_label(v.loc[city])]}'>{verdict_label(v.loc[city])}</span>"
            f"<div class='big' style='color:{COLOR[city]}'>{v.loc[city, 'deseason_r']:.2f}</div><div class='bigk'>correlation with PM2.5, season removed</div>"
            f"<div class='row'><span class='k'>Forecast gain from PM2.5</span><span class='v'>{v.loc[city, 'r2_gain_from_pm25']:+.2f} R²</span></div>"
            f"<div class='row'><span class='k'>Polluted weeks unnoticed</span><span class='v'>{card.loc[city, 'silent_share']:.0%}</span></div></div>",
            unsafe_allow_html=True)
    section("Awareness gap", "Silent pollution weeks",
            "Each bar is one week. Coloured: polluted (top third for that city) and people searched. "
            "Black: polluted, but search stayed below its usual level. Pale: ordinary week. Blank: no PM2.5 data.")
    show(silent_chart())
    section("Summary", "What we found")
    h = health_summary()
    findings = [
        "<b>Attention survives removing the shared season</b> in Bangkok, Kraków and Lahore. Lahore's estimate rests on one monitor and has a very wide interval.",
        f"<b>Health searches follow the season, not the air.</b> {h['city']} \"{h['label']}\" correlates with PM2.5 at {h['raw']:+.2f} raw and "
        f"{h['deseason']:+.2f} with the season removed; {h['significant']} of {h['series']} health series stay significant.",
        "<b>Search responds in the same week</b> as pollution in Bangkok and Kraków and within a week in Lahore.",
        "<b>Attention builds through Bangkok's season and wears out through Kraków's</b>; Lahore leans towards wearing out.",
        "<b>Melbourne shows nothing</b>: its air is almost always under the WHO guideline, so there is little to notice.",
        "<b>Past pollution does not forecast future attention</b> (Granger), because the response is faster than a week.",
    ]
    st.markdown("".join(f"<div class='finding'>{f}</div>" for f in findings), unsafe_allow_html=True)
    st.caption("Card label: not informative if the correlation does not survive the season, a reliable proxy if PM2.5 adds more "
               "than 0.05 R² to a calendar forecast, otherwise real but mostly seasonal.")


def page_explore(city):
    hero("City explorer", f"Explore {NAME[city]}",
         "Pick a search series, remove the season and shift the lag to see how the relationship holds up.")
    options = list(series_options(city))
    c1, c2, c3 = st.columns([2, 1, 2])
    choice = c1.selectbox("Search series", options)
    remove = c2.toggle("Remove the season", help="Subtract a seasonal fit from both series, leaving what the calendar does not explain.")
    lag = c3.slider("Shift search by (weeks)", -8, 8, 0, help="Positive: compare pollution with search that many weeks later.")
    d = get_pair(city, choice)
    if remove:
        d = S.deseasonalize(d)
    profile = F.lag_correlation(d)
    shifted = d.assign(search=d["search"].shift(-lag))
    both = shifted.dropna()
    pm_label = "PM2.5, season removed" if remove else "PM2.5 (µg/m³)"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pearson correlation", f"{stats.pearsonr(both['pm25_mean'], both['search'])[0]:.2f}")
    m2.metric("Spearman (rank) correlation", f"{stats.spearmanr(both['pm25_mean'], both['search'])[0]:.2f}")
    m3.metric("Weeks compared", len(both))
    peak = F.peak_lag(profile)
    m4.metric("Lag with the highest correlation", "no peak" if peak is None else f"{peak:+d} weeks",
              help="No peak: the highest correlation sits at the edge of the ±8 week window.")
    left, right = st.columns([3, 2])
    with left:
        st.caption("Pollution (top) and search (bottom) on a shared time axis")
        show(timeline_chart(shifted, city, choice, pm_label))
    with right:
        st.caption("Each dot is one week")
        show(scatter_chart(shifted, city, pm_label))
    st.caption("Correlation at every lag; the selected lag is highlighted")
    show(lag_chart(profile, city, lag))


def page_gap(city):
    hero("Awareness gap", f"When does {NAME[city]} notice?",
         "Five measurements of <i>when</i> search works as a signal of public awareness, beyond whether it correlates.")
    d = get_pair(city, ATTENTION)
    card = scorecard().loc[city]
    left, right = st.columns(2)
    with left:
        section("Response", "Level or surprise?")
        st.caption("Does attention follow the pollution the calendar predicts, or the part it does not? Bars are 95% intervals.")
        show(interval_chart(A.level_vs_surprise(d), city))
        section("Threshold", "Is there a switch-on level?")
        th = A.attention_threshold(d)
        better = "better than" if th["aic_gain_vs_linear"] > 2 else "no better than"
        span = d["pm25_mean"].max() - d["pm25_mean"].min()
        weak = th["aic_gain_vs_linear"] > 2 and th["high"] - th["low"] > span / 2
        st.caption(f"Large dots are medians of eight pollution bands. The flat-then-rising fit is {better} a straight line "
                   f"(best switch-on {th['threshold']:.0f} µg/m³, 95% interval {th['low']:.0f}–{th['high']:.0f})."
                   + (" The interval covers most of the observed range, so the switch-on point is weakly identified." if weak else ""))
        show(dose_chart(d, th, city))
    with right:
        section("Episodes", "How fast, and for how long?")
        ep = A.episode_response(d)
        st.caption(f"Average attention around {ep['summary']['episodes']} pollution spikes, relative to the four weeks before. "
                   "Grey band is a 95% bootstrap interval.")
        show(episode_chart(ep, city))
        section("Habituation", "Over a season")
        a, b = st.columns(2)
        direction = "builds" if card["habituation_per_week"] > 0 else "wears out"
        solid = card["habituation_p"] < 0.05
        a.metric("Attention per extra polluted week", f"{card['habituation_per_week']:+.3f} SD",
                 help="Among high-pollution weeks, holding the pollution level fixed.")
        b.metric("Polluted weeks unnoticed", f"{card['silent_share']:.0%}",
                 help=f"95% interval {card['silent_low']:.0%} to {card['silent_high']:.0%}")
        st.markdown(f"Attention **{direction}** as the season goes on." if solid else
                    "No reliable change in attention through the season.")
    section("Comparison", "Scorecard, all cities")
    sc = scorecard()
    table = pd.DataFrame({
        "Effect of unexpected pollution (SD per 10 µg/m³)": sc["surprise_effect"].map("{:.2f}".format),
        "Effect of seasonal pollution (SD per 10 µg/m³)": sc["expected_effect"].map("{:.2f}".format),
        "Pollution episodes studied": sc["episodes"].astype(int).astype(str),
        "Attention per extra polluted week (SD)": sc["habituation_per_week"].map("{:+.3f}".format),
        "Polluted weeks unnoticed": sc["silent_share"].map("{:.0%}".format),
    }).T
    table.columns = [NAME[c] for c in table.columns]
    st.dataframe(table, width="stretch")


def page_data(city):
    hero("Data and methods", "Where the numbers come from",
         "Coverage of the PM2.5 record, which sensors and search series were kept, and how every statistic is computed.")
    section("Coverage", "PM2.5 record by week")
    st.caption("Days of valid PM2.5 behind each week. Every city loses the same weeks in early 2023: that gap is in OpenAQ itself.")
    show(coverage_chart())
    sel = pd.read_csv(INTERIM / "sensor_selection.csv")
    pm25 = load()[1]
    q = sel.groupby("city").agg(candidates=("sensor_id", "size"), kept=("decision", lambda s: (s == "kept").sum()))
    q["weeks with data"] = pm25.groupby("city").size()
    q["median PM2.5 (µg/m³)"] = pm25.groupby("city")["pm25_mean"].median().round(1)
    q.index = [NAME[c] for c in q.index]
    st.dataframe(q.rename(columns={"candidates": "sensors found", "kept": "sensors kept"}), width="stretch")
    section("Search data", f"Usable search series in {NAME[city]}")
    st.caption("Grey series are zero in more than half of all weeks and are left out.")
    show(zero_chart(city))
    section("Method", "How the numbers are made")
    st.markdown(
        "- **PM2.5:** OpenAQ, every reference-grade monitor within 25 km with data on at least half the study days; "
        "daily median across monitors, averaged to Sunday–Saturday weeks.\n"
        "- **Search:** Google Trends, weekly, for each city's own region, in English and the local language. "
        "Health terms are country level, as are Lahore's English baseline terms.\n"
        "- **Attention index:** the first principal component of a city's usable awareness and protection series. "
        "Health terms are kept out of it.\n"
        "- **Removing the season:** a fit of trend plus annual sine waves is subtracted from each series.\n"
        "- **Uncertainty:** block bootstrap and circular-shift tests, which allow for neighbouring weeks being alike; "
        "p-values adjusted for multiple tests.")
    section("Caveats", "Limits")
    st.markdown(
        "- Three years of weekly data: five or six pollution episodes and three seasons per city.\n"
        "- The response is faster than a week, so weekly data cannot show which comes first.\n"
        "- Google Trends is a relative index: patterns can be compared between cities, levels cannot.\n"
        "- Melbourne is missing about a year of PM2.5; Kraków rests on two monitors and Lahore on one.\n"
        "- Lahore's missing weeks cluster in 2022 Q3 to 2023 Q1 and remove 10 of the 22 weeks of the 2022–23 smog season.\n"
        "- Health terms are measured for the whole country against one city's PM2.5, so a local health response would be diluted.\n"
        "- Google Trends values differ slightly between pulls.")


@st.cache_data
def results_table():
    return pd.read_csv(PROCESSED / "results_summary.csv")


def headline_row(t, city, role):
    term = HEADLINE_TERMS[city][role]
    return t[(t["city"] == city) & (t["term"] == term) & (t["geo_level"] == default_geo_level(city, term))].iloc[0]


def dumbbell_chart(rows, color, height=None):
    rows = rows.sort_values("raw_r")
    labels = rows["series"].str.replace(", country", "", regex=False).tolist()
    fig = go.Figure()
    for (_, r), lab in zip(rows.iterrows(), labels):
        fig.add_trace(go.Scatter(x=[r["raw_r"], r["deseason_r"]], y=[lab, lab], mode="lines",
                                 line=dict(color=AXIS, width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rows["raw_r"], y=labels, mode="markers", name="raw",
                             marker=dict(size=12, color=SURFACE, line=dict(color=INK2, width=2)),
                             hovertemplate="%{y}<br>raw r %{x:+.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=rows["deseason_r"], y=labels, mode="markers", name="season removed (95% interval)",
                             marker=dict(size=13, color=color, line=dict(color="white", width=1.5)),
                             error_x=dict(type="data", symmetric=False, array=rows["ci_high"] - rows["deseason_r"],
                                          arrayminus=rows["deseason_r"] - rows["ci_low"], color=color, thickness=2.5, width=0),
                             hovertemplate="%{y}<br>season removed r %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title_text="correlation with PM2.5", range=[-0.5, 0.9])
    fig.update_layout(showlegend=True)
    for tr in fig.data[:-2]:
        tr.showlegend = False
    return style(fig, height or 90 + 46 * len(rows), legend=True)


def contrast_chart(t):
    fig = make_subplots(rows=1, cols=len(CITIES), shared_yaxes=True, subplot_titles=[NAME[c] for c in CITIES], horizontal_spacing=0.04)
    for i, city in enumerate(CITIES, start=1):
        for role in ("health", "protection", "awareness"):
            r = headline_row(t, city, role)
            fig.add_trace(go.Scatter(x=[r["raw_r"], r["deseason_r"]], y=[role, role], mode="lines",
                                     line=dict(color=AXIS, width=2), hoverinfo="skip", showlegend=False), 1, i)
            fig.add_trace(go.Scatter(x=[r["raw_r"]], y=[role], mode="markers", showlegend=False,
                                     marker=dict(size=11, color=SURFACE, line=dict(color=INK2, width=2)),
                                     hovertemplate=f"{NAME[city]} {role}<br>raw %{{x:+.2f}}<extra></extra>"), 1, i)
            fig.add_trace(go.Scatter(x=[r["deseason_r"]], y=[role], mode="markers", showlegend=False,
                                     marker=dict(size=11, color=COLOR[city] if r["significant"] else "#ffffff",
                                                 line=dict(color=COLOR[city], width=2)),
                                     hovertemplate=f"{NAME[city]} {role}<br>season removed %{{x:+.2f}}<extra></extra>"), 1, i)
        fig.add_vline(x=0, line_color=INK, line_width=1, row=1, col=i)
        fig.update_xaxes(range=[-0.3, 0.9], dtick=0.3, row=1, col=i)
    fig.update_annotations(font=dict(size=14, color=INK))
    return style(fig, 260)


def page_health(city):
    t = results_table()
    health = t[t["role"] == "health"]
    ap = t[t["role"].isin(["awareness", "protection"])]
    drop = health.assign(drop=health["raw_r"] - health["deseason_r"]).sort_values("drop").iloc[-1]
    hero("Health signal", "Do health searches respond to the air, or to winter?",
         "Searches for symptoms and remedies rise in the smog season. Removing the shared seasonal cycle from both series "
         "shows whether anything is left once winter is taken out.")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    tile(c1, "Health series tested", f"{len(health)}", "dense series across 4 cities", "#6c4fd8")
    tile(c2, "Health significant, season removed", f"{int(health['significant'].sum())}", "after FDR correction", "#d6453d")
    tile(c3, "Awareness + protection significant", f"{int(ap['significant'].sum())} of {len(ap)}", "on the same test", "#12a06a")
    tile(c4, "Largest collapse", f"{drop['raw_r']:+.2f} → {drop['deseason_r']:+.2f}",
         f"{NAME[drop['city']]}, {drop['series'].split(' (')[0]}", COLOR[drop['city']])
    section("Selected city", f"Every health series in {NAME[city]}",
            "Open circle: raw correlation. Filled dot: correlation once the season is removed, with its 95% block-bootstrap interval.")
    show(dumbbell_chart(health[health["city"] == city], COLOR[city]))
    section("All cities", "Awareness and protection keep their link; health does not",
            "Headline series per role, raw (open) and season removed (filled when significant).")
    show(contrast_chart(t))
    section("Over time", f"PM2.5 and searches for breathing difficulty in {NAME[city]}")
    remove = st.toggle("Remove the season from both series", value=False)
    long = load()[0]
    term = HEADLINE_TERMS[city]["health"]
    d = F.pair(long, city, term, default_geo_level(city, term))
    if remove:
        d = S.deseasonalize(d)
    show(timeline_chart(d, city, "breathing difficulty", "PM2.5, season removed" if remove else "PM2.5 (µg/m³)"))
    st.markdown("<div class='finding'>A raw correlation between pollution and health searches is not evidence of a health response: "
                "cold-season illness and cold-season pollution arrive together.</div>", unsafe_allow_html=True)


PAGES = {"Overview": lambda c: page_overview(), "Explore a city": page_explore,
         "Awareness Gap": page_gap, "Health signal": page_health, "Data and methods": page_data}

with st.sidebar:
    st.markdown("<div class='brand'>Air quality vs search</div><div class='brand-sub'>PM2.5 and public attention</div>", unsafe_allow_html=True)
    st.markdown("<div class='side-label'>Pages</div>", unsafe_allow_html=True)
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.markdown("<div class='side-label'>City</div>", unsafe_allow_html=True)
    city = st.selectbox("City", CITIES, format_func=NAME.get, label_visibility="collapsed")
    st.markdown("<div class='side-label'>Sources</div>", unsafe_allow_html=True)
    st.caption("PM2.5 from OpenAQ reference monitors · search interest from Google Trends · weekly, 2022–2024")

PAGES[page](city)
