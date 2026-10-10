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
from aqs.config import CITIES, INTERIM, PROCESSED, WHO_24H_GUIDELINE

SURFACE, INK, INK2, MUTED, GRID, AXIS, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
COLOR = {"bangkok": "#2a78d6", "krakow": "#eb6834", "melbourne": "#1baf7a"}
NAME = {"bangkok": "Bangkok", "krakow": "Kraków", "melbourne": "Melbourne"}
ATTENTION = "attention index (all terms combined)"

st.set_page_config(page_title="Air quality vs search", page_icon="🌫️", layout="wide")


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
    return pd.read_csv(PROCESSED / "verdicts.csv", index_col="city")


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
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK2),
        margin=dict(l=10, r=10, t=30, b=10), hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
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
    return style(fig, 300)


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
    return style(fig, 240)


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
    st.title("Does bad air show up in what people search for?")
    st.markdown(
        "Weekly PM2.5 pollution against Google search interest in three contrasting cities, 2022–2024. "
        "Most studies stop at a correlation. This project asks **when the public notices bad air, and when it stops noticing**.")
    v, card = verdicts(), scorecard()
    for col, city in zip(st.columns(3), CITIES):
        with col.container(border=True):
            st.markdown(f"<span style='color:{COLOR[city]};font-size:1.4rem'>●</span> **{NAME[city]}**", unsafe_allow_html=True)
            st.subheader(verdict_label(v.loc[city]))
            st.metric("Correlation, season removed", f"{v.loc[city, 'deseason_r']:.2f}")
            st.metric("Forecast gain from PM2.5", f"{v.loc[city, 'r2_gain_from_pm25']:+.2f} R²")
            st.metric("Polluted weeks unnoticed", f"{card.loc[city, 'silent_share']:.0%}")
    st.subheader("Silent pollution weeks")
    st.caption("Each bar is one week. Coloured: polluted (top third for that city) and people searched. "
               "Black: polluted, but search stayed below its usual level. Pale: ordinary week. Blank: no PM2.5 data.")
    show(silent_chart())
    st.subheader("What we found")
    st.markdown(
        "- **Search responds in the same week** as pollution in Bangkok and Kraków, and the response fades within about two weeks.\n"
        "- **The relationship survives removing the shared season** in both cities, so it is more than two series peaking in the same months.\n"
        "- **Bangkok and Kraków differ over a season:** attention builds through Bangkok's smog season and wears out through Kraków's.\n"
        "- **Melbourne shows nothing**, because its air is almost always under the WHO guideline: there is little to notice.\n"
        "- **Past pollution does not forecast future attention** (Granger tests), because the response is faster than one week.")


def page_explore(city):
    st.title(f"Explore {NAME[city]}")
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
    m4.metric("Lag with the highest correlation", f"{int(profile.idxmax()):+d} weeks")
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
    st.title(f"The Awareness Gap — {NAME[city]}")
    st.markdown("Five measurements of *when* search works as a signal of public awareness, beyond whether it correlates.")
    d = get_pair(city, ATTENTION)
    card = scorecard().loc[city]
    left, right = st.columns(2)
    with left:
        st.subheader("Level or surprise?")
        st.caption("Does attention follow the pollution the calendar predicts, or the part it does not? Bars are 95% intervals.")
        show(interval_chart(A.level_vs_surprise(d), city))
        st.subheader("Is there a switch-on level?")
        th = A.attention_threshold(d)
        better = "better than" if th["aic_gain_vs_linear"] > 2 else "no better than"
        st.caption(f"Large dots are medians of eight pollution bands. The flat-then-rising fit is {better} a straight line "
                   f"(best switch-on {th['threshold']:.0f} µg/m³, 95% interval {th['low']:.0f}–{th['high']:.0f}).")
        show(dose_chart(d, th, city))
    with right:
        st.subheader("How fast, and for how long?")
        ep = A.episode_response(d)
        st.caption(f"Average attention around {ep['summary']['episodes']} pollution spikes, relative to the four weeks before. "
                   "Grey band is a 95% bootstrap interval.")
        show(episode_chart(ep, city))
        st.subheader("Over a season")
        a, b = st.columns(2)
        direction = "builds" if card["habituation_per_week"] > 0 else "wears out"
        solid = card["habituation_p"] < 0.05
        a.metric("Attention per extra polluted week", f"{card['habituation_per_week']:+.3f} SD",
                 help="Among high-pollution weeks, holding the pollution level fixed.")
        b.metric("Polluted weeks unnoticed", f"{card['silent_share']:.0%}",
                 help=f"95% interval {card['silent_low']:.0%} to {card['silent_high']:.0%}")
        st.markdown(f"Attention **{direction}** as the season goes on." if solid else
                    "No reliable change in attention through the season.")
    st.subheader("Scorecard, all cities")
    sc = scorecard()
    table = pd.DataFrame({
        "Effect of unexpected pollution (SD per 10 µg/m³)": sc["surprise_effect"].map("{:.2f}".format),
        "Effect of seasonal pollution (SD per 10 µg/m³)": sc["expected_effect"].map("{:.2f}".format),
        "Pollution episodes studied": sc["episodes"].astype(int).astype(str),
        "Attention per extra polluted week (SD)": sc["habituation_per_week"].map("{:+.3f}".format),
        "Polluted weeks unnoticed": sc["silent_share"].map("{:.0%}".format),
    }).T
    table.columns = [NAME[c] for c in table.columns]
    st.dataframe(table, use_container_width=True)


def page_data(city):
    st.title("Data and methods")
    st.subheader("PM2.5 coverage")
    st.caption("Days of valid PM2.5 behind each week. All three cities lose the same weeks in early 2023: that gap is in OpenAQ itself.")
    show(coverage_chart())
    sel = pd.read_csv(INTERIM / "sensor_selection.csv")
    pm25 = load()[1]
    q = sel.groupby("city").agg(candidates=("sensor_id", "size"), kept=("decision", lambda s: (s == "kept").sum()))
    q["weeks with data"] = pm25.groupby("city").size()
    q["median PM2.5 (µg/m³)"] = pm25.groupby("city")["pm25_mean"].median().round(1)
    q.index = [NAME[c] for c in q.index]
    st.dataframe(q.rename(columns={"candidates": "sensors found", "kept": "sensors kept"}), use_container_width=True)
    st.subheader(f"Usable search series — {NAME[city]}")
    st.caption("Grey series are zero in more than half of all weeks and are left out.")
    show(zero_chart(city))
    st.subheader("How the numbers are made")
    st.markdown(
        "- **PM2.5:** OpenAQ, every reference-grade monitor within 25 km with data on at least half the study days; "
        "daily median across monitors, averaged to Sunday–Saturday weeks.\n"
        "- **Search:** Google Trends, weekly, for each city's own region, in English and the local language.\n"
        "- **Attention index:** the first principal component of a city's usable search series.\n"
        "- **Removing the season:** a fit of trend plus annual sine waves is subtracted from each series.\n"
        "- **Uncertainty:** block bootstrap and circular-shift tests, which allow for neighbouring weeks being alike; "
        "p-values adjusted for multiple tests.")
    st.subheader("Limits")
    st.markdown(
        "- Three years of weekly data: five or six pollution episodes and three seasons per city.\n"
        "- The response is faster than a week, so weekly data cannot show which comes first.\n"
        "- Google Trends is a relative index: patterns can be compared between cities, levels cannot.\n"
        "- Melbourne is missing about a year of PM2.5; Kraków rests on two monitors.")


PAGES = {"Overview": lambda c: page_overview(), "Explore a city": page_explore,
         "Awareness Gap": page_gap, "Data and methods": page_data}

with st.sidebar:
    st.header("Air quality vs search")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    city = st.selectbox("City", CITIES, format_func=NAME.get)
    st.caption("PM2.5 from OpenAQ · search interest from Google Trends · 2022–2024")

PAGES[page](city)
