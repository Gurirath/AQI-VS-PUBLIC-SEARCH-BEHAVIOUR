"""Project plot theme and the EDA figures.

Conventions used in every figure:
  - colour identifies the city (Bangkok blue, Kraków orange, Melbourne green)
  - PM2.5 is always drawn in grey ink, search in the city colour
  - one y-axis per panel; two measures are compared as z-scores or in stacked panels
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf

from . import features as F
from .config import CITIES, FIGURES, WHO_24H_GUIDELINE

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CITY_COLOR = {"bangkok": "#2a78d6", "krakow": "#eb6834", "melbourne": "#1baf7a"}
CITY_NAME = {"bangkok": "Bangkok", "krakow": "Kraków", "melbourne": "Melbourne"}
PM_COLOR = INK2
EMPTY = "#f0efec"

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fbe0d3", "#f1946c", "#c9501f", "#7a2c0c"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", EMPTY, "#f09a99", "#b52f2f"])
for _cm in (SEQ_BLUE, SEQ_ORANGE, DIVERGING):
    _cm.set_bad(EMPTY)


def use_theme():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def finish(fig, title, subtitle=None, name=None):
    """Figure-level title and subtitle, then save under outputs/figures."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha="left", va="top", fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.01, 1 - 0.38 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.46 if subtitle else 0.24) / h))
    if name:
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / f"{name}.png")
    return fig


def _city_axes(ncols=3, nrows=1, height=3.2, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, height * nrows), squeeze=False, **kw)
    return fig, axes


def _note(ax, text, loc="upper right"):
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2))


# ---------------------------------------------------------------- data quality

def coverage_strips(pm25):
    """Days of PM2.5 data behind each week, one strip per city."""
    weeks = pd.date_range(pm25["week"].min(), pm25["week"].max(), freq=F.WEEK)
    fig, axes = plt.subplots(3, 1, figsize=(12, 3.4), sharex=True)
    for ax, city in zip(axes, CITIES):
        days = pm25[pm25["city"] == city].set_index("week")["n_days"].reindex(weeks).fillna(0)
        mesh = ax.pcolormesh(weeks, [0, 1], days.to_numpy()[None, :-1], cmap=SEQ_BLUE, vmin=0, vmax=7)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_ylabel(CITY_NAME[city], rotation=0, ha="right", va="center", color=INK)
        ax.text(1.01, 0.5, f"{(days >= 4).sum()} of {len(weeks)} weeks", transform=ax.transAxes,
                va="center", fontsize=9, color=INK2)
    cb = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.22, aspect=70)
    cb.set_label("days with valid PM2.5 in the week (palest = none)")
    cb.outline.set_visible(False)
    fig.suptitle("Where the PM2.5 record has holes", x=0.01, y=0.99, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.savefig(FIGURES / "eda_coverage_strips.png")
    return fig


def zero_inflation(long):
    """Share of zero weeks in every search series."""
    summ = F.series_summary(long)
    fig, axes = _city_axes(height=3.6, sharex=True)
    for ax, city in zip(axes[0], CITIES):
        d = summ[summ["city"] == city].sort_values("zero_share")
        ax.barh(d["label"], d["zero_share"] * 100, height=0.62,
                color=[CITY_COLOR[city] if ok else AXIS for ok in d["dense"]])
        ax.axvline(50, color=INK2, linewidth=1)
        ax.set_title(CITY_NAME[city])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of weeks with zero search interest")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0][0].text(52, -0.4, "too sparse past here", fontsize=8.5, color=INK2, va="center")
    return finish(fig, "Which search series are usable",
                  "Grey bars are zero in more than half of all weeks and are dropped from continuous analysis.",
                  "eda_zero_inflation")


# ------------------------------------------------------------------ univariate

def distributions(pm25):
    """PM2.5 histograms, raw and log-scaled, with shape statistics."""
    fig, axes = _city_axes(nrows=2, height=2.8)
    for j, city in enumerate(CITIES):
        x = pm25.loc[pm25["city"] == city, "pm25_mean"]
        for i, (vals, label) in enumerate([(x, "weekly PM2.5 (µg/m³)"), (np.log10(x), "log₁₀ weekly PM2.5")]):
            ax = axes[i][j]
            ax.hist(vals, bins=24, color=CITY_COLOR[city], edgecolor=SURFACE, linewidth=1)
            guide = WHO_24H_GUIDELINE if i == 0 else np.log10(WHO_24H_GUIDELINE)
            if vals.max() > guide:
                ax.axvline(guide, color=INK, linewidth=1)
                ax.text(guide, ax.get_ylim()[1], " WHO 15 ", fontsize=8.5, color=INK, va="top",
                        ha="right" if city == "melbourne" else "left")
            _note(ax, f"skew {stats.skew(vals):+.2f}\nkurtosis {stats.kurtosis(vals):+.2f}",
                  "upper left" if city == "melbourne" else "upper right")
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
            ax.set_xlabel(label)
            ax.set_ylabel("weeks")
            if i == 0:
                ax.set_title(CITY_NAME[city])
    return finish(fig, "PM2.5 is right-skewed; a log scale removes most of the skew",
                  "Top row raw, bottom row log-transformed. Kurtosis is excess kurtosis (0 = normal).",
                  "eda_distributions")


def ecdf(pm25):
    """Cumulative distribution of weekly PM2.5, all cities on one log axis."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    label_at = {"bangkok": 60, "krakow": 25, "melbourne": 60}  # % level where each curve has clear space to its left
    for city in CITIES:
        x = np.sort(pm25.loc[pm25["city"] == city, "pm25_mean"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x) * 100
        above = (x > WHO_24H_GUIDELINE).mean() * 100
        ax.step(x, y, where="post", color=CITY_COLOR[city],
                label=f"{CITY_NAME[city]}: {above:.0f}% of weeks above WHO 15")
        ax.text(np.percentile(x, label_at[city]) * 0.95, label_at[city], CITY_NAME[city],
                color=INK, fontsize=9, ha="right", va="center")
    ax.axvline(WHO_24H_GUIDELINE, color=INK, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 15, 25, 50])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(3, 70)
    ax.set_xlabel("weekly PM2.5 (µg/m³, log scale)")
    ax.set_ylabel("% of weeks at or below")
    ax.legend(loc="lower right")
    return finish(fig, "Three different pollution regimes", name="eda_ecdf")


def qq_plots(pm25):
    """Normal QQ plots for raw and log PM2.5."""
    fig, axes = _city_axes(nrows=2, height=2.8)
    for j, city in enumerate(CITIES):
        x = pm25.loc[pm25["city"] == city, "pm25_mean"]
        for i, (vals, label) in enumerate([(x, "raw"), (np.log10(x), "log₁₀")]):
            ax = axes[i][j]
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            ax.plot(osm, slope * osm + intercept, color=INK2, linewidth=1)
            ax.scatter(osm, osr, s=14, color=CITY_COLOR[city], edgecolor=SURFACE, linewidth=0.5)
            _note(ax, f"{label}: fit r = {r:.3f}", "lower right")
            ax.set_xlabel("normal quantile")
            ax.set_ylabel("sample quantile")
            if i == 0:
                ax.set_title(CITY_NAME[city])
    return finish(fig, "How far PM2.5 is from normal, before and after a log transform",
                  "Points on the line follow a normal distribution; the upper tail peels away in the raw data.",
                  "eda_qq")


def outlier_timeline(pm25):
    """Weeks that are unusual for their time of year (robust z of the seasonal anomaly)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.6), sharex=True)
    for ax, city in zip(axes, CITIES):
        s = pm25[pm25["city"] == city].set_index("week")["pm25_mean"].asfreq(F.WEEK)
        z = F.robust_z(F.seasonal_anomaly(s))
        hot = z > 3
        ax.plot(s.index, s, color=PM_COLOR, linewidth=1.5)
        ax.scatter(s.index[hot], s[hot], s=46, color=CITY_COLOR[city], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"{CITY_NAME[city]} — {int(hot.sum())} unusual weeks")
    return finish(fig, "Pollution spikes that the season does not explain",
                  "Marked weeks sit more than 3 robust standard deviations above the usual level for that month.",
                  "eda_outliers")


# -------------------------------------------------------------- time structure

def seasonal_profile(long):
    """Month-of-year distributions of PM2.5 and the headline awareness search."""
    fig, axes = _city_axes(nrows=2, height=2.8, sharex=True)
    months = list("JFMAMJJASOND")
    for j, city in enumerate(CITIES):
        d = F.headline(long, city).dropna()
        for i, (col, color, label) in enumerate([("pm25_mean", PM_COLOR, "PM2.5 (µg/m³)"),
                                                 ("search", CITY_COLOR[city], "search interest (0–100)")]):
            ax = axes[i][j]
            groups = [d.loc[d.index.month == m, col] for m in range(1, 13)]
            ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, edgecolor=color, alpha=0.35),
                       medianprops=dict(color=color, linewidth=2),
                       whiskerprops=dict(color=color, linewidth=1), capprops=dict(visible=False))
            ax.set_xticks(range(1, 13), months)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel(label)
            if i == 0:
                ax.set_title(CITY_NAME[city])
            else:
                _note(ax, F.term_label(city, F.HEADLINE_TERMS[city]["awareness"]), "upper left")
    return finish(fig, "The seasonal calendar of pollution and of attention",
                  "Box = middle half of weeks in that month, line = median. Top row PM2.5, bottom row search.",
                  "eda_seasonal_profile")


def _year_week_grid(s):
    """Series -> year x week-of-year matrix."""
    df = pd.DataFrame({"v": s, "year": s.index.year, "woy": s.index.dayofyear // 7})
    return df.pivot_table(index="year", columns="woy", values="v").reindex(columns=range(53))


def calendar_heatmaps(long):
    """Twin calendars: PM2.5 above, search below, one column per city."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 3.6))
    for j, city in enumerate(CITIES):
        d = F.headline(long, city)
        for i, (col, cmap, label) in enumerate([("pm25_mean", SEQ_BLUE, "PM2.5"), ("search", SEQ_ORANGE, "search")]):
            ax = axes[i][j]
            grid = _year_week_grid(d[col])
            mesh = ax.pcolormesh(np.arange(54), np.arange(len(grid) + 1), np.ma.masked_invalid(grid.to_numpy()),
                                 cmap=cmap, edgecolors=SURFACE, linewidth=0.6)
            ax.set_yticks(np.arange(len(grid)) + 0.5, grid.index)
            ax.invert_yaxis()
            ax.set_xticks(np.array([0, 13, 26, 39]) + 0.5, ["Jan", "Apr", "Jul", "Oct"])
            ax.grid(False)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0, labelsize=8)
            ax.set_title(f"{CITY_NAME[city]} — {label}" if i == 0 else
                         f"{label}: {F.term_label(city, F.HEADLINE_TERMS[city]['awareness'])}", fontsize=9.5)
    return finish(fig, "Pollution and attention, week by week",
                  "Each cell is one week; darker is higher. Pale grey cells have no PM2.5 data.",
                  "eda_calendar_heatmaps")


def autocorrelation(long, nlags=60):
    """How long each series remembers itself."""
    fig, axes = _city_axes(nrows=2, height=2.6, sharex=True, sharey=True)
    for j, city in enumerate(CITIES):
        d = F.headline(long, city)
        for i, (col, color, label) in enumerate([("pm25_mean", PM_COLOR, "PM2.5"), ("search", CITY_COLOR[city], "search")]):
            ax = axes[i][j]
            r = acf(d[col], nlags=nlags, missing="conservative")
            band = 1.96 / np.sqrt(d[col].notna().sum())
            ax.axhspan(-band, band, color=EMPTY, zorder=0)
            ax.vlines(range(1, nlags + 1), 0, r[1:], color=color, linewidth=1.6)
            ax.axhline(0, color=AXIS, linewidth=1)
            ax.axvline(52, color=INK, linewidth=1)
            ax.set_ylabel(f"{label} autocorrelation")
            if i == 0:
                ax.set_title(CITY_NAME[city])
            else:
                ax.set_xlabel("lag (weeks)")
    axes[0][0].text(52, axes[0][0].get_ylim()[1], "one year ", ha="right", va="top", fontsize=8.5, color=INK)
    return finish(fig, "Bangkok and Kraków carry a strong annual cycle; Melbourne does not",
                  "Bars outside the grey band are distinguishable from zero. Neighbouring weeks are not independent observations.",
                  "eda_autocorrelation")


def stl_decomposition(long, cities=("bangkok", "krakow")):
    """Trend, seasonal and residual parts of PM2.5 and search."""
    fig, axes = plt.subplots(4, 2 * len(cities), figsize=(6.2 * len(cities), 7.4), sharex=True)
    for c, city in enumerate(cities):
        d = F.headline(long, city)
        for m, (col, color, label) in enumerate([("pm25_mean", PM_COLOR, "PM2.5"), ("search", CITY_COLOR[city], "search")]):
            parts = F.stl_parts(d[col], max_gap=12)
            for i, part in enumerate(["observed", "trend", "seasonal", "residual"]):
                ax = axes[i][2 * c + m]
                ax.plot(parts.index, parts[part], color=color, linewidth=1.4)
                if part in ("seasonal", "residual"):
                    ax.axhline(0, color=AXIS, linewidth=1)
                if i == 0:
                    ax.set_title(f"{CITY_NAME[city]} — {label}")
                if c == 0 and m == 0:
                    ax.set_ylabel(part)
                ax.xaxis.set_major_locator(mpl.dates.YearLocator())
                ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    return finish(fig, "Splitting each series into trend, season and what is left",
                  "STL with one fixed annual shape. The residual row is what the calendar cannot explain. "
                  "Gaps of up to 12 weeks are interpolated for the fit; Melbourne's year-long gap is too long for that.",
                  "eda_stl")


# ------------------------------------------------------------------- bivariate

def scatter_lowess(long):
    """PM2.5 against headline search with a LOWESS curve."""
    fig, axes = _city_axes(height=3.6)
    for ax, city in zip(axes[0], CITIES):
        d = F.headline(long, city).dropna()
        ax.scatter(d["pm25_mean"], d["search"], s=18, color=CITY_COLOR[city], alpha=0.55, edgecolor="none")
        fit = lowess(d["search"], d["pm25_mean"], frac=0.6)
        ax.plot(fit[:, 0], fit[:, 1], color=INK, linewidth=2)
        r, rho = stats.pearsonr(d["pm25_mean"], d["search"])[0], stats.spearmanr(d["pm25_mean"], d["search"])[0]
        _note(ax, f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}", "upper left")
        ax.set_title(CITY_NAME[city])
        ax.set_xlabel("weekly PM2.5 (µg/m³)")
        ax.set_ylabel(f"search: {F.term_label(city, F.HEADLINE_TERMS[city]['awareness'])}")
    return finish(fig, "Search climbs with pollution in Bangkok and Kraków; Melbourne shows no clear relationship",
                  "Black line is a LOWESS smooth: it follows the data without assuming a linear relationship.",
                  "eda_scatter_lowess")


def pearson_vs_spearman(long):
    """Linear vs rank correlation for every dense series."""
    summ = F.series_summary(long)
    summ = summ[summ["dense"]]
    rows = []
    for _, s in summ.iterrows():
        d = F.pair(long, s["city"], s["term"], s["geo_level"]).dropna()
        rows.append({"city": s["city"], "label": s["label"],
                     "pearson": stats.pearsonr(d["pm25_mean"], d["search"])[0],
                     "spearman": stats.spearmanr(d["pm25_mean"], d["search"])[0]})
    res = pd.DataFrame(rows)
    fig, axes = _city_axes(height=3.4, sharex=True)
    for ax, city in zip(axes[0], CITIES):
        d = res[res["city"] == city].sort_values("pearson")
        ax.hlines(d["label"], d["pearson"], d["spearman"], color=AXIS, linewidth=2)
        ax.scatter(d["pearson"], d["label"], s=52, color=CITY_COLOR[city], label="Pearson (linear)", zorder=3)
        ax.scatter(d["spearman"], d["label"], s=52, facecolor=SURFACE, edgecolor=CITY_COLOR[city],
                   linewidth=2, label="Spearman (rank)", zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(CITY_NAME[city])
        ax.set_xlabel("correlation with weekly PM2.5")
        ax.set_xlim(-0.3, 1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    proxies = [mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=7, color=INK2, label="Pearson (linear)"),
               mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE,
                                markeredgecolor=INK2, markeredgewidth=2, label="Spearman (rank)")]
    fig.legend(handles=proxies, loc="upper right", ncols=2)
    finish(fig, "Correlation of every usable search series with PM2.5",
           "Filled = Pearson, hollow = Spearman. A gap between them signals a non-linear or outlier-driven relationship.",
           "eda_pearson_vs_spearman")
    return fig, res


def rolling_correlations(long, window=26):
    """Is the relationship stable through time?"""
    fig, ax = plt.subplots(figsize=(11, 4))
    for city in CITIES:
        r = F.rolling_correlation(F.headline(long, city), window)
        ax.plot(r.index, r, color=CITY_COLOR[city], label=CITY_NAME[city])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"{window}-week rolling correlation")
    ax.legend(loc="lower left", ncols=3)
    return finish(fig, "The pollution–search link is strong in Bangkok and Kraków, but not constant",
                  "Rolling Pearson correlation between PM2.5 and each city's headline search term. Breaks are data gaps.",
                  "eda_rolling_correlation")


def lag_profiles(long, max_lag=8):
    """Correlation at each lead and lag, awareness and protection terms."""
    fig, axes = _city_axes(height=3.4, sharey=True)
    for ax, city in zip(axes[0], CITIES):
        for role, color, style in [("awareness", CITY_COLOR[city], "o"), ("protection", INK2, "s")]:
            cc = F.lag_correlation(F.headline(long, city, role), max_lag)
            ax.plot(cc.index, cc, color=color, marker=style, markersize=4.5,
                    label=f"{role}: {F.term_label(city, F.HEADLINE_TERMS[city][role])}")
        ax.axvline(0, color=INK, linewidth=1)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(CITY_NAME[city])
        ax.set_xlabel("← search first      lag (weeks)      search after →")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(-0.45, 1.1)
    axes[0][0].set_ylabel("correlation with PM2.5")
    return finish(fig, "Search responds in the same week as the pollution",
                  "Correlation of search at week t+lag with PM2.5 at week t, on correctly aligned weeks.",
                  "eda_lag_profiles")


# ---------------------------------------------------------------- multivariate

def correlation_heatmaps(long):
    """Correlation matrix of PM2.5 and every dense search series."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(wspace=0.75))
    for ax, city in zip(axes, CITIES):
        corr = F.dense_wide(long, city).corr()
        corr.index = corr.columns = [c.replace("English, ", "EN, ") for c in corr.columns]
        n = len(corr)
        mesh = ax.pcolormesh(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, edgecolors=SURFACE, linewidth=2)
        for i in range(n):
            for j in range(n):
                v = corr.iat[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else INK)
        ax.set_xticks(np.arange(n) + 0.5, corr.columns, rotation=40, ha="right")
        ax.set_yticks(np.arange(n) + 0.5, corr.index)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(CITY_NAME[city])
    cb = fig.colorbar(mesh, ax=axes, fraction=0.015, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("Pearson correlation")
    fig.suptitle("How the search terms move with PM2.5 and with each other", x=0.01, y=0.99, ha="left",
                 va="top", fontsize=13, fontweight="bold")
    fig.savefig(FIGURES / "eda_correlation_heatmaps.png")
    return fig


def attention_index(long, city):
    """First principal component of a city's dense search series, oriented so higher = more searching."""
    wide = F.dense_wide(long, city)
    X = wide.drop(columns="PM2.5")
    Z = (X - X.mean()) / X.std()
    pca = PCA().fit(Z)
    pc1 = pca.transform(Z)[:, 0]
    if np.corrcoef(pc1, Z.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    return pd.DataFrame({"pm25_mean": wide["PM2.5"], "attention": pc1}, index=wide.index), pca.explained_variance_ratio_


def pca_summary(long):
    """Variance explained by each component, and PC1 against PM2.5."""
    fig, axes = _city_axes(nrows=2, height=2.9)
    for j, city in enumerate(CITIES):
        idx, ratio = attention_index(long, city)
        ax = axes[0][j]
        ax.bar(range(1, len(ratio) + 1), ratio * 100, width=0.6, color=CITY_COLOR[city])
        ax.set_xticks(range(1, len(ratio) + 1), [f"PC{k}" for k in range(1, len(ratio) + 1)])
        ax.set_ylabel("% of variance")
        ax.grid(axis="x", visible=False)
        ax.set_title(CITY_NAME[city])
        _note(ax, f"PC1 carries {ratio[0]:.0%}\nof {len(ratio)} series")
        ax = axes[1][j]
        ax.scatter(idx["pm25_mean"], idx["attention"], s=18, color=CITY_COLOR[city], alpha=0.55, edgecolor="none")
        _note(ax, f"r = {idx.corr().iat[0, 1]:.2f}", "upper left")
        ax.set_xlabel("weekly PM2.5 (µg/m³)")
        ax.set_ylabel("attention index (PC1)")
    return finish(fig, "In Bangkok and Kraków one 'attention' signal sits behind the separate search terms",
                  "PCA on each city's standardised search series. Bottom row: the first component against PM2.5.",
                  "eda_pca")


def regimes(long, k=3):
    """K-means on standardised PM2.5 and attention: what kinds of week are there?"""
    fig, axes = _city_axes(height=3.8)
    markers = ["o", "s", "^"]
    out = {}
    for ax, city in zip(axes[0], CITIES):
        idx, _ = attention_index(long, city)
        Z = (idx - idx.mean()) / idx.std()
        labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Z)
        order = np.argsort([Z["pm25_mean"][labels == c].mean() for c in range(k)])
        for rank, c in enumerate(order):
            m = labels == c
            ax.scatter(Z["pm25_mean"][m], Z["attention"][m], s=26, marker=markers[rank], color=CITY_COLOR[city],
                       alpha=0.35 + 0.3 * rank, edgecolor=SURFACE, linewidth=0.5,
                       label=f"{m.sum()} weeks: PM2.5 {Z['pm25_mean'][m].mean():+.1f}, attention {Z['attention'][m].mean():+.1f}")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(CITY_NAME[city])
        ax.set_xlabel("PM2.5 (standardised)")
        ax.set_ylabel("attention index (standardised)")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.4)
        ax.legend(loc="upper left", fontsize=8)
        out[city] = pd.Series(labels, index=idx.index)
    finish(fig, "Clustering weeks by pollution and attention",
           "K-means with three clusters per city. The legend gives each cluster's size and its average position, in standard deviations.",
           "eda_regimes")
    return fig, out
