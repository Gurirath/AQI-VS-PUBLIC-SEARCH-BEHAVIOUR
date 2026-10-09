"""Loading and derived series shared by the notebooks and the dashboard."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import STL

from .config import BASELINE_TERMS, HEADLINE_TERMS, LOCAL_TERMS, MAX_ZERO_SHARE, PROCESSED, INTERIM

WEEK = "7D"


def load_long():
    return pd.read_csv(PROCESSED / "search_long.csv", parse_dates=["week"])


def load_pm25():
    return pd.read_csv(INTERIM / "aq_weekly_all.csv", parse_dates=["week"])


def term_label(city, term, geo_level="region"):
    """Readable English name for a search series."""
    local = {t["key"]: t["label"] for t in LOCAL_TERMS[city]}
    if term in local:
        return local[term]
    return f"{BASELINE_TERMS[term]} (English, {geo_level})"


def pair(long, city, term, geo_level="region"):
    """PM2.5 and one search series on a regular weekly index (gaps stay NaN)."""
    d = long[(long["city"] == city) & (long["term"] == term) & (long["geo_level"] == geo_level)]
    return d.set_index("week")[["pm25_mean", "search"]].asfreq(WEEK)


def headline(long, city, role="awareness"):
    return pair(long, city, HEADLINE_TERMS[city][role])


def series_summary(long):
    """One row per search series: length, zero share, and whether it is dense enough to analyse."""
    g = long.groupby(["city", "geo_level", "term"])["search"]
    out = g.agg(weeks="size", zero_share=lambda s: (s == 0).mean()).reset_index()
    out["label"] = [term_label(c, t, g) for c, t, g in zip(out["city"], out["term"], out["geo_level"])]
    out["dense"] = out["zero_share"] <= MAX_ZERO_SHARE
    return out


def dense_wide(long, city):
    """PM2.5 plus every dense search series for a city, one column each."""
    summ = series_summary(long)
    keep = summ[(summ["city"] == city) & summ["dense"]]
    cols = {"PM2.5": pair(long, city, keep["term"].iloc[0], keep["geo_level"].iloc[0])["pm25_mean"]}
    for _, r in keep.iterrows():
        cols[r["label"]] = pair(long, city, r["term"], r["geo_level"])["search"]
    return pd.DataFrame(cols).dropna()


def robust_z(x):
    """Z-score using median and MAD, so outliers do not hide themselves."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * mad)


def seasonal_anomaly(s):
    """Deviation from the month-of-year median. Works on gappy series, unlike STL."""
    return s - s.groupby(s.index.month).transform("median")


def stl_parts(s, max_gap=8):
    """STL trend/seasonal/residual. Gaps up to max_gap weeks are interpolated;
    returns None when the series has longer holes than that.

    The seasonal smoother is set wide enough to force one fixed annual shape.
    With only three years of data the default lets the seasonal component
    change every year, which swallows the residual it is meant to expose."""
    gaps = s.isna().astype(int).groupby(s.notna().cumsum()).sum().max()
    if gaps > max_gap:
        return None
    fit = STL(s.interpolate(limit_direction="both"), period=52, seasonal=1001).fit()
    parts = pd.DataFrame({"observed": s, "trend": fit.trend, "seasonal": fit.seasonal, "residual": fit.resid})
    parts.loc[s.isna(), "residual"] = np.nan
    return parts


def lag_correlation(d, max_lag=8):
    """Correlation of search at week t+k with PM2.5 at week t.
    Positive k: search comes after pollution. Shifting on the regular index
    keeps weeks either side of a gap from being paired up."""
    return pd.Series({k: d["search"].shift(-k).corr(d["pm25_mean"]) for k in range(-max_lag, max_lag + 1)})


def rolling_correlation(d, window=26, min_periods=20):
    return d["pm25_mean"].rolling(window, min_periods=min_periods).corr(d["search"])


def attention_index(long, city):
    """First principal component of a city's dense search series, oriented so
    higher = more searching. Returns (PM2.5 + index on dense weeks, variance shares)."""
    wide = dense_wide(long, city)
    X = wide.drop(columns="PM2.5")
    Z = (X - X.mean()) / X.std()
    pca = PCA().fit(Z)
    pc1 = pca.transform(Z)[:, 0]
    if np.corrcoef(pc1, Z.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    idx = pd.DataFrame({"pm25_mean": wide["PM2.5"], "attention": pc1}, index=wide.index)
    return idx, pca.explained_variance_ratio_


def attention_pair(long, city):
    """PM2.5 and the standardised attention index on a regular weekly index,
    in the same shape as pair() so the two are interchangeable."""
    idx, _ = attention_index(long, city)
    idx["search"] = (idx["attention"] - idx["attention"].mean()) / idx["attention"].std()
    return idx[["pm25_mean", "search"]].asfreq(WEEK)
