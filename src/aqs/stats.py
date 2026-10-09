"""Statistics that hold up against shared seasonality and autocorrelation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import features as F
from .config import CITIES

RNG_SEED = 7


def seasonal_design(index, harmonics=3):
    """Intercept, linear trend and annual sine/cosine pairs for a weekly index."""
    t = (index - index[0]).days / 365.25
    cols = {"const": np.ones(len(index)), "trend": t}
    for k in range(1, harmonics + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t)
    return pd.DataFrame(cols, index=index)


def expected_and_surprise(s, harmonics=3):
    """Split a series into what the calendar predicts and what it does not.

    A harmonic regression (trend + annual sine waves) is fitted to the weeks
    that exist, so unlike STL it needs no interpolation across gaps."""
    X = seasonal_design(s.index, harmonics)
    ok = s.notna()
    fit = sm.OLS(s[ok], X[ok]).fit()
    expected = pd.Series(fit.predict(X), index=s.index)
    return pd.DataFrame({"expected": expected, "surprise": s - expected})


def deseasonalize(d, harmonics=3):
    """Seasonal residuals of both columns of a PM2.5/search pair."""
    return pd.DataFrame({c: expected_and_surprise(d[c], harmonics)["surprise"] for c in ["pm25_mean", "search"]})


def _corr(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(x[ok], y[ok])[0, 1]


def shift_pvalue(x, y, min_shift=8):
    """Two-sided p-value from circular shifts of one series against the other.

    Shifting keeps each series' own autocorrelation and seasonality intact
    while breaking the week-to-week pairing, so the null distribution reflects
    how large a correlation two unrelated but equally persistent series give."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = _corr(x, y)
    null = np.array([_corr(x, np.roll(y, k)) for k in range(min_shift, len(y) - min_shift + 1)])
    return (np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)


def block_bootstrap_ci(x, y, block=8, n_boot=2000, seed=RNG_SEED):
    """95% interval for a correlation, resampling 8-week blocks of weeks so
    that the dependence between neighbouring weeks is carried into each resample."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    r = np.array([_corr(x[i], y[i]) for i in idx])
    return np.nanpercentile(r, [2.5, 97.5])


def bh_adjust(p):
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _series_row(city, label, d):
    res = deseasonalize(d)
    both = d.dropna()
    lags = F.lag_correlation(res.rename(columns={"pm25_mean": "pm25_mean", "search": "search"}))
    lo, hi = block_bootstrap_ci(res["pm25_mean"], res["search"])
    return {
        "city": city, "series": label, "weeks": len(both),
        "raw_r": stats.pearsonr(both["pm25_mean"], both["search"])[0],
        "raw_spearman": stats.spearmanr(both["pm25_mean"], both["search"])[0],
        "deseason_r": _corr(res["pm25_mean"].to_numpy(), res["search"].to_numpy()),
        "ci_low": lo, "ci_high": hi,
        "p_shift": shift_pvalue(res["pm25_mean"], res["search"]),
        "peak_lag": int(lags.idxmax()),
    }


def correlation_table(long):
    """Raw and deseasonalised correlation with PM2.5 for every dense series
    and for each city's attention index, with FDR-adjusted p-values."""
    summ = F.series_summary(long)
    rows = []
    for city in CITIES:
        rows.append(_series_row(city, "attention index (PC1)", F.attention_pair(long, city)))
        for _, s in summ[(summ["city"] == city) & summ["dense"]].iterrows():
            rows.append(_series_row(city, s["label"], F.pair(long, city, s["term"], s["geo_level"])))
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_adjust(out["p_shift"])
    out["significant"] = out["p_fdr"] < 0.05
    return out


def tertile_test(d):
    """Kruskal-Wallis test of search across low/mid/high PM2.5 weeks, with
    epsilon-squared as the effect size. Rank-based, so skew does not distort it."""
    d = d.dropna()
    bins = pd.qcut(d["pm25_mean"], 3, labels=["low", "mid", "high"])
    groups = [d.loc[bins == b, "search"] for b in ["low", "mid", "high"]]
    h, p = stats.kruskal(*groups)
    n = len(d)
    return {"H": h, "p": p, "epsilon_sq": (h - 2) / (n - 3), **{f"median_{b}": g.median() for b, g in zip(["low", "mid", "high"], groups)}}
