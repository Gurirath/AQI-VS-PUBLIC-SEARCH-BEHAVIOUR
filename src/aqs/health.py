import numpy as np
import pandas as pd

from . import features as F
from .config import CITIES, HEADLINE_TERMS, default_geo_level
from .stats import RNG_SEED, _corr, deseasonalize

ROLES = ("awareness", "protection", "health")
MAX_LAG = 8
MIN_YEAR_WEEKS = 20


def with_drop(table):
    return table.assign(drop=table["raw_r"] - table["deseason_r"])


def health_series(table):
    return with_drop(table[table["role"] == "health"]).reset_index(drop=True)


def role_contrast(table):
    rows = []
    for city in CITIES:
        for role in ROLES:
            term = HEADLINE_TERMS[city][role]
            rows.append(table[(table["city"] == city) & (table["term"] == term)
                              & (table["geo_level"] == default_geo_level(city, term))])
    return with_drop(pd.concat(rows)).reset_index(drop=True)


def confound_summary(table):
    t = with_drop(table[table["role"] != "attention"])
    out = t.groupby(["city", "role"], sort=False).agg(
        series=("series", "size"), median_raw_r=("raw_r", "median"), median_deseason_r=("deseason_r", "median"),
        max_raw_r=("raw_r", "max"), max_deseason_r=("deseason_r", "max"), significant=("significant", "sum")
    ).reset_index()
    order = out["city"].map(list(CITIES).index) * 10 + out["role"].map(F.ROLE_ORDER.index)
    return out.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)


def peak_lag_ci(d, max_lag=MAX_LAG, n_boot=2000, block=8, seed=RNG_SEED):
    res = deseasonalize(d)
    profile = F.lag_correlation(res, max_lag)
    lags = profile.index.to_numpy()
    pm = res["pm25_mean"].to_numpy()
    shifted = np.column_stack([res["search"].shift(-k).to_numpy() for k in lags])
    n = len(pm)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    peaks = np.array([lags[np.nanargmax([_corr(pm[i], shifted[i, j]) for j in range(len(lags))])] for i in idx])
    low, high = np.percentile(peaks, [2.5, 97.5])
    peak = F.peak_lag(profile)
    return {"weeks": int(res.dropna().shape[0]), "peak_lag": peak,
            "peak_r": profile.max() if peak is not None else np.nan,
            "ci_low": low if abs(low) < max_lag else np.nan, "ci_high": high if abs(high) < max_lag else np.nan,
            "interior_share": float((np.abs(peaks) < max_lag).mean()),
            "same_week_share": float((peaks == 0).mean())}


def cascade(long):
    rows = []
    for city in CITIES:
        for role in ROLES:
            rows.append({"city": city, "role": role, "series": F.role_label(city, role),
                         **peak_lag_ci(F.headline(long, city, role))})
    out = pd.DataFrame(rows)
    for col in ["peak_lag", "ci_low", "ci_high"]:
        out[col] = out[col].astype("Int64")
    return out


def yearly_correlation(d, min_weeks=MIN_YEAR_WEEKS):
    res = deseasonalize(d).dropna()
    rows = []
    for year, g in res.groupby(res.index.year):
        enough = len(g) >= min_weeks
        rows.append({"year": int(year), "weeks": len(g), "skipped": not enough,
                     "deseason_r": g["pm25_mean"].corr(g["search"]) if enough else np.nan})
    return pd.DataFrame(rows)


def yoy(long):
    frames = []
    for city in CITIES:
        for role in ROLES:
            part = yearly_correlation(F.headline(long, city, role))
            part.insert(0, "series", F.role_label(city, role))
            part.insert(0, "role", role)
            part.insert(0, "city", city)
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


def summary_matrices(table):
    extra = table.assign(peak=table["peak_lag"].astype(float), sig=table["significant"].astype(float))
    return {"deseason_r": F.slot_matrix(table, "deseason_r"), "raw_r": F.slot_matrix(table, "raw_r"),
            "peak_lag": F.slot_matrix(extra, "peak"), "present": F.slot_matrix(table, "weeks").notna(),
            "significant": F.slot_matrix(extra, "sig") == 1}
