import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import features as F
from .config import CITIES
from .stats import RNG_SEED, expected_and_surprise

HAC_LAGS = 8
EPISODE_Z = 2.0
EPISODE_QUIET = 3
WINDOW = (-4, 8)
HIGH_QUANTILE = 2 / 3
SEASON_START = {"bangkok": 7, "krakow": 7, "melbourne": 1, "lahore": 10}


def _hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def level_vs_surprise(d):
    parts = expected_and_surprise(d["pm25_mean"]) / 10
    df = pd.concat([d["search"], parts], axis=1).dropna()
    fit = _hac(df["search"], df[["expected", "surprise"]])
    ci = fit.conf_int()
    return pd.DataFrame({"effect": fit.params, "low": ci[0], "high": ci[1], "p": fit.pvalues}).loc[["expected", "surprise"]]


def _hinge_sse(pm, y, tau):
    X = np.column_stack([np.ones(len(pm)), np.clip(pm - tau, 0, None)])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum()), beta


def _best_hinge(pm, y):
    grid = np.quantile(pm, np.linspace(0.05, 0.85, 41))
    fits = [(_hinge_sse(pm, y, t), t) for t in grid]
    (sse, beta), tau = min(fits, key=lambda f: f[0][0])
    return tau, beta, sse


def attention_threshold(d, n_boot=500, block=8, seed=RNG_SEED):
    df = d.dropna()
    pm, y = df["pm25_mean"].to_numpy(), df["search"].to_numpy()
    n = len(pm)
    tau, beta, sse = _best_hinge(pm, y)
    lin = np.polyfit(pm, y, 1)
    sse_lin = float(((y - np.polyval(lin, pm)) ** 2).sum())
    aic = lambda s, k: n * np.log(s / n) + 2 * k
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    taus = [_best_hinge(pm[i], y[i])[0] for i in idx]
    return {"threshold": tau, "low": np.percentile(taus, 2.5), "high": np.percentile(taus, 97.5),
            "base": beta[0], "slope_per_10": beta[1] * 10,
            "aic_gain_vs_linear": aic(sse_lin, 2) - aic(sse, 3)}


def episodes(d):
    z = F.robust_z(expected_and_surprise(d["pm25_mean"])["surprise"])
    hot = (z >= EPISODE_Z).fillna(False)
    recent = hot.shift(1).rolling(EPISODE_QUIET, min_periods=1).max().fillna(0).astype(bool)
    return list(d.index[hot & ~recent])


def episode_windows(d):
    rows = []
    offsets = range(WINDOW[0], WINDOW[1] + 1)
    for onset in episodes(d):
        win = d.reindex([onset + pd.Timedelta(weeks=k) for k in offsets])
        pre = win.iloc[: -WINDOW[0]]
        if pre["search"].notna().sum() < 2 or pre["pm25_mean"].notna().sum() < 2:
            continue
        rel = win - pre.mean()
        rel.index = pd.Index(offsets, name="week")
        rows.append(rel.assign(onset=onset).reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["week", "pm25_mean", "search", "onset"])


def episode_response(d, n_boot=2000, seed=RNG_SEED):
    win = episode_windows(d)
    out = {}
    rng = np.random.default_rng(seed)
    for col in ["pm25_mean", "search"]:
        mat = win.pivot(index="onset", columns="week", values=col).to_numpy()
        boots = np.array([np.nanmean(mat[rng.integers(0, len(mat), len(mat))], axis=0) for _ in range(n_boot)])
        out[col] = pd.DataFrame({"mean": np.nanmean(mat, axis=0), "low": np.nanpercentile(boots, 2.5, axis=0),
                                 "high": np.nanpercentile(boots, 97.5, axis=0)},
                                index=range(WINDOW[0], WINDOW[1] + 1))
    curve = out["search"]["mean"].loc[0:]
    peak = int(curve.idxmax())
    after = curve.loc[peak:]
    faded = after[after <= curve.max() / 2]
    out["summary"] = {"episodes": win["onset"].nunique(), "peak_week": peak, "peak_response": curve.max(),
                      "peak_low": out["search"]["low"].loc[peak], "peak_high": out["search"]["high"].loc[peak],
                      "weeks_to_half": int(faded.index[0] - peak) if len(faded) else np.nan}
    return out


def with_season(d, city):
    df = d.copy()
    start = SEASON_START[city]
    df["season"] = df.index.year - (df.index.month < start)
    df["high"] = df["pm25_mean"] > df["pm25_mean"].quantile(HIGH_QUANTILE)
    df["prior_high_weeks"] = df.groupby("season")["high"].cumsum() - df["high"]
    return df


def habituation(d, city):
    df = with_season(d, city)
    df = df[df["high"]].dropna(subset=["search", "pm25_mean"])

    def fit(part):
        X = pd.DataFrame({"pm25_per_10": part["pm25_mean"] / 10, "prior_high_weeks": part["prior_high_weeks"]})
        return _hac(part["search"], X)

    full = fit(df)
    ci = full.conf_int().loc["prior_high_weeks"]
    loso = [fit(df[df["season"] != s]).params["prior_high_weeks"]
            for s, n in df["season"].value_counts().items() if n >= 5]
    return {"high_weeks": len(df), "effect_per_week": full.params["prior_high_weeks"], "low": ci[0], "high": ci[1],
            "p": full.pvalues["prior_high_weeks"], "leave_one_season_min": min(loso), "leave_one_season_max": max(loso)}


def silent_weeks(d, city, n_boot=2000, block=8, seed=RNG_SEED):
    df = with_season(d, city).dropna(subset=["search", "pm25_mean"])
    df["silent"] = df["high"] & (df["search"] < df["search"].median())
    high = df[df["high"]]
    rng = np.random.default_rng(seed)
    n = len(high)
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n].clip(max=n - 1)
    shares = high["silent"].to_numpy()[idx].mean(axis=1)
    late = high["prior_high_weeks"] >= high["prior_high_weeks"].median()
    return df, {"high_weeks": n, "silent_weeks": int(high["silent"].sum()), "silent_share": high["silent"].mean(),
                "low": np.percentile(shares, 2.5), "high": np.percentile(shares, 97.5),
                "silent_share_early": high.loc[~late, "silent"].mean(), "silent_share_late": high.loc[late, "silent"].mean()}


def scorecard(long, role="attention"):
    rows = []
    for city in CITIES:
        d = F.role_pair(long, city, role)
        ls, th, ep, hb = level_vs_surprise(d), attention_threshold(d), episode_response(d)["summary"], habituation(d, city)
        _, sw = silent_weeks(d, city)
        rows.append({
            "city": city,
            "surprise_effect": ls.loc["surprise", "effect"], "surprise_p": ls.loc["surprise", "p"],
            "expected_effect": ls.loc["expected", "effect"], "expected_p": ls.loc["expected", "p"],
            "threshold": th["threshold"], "threshold_low": th["low"], "threshold_high": th["high"],
            "threshold_aic_gain": th["aic_gain_vs_linear"],
            "episodes": ep["episodes"], "peak_week": ep["peak_week"], "peak_response": ep["peak_response"],
            "peak_low": ep["peak_low"], "peak_high": ep["peak_high"], "weeks_to_half": ep["weeks_to_half"],
            "habituation_per_week": hb["effect_per_week"], "habituation_p": hb["p"],
            "habituation_loso_min": hb["leave_one_season_min"], "habituation_loso_max": hb["leave_one_season_max"],
            "silent_share": sw["silent_share"], "silent_low": sw["low"], "silent_high": sw["high"],
            "silent_early": sw["silent_share_early"], "silent_late": sw["silent_share_late"],
        })
    return pd.DataFrame(rows).set_index("city")
