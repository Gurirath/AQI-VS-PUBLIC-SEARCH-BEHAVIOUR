import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller

from . import features as F
from .config import CITIES
from .stats import bh_adjust, deseasonalize

MAX_LAG = 4
RF_PARAMS = dict(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)
PM_FEATURES = ["pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag4", "pm25_roll4"]
SEASON_FEATURES = ["month_sin", "month_cos"]


def stationarity_table(long):
    rows = []
    for city in CITIES:
        d = F.attention_pair(long, city)
        for label, frame in [("raw", d), ("season removed", deseasonalize(d))]:
            for col, name in [("pm25_mean", "PM2.5"), ("search", "attention")]:
                p = adfuller(frame[col].dropna())[1]
                rows.append({"city": city, "series": name, "version": label, "adf_p": p, "stationary": p < 0.05})
    return pd.DataFrame(rows)


def granger(d, cause, effect, lags):
    df = pd.DataFrame({"y": d[effect]})
    for k in range(1, lags + 1):
        df[f"own{k}"] = d[effect].shift(k)
        df[f"other{k}"] = d[cause].shift(k)
    df = df.dropna()
    own = [f"own{k}" for k in range(1, lags + 1)]
    other = [f"other{k}" for k in range(1, lags + 1)]
    restricted = sm.OLS(df["y"], sm.add_constant(df[own])).fit()
    full = sm.OLS(df["y"], sm.add_constant(df[own + other])).fit()
    f, p, _ = full.compare_f_test(restricted)
    return {"weeks": len(df), "F": f, "p": p}


def granger_table(long, role="attention"):
    rows = []
    for city in CITIES:
        res = deseasonalize(F.role_pair(long, city, role))
        for direction, cause, effect in [(f"PM2.5 → {role}", "pm25_mean", "search"),
                                         (f"{role} → PM2.5 (placebo)", "search", "pm25_mean")]:
            for lags in range(1, MAX_LAG + 1):
                rows.append({"city": city, "direction": direction, "lags": lags, **granger(res, cause, effect, lags)})
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_adjust(out["p"])
    out["significant"] = out["p_fdr"] < 0.05
    return out


def model_frame(d):
    df = pd.DataFrame({"attention": d["search"], "pm25": d["pm25_mean"]})
    for k in range(1, 5):
        df[f"pm25_lag{k}"] = d["pm25_mean"].shift(k)
    df["pm25_roll4"] = d["pm25_mean"].rolling(4).mean()
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return df.dropna()


def out_of_fold(df, features, params=RF_PARAMS, splits=5):
    pred = pd.Series(np.nan, index=df.index)
    for train, test in TimeSeriesSplit(n_splits=splits).split(df):
        rf = RandomForestRegressor(**params).fit(df.iloc[train][features], df.iloc[train]["attention"])
        pred.iloc[test] = rf.predict(df.iloc[test][features])
    return pred.dropna()


def forecast_skill(long, depths=(2, 4, 6), role="attention"):
    rows, preds = [], {}
    for city in CITIES:
        df = model_frame(F.role_pair(long, city, role))
        row = {"city": city, "weeks": len(df)}
        for name, feats in [("calendar", SEASON_FEATURES), ("calendar_pm25", SEASON_FEATURES + PM_FEATURES)]:
            pred = out_of_fold(df, feats)
            actual = df.loc[pred.index, "attention"]
            row[f"r2_{name}"] = r2_score(actual, pred)
            row[f"mae_{name}"] = mean_absolute_error(actual, pred)
            preds[(city, name)] = pred
        row["r2_gain_from_pm25"] = row["r2_calendar_pm25"] - row["r2_calendar"]
        for depth in depths:
            pred = out_of_fold(df, SEASON_FEATURES + PM_FEATURES, {**RF_PARAMS, "max_depth": depth})
            row[f"r2_depth{depth}"] = r2_score(df.loc[pred.index, "attention"], pred)
        rf = RandomForestRegressor(**RF_PARAMS).fit(df[SEASON_FEATURES + PM_FEATURES], df["attention"])
        imp = pd.Series(rf.feature_importances_, index=SEASON_FEATURES + PM_FEATURES)
        row["importance_pm25"] = imp[PM_FEATURES].sum()
        row["top_feature"] = imp.idxmax()
        preds[(city, "actual")] = df["attention"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("city"), preds


def verdicts(corr, grang, skill, card, role="attention"):
    rows = []
    for city in CITIES:
        c = corr[(corr["city"] == city) & (corr["series"] == F.role_label(city, role))].iloc[0]
        g = grang[grang["city"] == city]
        forward = g["direction"].str.startswith("PM2.5")
        fwd, rev = g[forward], g[~forward]
        rows.append({
            "city": city,
            "deseason_r": c["deseason_r"], "correlation_survives": bool(c["significant"]),
            "granger_forward_min_p_fdr": fwd["p_fdr"].min(), "granger_forward": bool(fwd["significant"].any()),
            "granger_placebo_min_p_fdr": rev["p_fdr"].min(), "granger_placebo": bool(rev["significant"].any()),
            "r2_gain_from_pm25": skill.loc[city, "r2_gain_from_pm25"],
            "silent_share": card.loc[city, "silent_share"],
            "habituation_per_week": card.loc[city, "habituation_per_week"],
        })
    return pd.DataFrame(rows).set_index("city")
