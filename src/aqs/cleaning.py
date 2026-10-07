"""Raw pulls -> analysis-ready weekly tables.

Outputs
  data/interim/sensor_selection.csv   every candidate sensor, kept or dropped, and why
  data/interim/aq_weekly_all.csv      weekly PM2.5 per city (+ sensor/day counts)
  data/interim/trends_weekly_all.csv  weekly search interest, long format
  data/processed/search_long.csv      PM2.5 joined to every search series, long format
  data/processed/all_cities_combined.csv  PM2.5 + the three baseline terms, wide

Weeks run Sunday-Saturday and are labelled by their Sunday, because that is how
Google Trends reports them. PM2.5 is aggregated onto the same weeks so the two
series cover identical days.

Run:  python -m aqs.cleaning
"""
import pandas as pd

from .config import (BASELINE_TERMS, CITIES, DATE_FROM, DATE_TO, INTERIM, LOCAL_TERMS,
                     PROCESSED, RAW_AQ, RAW_TRENDS)

PM25_MAX = 500            # µg/m³; above this is an instrument fault, not air
MIN_DAY_COVERAGE = 50     # % of hours a sensor must report for its daily mean to count
MIN_SENSOR_DAYS = 0.5     # share of study days a sensor needs to join the city panel
MIN_DAYS_PER_WEEK = 4     # a weekly mean needs at least this many days behind it

STUDY_DAYS = (pd.Timestamp(DATE_TO) - pd.Timestamp(DATE_FROM)).days + 1


def week_start(dates):
    """Sunday on or before each date (Google Trends' week label)."""
    dates = pd.to_datetime(dates)
    return dates - pd.to_timedelta((dates.dt.weekday + 1) % 7, unit="D")


def valid_days(city):
    """Sensor-days that pass the value and coverage filters."""
    df = pd.read_csv(RAW_AQ / f"{city}_raw.csv", parse_dates=["date"])
    df = df[df["date"].between(DATE_FROM, DATE_TO)]
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= PM25_MAX)]
    return df[df["percent_complete"] >= MIN_DAY_COVERAGE]


def select_sensors(city, days):
    """Apply the panel rule: reference-grade monitors present for most of the window.

    Low-cost community sensors are excluded because they join part-way through
    and would shift the city average for reasons unrelated to air quality.
    """
    sensors = pd.read_csv(RAW_AQ / f"{city}_sensors.csv")
    sensors["valid_days"] = sensors["sensor_id"].map(days.groupby("sensor_id").size()).fillna(0).astype(int)
    sensors["valid_share"] = (sensors["valid_days"] / STUDY_DAYS).round(3)
    reason = pd.Series("kept", index=sensors.index)
    reason[sensors["valid_share"] < MIN_SENSOR_DAYS] = f"under {MIN_SENSOR_DAYS:.0%} of study days"
    reason[~sensors["is_monitor"].astype(bool)] = "not a reference-grade monitor"
    sensors["decision"] = reason
    return sensors


def weekly_pm25(city):
    days = valid_days(city)
    sensors = select_sensors(city, days)
    kept = sensors.loc[sensors["decision"] == "kept", "sensor_id"]
    days = days[days["sensor_id"].isin(kept)]
    # median across sensors per day is robust to one sensor misbehaving
    daily = days.groupby("date")["pm25"].agg(pm25="median", n_sensors="size").reset_index()
    daily["week"] = week_start(daily["date"])
    weekly = daily.groupby("week").agg(
        pm25_mean=("pm25", "mean"), n_days=("pm25", "size"), n_sensors=("n_sensors", "mean")
    ).reset_index()
    weekly = weekly[weekly["n_days"] >= MIN_DAYS_PER_WEEK]
    weekly["n_sensors"] = weekly["n_sensors"].round(1)
    weekly.insert(1, "city", city)
    return weekly, sensors


def term_roles(city):
    roles = {k: "awareness" for k in BASELINE_TERMS}
    roles.update({"air_purifier": "protection", "N95_mask": "protection"})
    roles.update({t["key"]: t["role"] for t in LOCAL_TERMS[city]})
    return roles


def weekly_trends(city):
    """Every pulled series for the city, long format, on Google's own Sunday weeks."""
    cfg, roles, frames = CITIES[city], term_roles(city), []
    for path in sorted(RAW_TRENDS.glob(f"{city}__*.csv")):
        _, geo, key = path.stem.split("__")
        df = pd.read_csv(path)
        df = df[~df["is_partial"]]
        frames.append(pd.DataFrame({
            "week": pd.to_datetime(df["date"].str[:10]),
            "city": city,
            "geo_level": "country" if geo == cfg["geo_country"] else "region",
            "term": key,
            "keyword": df["keyword"],
            "role": roles[key],
            "search": df["value"],
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    aq, sensors, trends = [], [], []
    for city in CITIES:
        weekly, sel = weekly_pm25(city)
        aq.append(weekly)
        sensors.append(sel)
        trends.append(weekly_trends(city))
        print(f"{city}: {len(weekly)} PM2.5 weeks from {(sel['decision'] == 'kept').sum()} of "
              f"{len(sel)} sensors; {trends[-1].groupby(['geo_level', 'term']).ngroups} search series")
    aq, sensors, trends = (pd.concat(x, ignore_index=True) for x in (aq, sensors, trends))

    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sensors.to_csv(INTERIM / "sensor_selection.csv", index=False)
    aq.to_csv(INTERIM / "aq_weekly_all.csv", index=False)
    trends.to_csv(INTERIM / "trends_weekly_all.csv", index=False)

    long = trends.merge(aq, on=["week", "city"], how="inner").sort_values(["city", "geo_level", "term", "week"])
    long.to_csv(PROCESSED / "search_long.csv", index=False)

    baseline = trends[(trends["geo_level"] == "region") & trends["term"].isin(BASELINE_TERMS)]
    wide = baseline.pivot(index=["week", "city"], columns="term", values="search").reset_index()
    wide = aq[["week", "city", "pm25_mean"]].merge(wide, on=["week", "city"], how="inner")
    wide = wide.sort_values(["city", "week"])[["week", "pm25_mean", "city", *BASELINE_TERMS]]
    wide.to_csv(PROCESSED / "all_cities_combined.csv", index=False)
    print(f"saved: {len(long)} long rows, {len(wide)} wide rows")


if __name__ == "__main__":
    main()
