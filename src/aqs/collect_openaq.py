import os
import sys
import time

import certifi
import openaq
import pandas as pd
from dotenv import load_dotenv

from .config import CITIES, DATE_FROM, DATE_TO, PM25_PARAMETER_ID, RADIUS_M, RAW_AQ, ROOT

PAGE_SIZE = 1000


def _call(fn, **kwargs):
    for attempt in range(5):
        try:
            result = fn(**kwargs)
            time.sleep(1.1)
            return result
        except Exception as e:
            wait = 65 if "rate" in type(e).__name__.lower() else 10 * (attempt + 1)
            print(f"    {type(e).__name__}: {e} — retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up after 5 attempts: {kwargs}")


def discover_sensors(client, city):
    cfg = CITIES[city]
    sensors, page = [], 1
    while True:
        resp = _call(
            client.locations.list,
            coordinates=(cfg["lat"], cfg["lon"]),
            radius=RADIUS_M,
            parameters_id=[PM25_PARAMETER_ID],
            limit=PAGE_SIZE,
            page=page,
        )
        for loc in resp.results:
            first = loc.datetime_first.utc[:10] if loc.datetime_first else None
            last = loc.datetime_last.utc[:10] if loc.datetime_last else None
            if not first or not last or first > DATE_TO or last < DATE_FROM:
                continue
            for s in loc.sensors:
                if s.parameter.name == "pm25":
                    sensors.append({
                        "city": city,
                        "location_id": loc.id,
                        "location_name": loc.name,
                        "sensor_id": s.id,
                        "is_monitor": loc.is_monitor,
                        "provider": loc.provider.name if loc.provider else None,
                        "location_first": first,
                        "location_last": last,
                    })
        if len(resp.results) < PAGE_SIZE:
            return sensors
        page += 1


def pull_sensor(client, sensor_id):
    records, page = [], 1
    while True:
        resp = _call(
            client.measurements.list,
            sensors_id=sensor_id,
            data="days",
            date_from=DATE_FROM,
            date_to=DATE_TO,
            limit=PAGE_SIZE,
            page=page,
        )
        for m in resp.results:
            records.append({
                "sensor_id": sensor_id,
                "date": m.period.datetime_from.local[:10],
                "pm25": m.value,
                "percent_complete": m.coverage.percent_complete if m.coverage else None,
            })
        if len(resp.results) < PAGE_SIZE:
            return records
        page += 1


def collect_city(client, city, force=False):
    out = RAW_AQ / f"{city}_raw.csv"
    manifest_out = RAW_AQ / f"{city}_sensors.csv"
    if out.exists() and not force:
        print(f"{city}: {out.relative_to(ROOT)} exists, skipping (use --force)")
        return
    sensors = discover_sensors(client, city)
    print(f"{city}: {len(sensors)} candidate PM2.5 sensors", flush=True)
    frames = []
    for s in sensors:
        recs = pull_sensor(client, s["sensor_id"])
        s["n_days"] = len(recs)
        if recs:
            df = pd.DataFrame(recs)
            df["location_id"] = s["location_id"]
            s["first_day"], s["last_day"] = df["date"].min(), df["date"].max()
            s["mean_percent_complete"] = round(df["percent_complete"].mean(), 1)
            frames.append(df)
        print(f"  sensor {s['sensor_id']} ({s['location_name']}): {len(recs)} days", flush=True)
    RAW_AQ.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensors).to_csv(manifest_out, index=False)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw.to_csv(out, index=False)
    print(f"{city}: saved {len(raw)} sensor-days from {len(frames)} sensors", flush=True)


def collect(cities=None, force=False):
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    client = openaq.OpenAQ(api_key=os.environ["OPENAQ_API_KEY"])
    try:
        for city in cities or list(CITIES):
            collect_city(client, city, force=force)
    finally:
        client.close()


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    collect([a for a in argv if not a.startswith("--")], force="--force" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
