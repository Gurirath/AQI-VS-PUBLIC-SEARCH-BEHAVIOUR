"""Pull weekly Google Trends interest for every study city.

Two geographies per city:
  - country level, baseline English terms only (reproduces the first iteration)
  - region level (the city's own state/province), baseline + local-language terms

Each series is saved untouched (Google's own week-start dates) to
data/raw/trends/{city}__{geo}__{key}.csv. Existing files are skipped, so the
script can be re-run until every series has been fetched.

Google allows roughly eight fresh Explore sessions an hour per IP, so the pull
is paced at one series every PAUSE_S seconds and sits out COOLDOWN_S after a
refusal. A full pull takes about three hours.

Run:  python -m aqs.collect_trends [city ...]
"""
import sys
import time

import pandas as pd
from trendspyg import RateLimitError, download_google_trends_interest_over_time
from trendspyg.config import US_STATES

from .config import BASELINE_TERMS, CITIES, LOCAL_TERMS, RAW_TRENDS, TRENDS_TIMEFRAME

PAUSE_S = 450
COOLDOWN_S = 35 * 60

# trendspyg only whitelists countries and US states, but passes the code to
# Google unchanged, and Google accepts any ISO 3166-2 region.
US_STATES.update({cfg["geo_region"]: city for city, cfg in CITIES.items()})


def series_plan(city):
    """(geo, key, keyword) for every series this city needs."""
    cfg = CITIES[city]
    plan = [(cfg["geo_region"], t["key"], t["keyword"]) for t in LOCAL_TERMS[city]]
    plan += [(cfg["geo_region"], k, kw) for k, kw in BASELINE_TERMS.items()]
    plan += [(cfg["geo_country"], k, kw) for k, kw in BASELINE_TERMS.items()]
    return plan


def main(argv):
    cities = argv or list(CITIES)
    RAW_TRENDS.mkdir(parents=True, exist_ok=True)
    failed = []
    for city in cities:
        for geo, key, keyword in series_plan(city):
            out = RAW_TRENDS / f"{city}__{geo}__{key}.csv"
            if out.exists():
                continue
            print(f"{city} {geo} '{keyword}'...", end=" ", flush=True)
            try:
                points = download_google_trends_interest_over_time(
                    keyword, geo=geo, timeframe=TRENDS_TIMEFRAME, cookies="disk"
                )
                df = pd.DataFrame(points)
                df["keyword"], df["geo"] = keyword, geo
                df.to_csv(out, index=False)
                print(f"{len(df)} weeks", flush=True)
            except RateLimitError:
                failed.append(out.name)
                print(f"rate limited, cooling down {COOLDOWN_S // 60} min", flush=True)
                time.sleep(COOLDOWN_S)
                continue
            except Exception as e:
                failed.append(out.name)
                print(f"FAILED ({type(e).__name__}: {str(e).splitlines()[0]})", flush=True)
            time.sleep(PAUSE_S)
    if failed:
        print(f"\n{len(failed)} series failed, re-run to retry: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
