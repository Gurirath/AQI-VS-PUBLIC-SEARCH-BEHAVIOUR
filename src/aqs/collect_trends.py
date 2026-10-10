import sys
import time

import pandas as pd
from trendspyg import RateLimitError, download_google_trends_interest_over_time
from trendspyg.config import US_STATES

from .config import BASELINE_TERMS, CITIES, LOCAL_TERMS, RAW_TRENDS, ROOT, TRENDS_TIMEFRAME, health_terms

PAUSE_S = 450
COOLDOWN_S = 35 * 60

US_STATES.update({cfg["geo_region"]: city for city, cfg in CITIES.items()})


def series_plan(city):
    cfg = CITIES[city]
    plan = [(cfg["geo_region"], t["key"], t["keyword"]) for t in LOCAL_TERMS[city]]
    plan += [(cfg["geo_region"], k, kw) for k, kw in BASELINE_TERMS.items()]
    plan += [(cfg["geo_country"], k, kw) for k, kw in BASELINE_TERMS.items()]
    plan += [(cfg["geo_region"], t["key"], t["keyword"]) for t in health_terms(city)]
    return plan


def missing_series(cities):
    missing = []
    for city in cities:
        for geo, key, keyword in series_plan(city):
            out = RAW_TRENDS / f"{city}__{geo}__{key}.csv"
            if not out.exists():
                missing.append((city, geo, key, keyword, out))
    return missing


def dry_run(cities):
    missing = missing_series(cities)
    total = sum(len(series_plan(city)) for city in cities)
    for city, geo, key, keyword, _ in missing:
        print(f"{city} {geo} {key} '{keyword}'")
    seconds = len(missing) * PAUSE_S
    print(f"\n{len(missing)} of {total} series missing from {RAW_TRENDS.relative_to(ROOT)}")
    print(f"estimated time at one series every {PAUSE_S}s: {seconds // 3600}h {seconds % 3600 // 60:02d}m "
          f"(plus {COOLDOWN_S // 60} min for each rate-limit refusal)")


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cities = [a for a in argv if not a.startswith("--")] or list(CITIES)
    if "--dry-run" in argv:
        dry_run(cities)
        return
    RAW_TRENDS.mkdir(parents=True, exist_ok=True)
    failed = []
    for city, geo, key, keyword, out in missing_series(cities):
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
