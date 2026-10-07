"""Shared project configuration: study window, cities, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_AQ = ROOT / "data" / "raw" / "openaq"
RAW_TRENDS = ROOT / "data" / "raw" / "trends"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"

DATE_FROM = "2022-01-01"
DATE_TO = "2024-12-31"

# OpenAQ station search: every PM2.5 sensor within RADIUS_M of the city centre.
RADIUS_M = 25_000
PM25_PARAMETER_ID = 2

CITIES = {
    "bangkok": {"lat": 13.7563, "lon": 100.5018, "geo_country": "TH", "geo_region": "TH-10"},
    "krakow": {"lat": 50.0647, "lon": 19.9450, "geo_country": "PL", "geo_region": "PL-MA"},
    "melbourne": {"lat": -37.8136, "lon": 144.9631, "geo_country": "AU", "geo_region": "AU-VIC"},
}

TRENDS_TIMEFRAME = f"{DATE_FROM} {DATE_TO}"

# The three English terms from the first iteration, kept as the baseline.
BASELINE_TERMS = {"air_quality": "air quality", "air_purifier": "air purifier", "N95_mask": "N95 mask"}

# Terms people actually type locally, grouped by the behaviour they signal:
# awareness (looking up the problem) vs. protection (acting on it).
LOCAL_TERMS = {
    "bangkok": [
        {"key": "pm25_th", "keyword": "PM2.5", "role": "awareness"},
        {"key": "dust_th", "keyword": "ฝุ่น", "role": "awareness"},
        {"key": "air_quality_th", "keyword": "คุณภาพอากาศ", "role": "awareness"},
        {"key": "air_purifier_th", "keyword": "เครื่องฟอกอากาศ", "role": "protection"},
        {"key": "mask_th", "keyword": "หน้ากาก N95", "role": "protection"},
    ],
    "krakow": [
        {"key": "smog_pl", "keyword": "smog", "role": "awareness"},
        {"key": "air_quality_pl", "keyword": "jakość powietrza", "role": "awareness"},
        {"key": "air_purifier_pl", "keyword": "oczyszczacz powietrza", "role": "protection"},
        {"key": "mask_pl", "keyword": "maska antysmogowa", "role": "protection"},
    ],
    "melbourne": [
        {"key": "p2_mask_au", "keyword": "P2 mask", "role": "protection"},
    ],
}
