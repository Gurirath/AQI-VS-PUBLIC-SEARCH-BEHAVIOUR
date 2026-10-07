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
        {"key": "pm25_th", "label": "PM2.5 (Thai usage)", "keyword": "PM2.5", "role": "awareness"},
        {"key": "dust_th", "label": "dust (Thai)", "keyword": "ฝุ่น", "role": "awareness"},
        {"key": "air_quality_th", "label": "air quality (Thai)", "keyword": "คุณภาพอากาศ", "role": "awareness"},
        {"key": "air_purifier_th", "label": "air purifier (Thai)", "keyword": "เครื่องฟอกอากาศ", "role": "protection"},
        {"key": "mask_th", "label": "N95 mask (Thai)", "keyword": "หน้ากาก N95", "role": "protection"},
    ],
    "krakow": [
        {"key": "smog_pl", "label": "smog (Polish)", "keyword": "smog", "role": "awareness"},
        {"key": "air_quality_pl", "label": "air quality (Polish)", "keyword": "jakość powietrza", "role": "awareness"},
        {"key": "air_purifier_pl", "label": "air purifier (Polish)", "keyword": "oczyszczacz powietrza", "role": "protection"},
        {"key": "mask_pl", "label": "anti-smog mask (Polish)", "keyword": "maska antysmogowa", "role": "protection"},
    ],
    "melbourne": [
        {"key": "p2_mask_au", "label": "P2 mask", "keyword": "P2 mask", "role": "protection"},
    ],
}

# One awareness and one protection series per city for headline figures: the
# local-language term with the fewest zero weeks in each role (chosen on data
# density, before looking at how well it tracks PM2.5).
HEADLINE_TERMS = {
    "bangkok": {"awareness": "dust_th", "protection": "air_purifier_th"},
    "krakow": {"awareness": "smog_pl", "protection": "air_purifier_pl"},
    "melbourne": {"awareness": "air_quality", "protection": "air_purifier"},
}

# A series with more zero weeks than this is too sparse to treat as continuous.
MAX_ZERO_SHARE = 0.5

WHO_24H_GUIDELINE = 15  # µg/m³, WHO 2021 24-hour PM2.5 guideline
