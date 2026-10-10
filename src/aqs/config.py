from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_AQ = ROOT / "data" / "raw" / "openaq"
RAW_TRENDS = ROOT / "data" / "raw" / "trends"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"

DATE_FROM = "2022-01-01"
DATE_TO = "2024-12-31"

RADIUS_M = 25_000
PM25_PARAMETER_ID = 2

CITIES = {
    "bangkok": {"lat": 13.7563, "lon": 100.5018, "geo_country": "TH", "geo_region": "TH-10"},
    "krakow": {"lat": 50.0647, "lon": 19.9450, "geo_country": "PL", "geo_region": "PL-MA"},
    "melbourne": {"lat": -37.8136, "lon": 144.9631, "geo_country": "AU", "geo_region": "AU-VIC"},
    "lahore": {"lat": 31.5204, "lon": 74.3587, "geo_country": "PK", "geo_region": "PK-PB"},
}

TRENDS_TIMEFRAME = f"{DATE_FROM} {DATE_TO}"

BASELINE_TERMS = {"air_quality": "air quality", "air_purifier": "air purifier", "N95_mask": "N95 mask"}

BASELINE_GEO_LEVELS = {
    "bangkok": ("region", "country"),
    "krakow": ("region", "country"),
    "melbourne": ("region", "country"),
    "lahore": ("country",),
}

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
    "lahore": [
        {"key": "smog_pk", "label": "smog", "keyword": "smog", "role": "awareness"},
        {"key": "smog_ur", "label": "smog (Urdu)", "keyword": "سموگ", "role": "awareness"},
        {"key": "mask_pk", "label": "mask", "keyword": "mask", "role": "protection"},
    ],
}

HEALTH_TERMS = [
    {"key": "hospital", "keyword": "hospital",
     "local": {"bangkok": "โรงพยาบาล", "krakow": "szpital"}},
    {"key": "breathing_difficulty", "keyword": "breathing difficulty",
     "local": {"bangkok": "หายใจลำบาก", "krakow": "duszność"}, "no_english": ("bangkok", "krakow")},
    {"key": "cough_medicine", "keyword": "cough medicine",
     "local": {"bangkok": "ยาแก้ไอ", "krakow": "syrop na kaszel"}},
    {"key": "eye_drops", "keyword": "eye drops",
     "local": {"bangkok": "ยาหยอดตา", "krakow": "krople do oczu"}},
    {"key": "chest_pain", "keyword": "chest pain",
     "local": {"bangkok": "เจ็บหน้าอก", "krakow": "ból w klatce piersiowej"}},
]

HEALTH_LOCAL_LANGUAGE = {"bangkok": ("th", "Thai"), "krakow": ("pl", "Polish")}

HEALTH_GEO_LEVEL = "country"


def health_terms(city):
    terms = []
    for t in HEALTH_TERMS:
        if city not in t.get("no_english", ()):
            terms.append({"key": t["key"], "label": f"{t['keyword']} (English, {HEALTH_GEO_LEVEL})",
                          "keyword": t["keyword"], "role": "health"})
        if city in t["local"]:
            suffix, language = HEALTH_LOCAL_LANGUAGE[city]
            terms.append({"key": f"{t['key']}_{suffix}", "label": f"{t['keyword']} ({language})",
                          "keyword": t["local"][city], "role": "health"})
    return terms


def default_geo_level(city, term):
    if term in BASELINE_TERMS:
        return BASELINE_GEO_LEVELS[city][0]
    if any(t["key"] == term for t in health_terms(city)):
        return HEALTH_GEO_LEVEL
    return "region"


HEADLINE_TERMS = {
    "bangkok": {"awareness": "dust_th", "protection": "air_purifier_th"},
    "krakow": {"awareness": "smog_pl", "protection": "air_purifier_pl"},
    "melbourne": {"awareness": "air_quality", "protection": "air_purifier"},
    "lahore": {"awareness": "air_quality", "protection": "air_purifier"},
}

MAX_ZERO_SHARE = 0.5

WHO_24H_GUIDELINE = 15
