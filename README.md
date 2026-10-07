# Air Quality vs. Public Search Behaviour

Does PM2.5 pollution show up in what people search for, and when does the public
stop noticing? Weekly PM2.5 (OpenAQ) against Google Trends search interest for
Bangkok, Kraków and Melbourne, 2022–2024.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put an OpenAQ API key in `.env` (not tracked):

```
OPENAQ_API_KEY="..."
```

## Rebuilding the data

Run from the repo root with `PYTHONPATH=src`.

```bash
PYTHONPATH=src .venv/bin/python -m aqs.collect_openaq   # daily PM2.5, every sensor within 25 km
PYTHONPATH=src .venv/bin/python -m aqs.collect_trends   # weekly search interest (drives Chrome; slow)
PYTHONPATH=src .venv/bin/python -m aqs.cleaning         # filter, align weeks, merge
```

Both collectors skip files that already exist, so they can be re-run after a
failure. Raw pulls land in `data/raw/` (not tracked); `aqs.cleaning` writes the
tracked tables in `data/interim/` and `data/processed/`.

## Layout

| Path | Contents |
|---|---|
| `src/aqs/` | collection, cleaning and shared configuration |
| `notebooks/` | analysis notebooks |
| `data/interim/` | weekly PM2.5, weekly search series, sensor selection log |
| `data/processed/` | merged analysis tables |
| `outputs/figures/` | saved figures |

## Data conventions

- Weeks run Sunday–Saturday and are labelled by their Sunday, matching Google Trends.
- PM2.5 uses reference-grade monitors present for at least half the study window;
  `data/interim/sensor_selection.csv` lists every candidate sensor and why it was kept or dropped.
- Search interest is pulled for the city's own region (Bangkok `TH-10`, Małopolskie `PL-MA`,
  Victoria `AU-VIC`) in both English and the local language.
