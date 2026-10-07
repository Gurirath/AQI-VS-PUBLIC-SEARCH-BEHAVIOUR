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

## Running the analysis

Notebooks 05–08 are the current analysis. Each runs top to bottom from the tracked data.

| Notebook | What it does |
|---|---|
| `05_eda.ipynb` | Data quality, distributions, seasonality, correlation, PCA, clustering |
| `06_statistics.ipynb` | Correlations with the season removed, bootstrap intervals, corrected p-values |
| `07_awareness_gap.ipynb` | When attention responds, how fast, whether it wears out, and silent pollution weeks |
| `08_causal_and_verdict.ipynb` | Granger tests in both directions, forecast skill, per-city verdict |

Notebooks 01–04 are the first iteration and are kept for reference. They used
misaligned weeks and are superseded by `src/aqs` and notebooks 05–08; 02 no
longer runs against the current raw files.

## Dashboard

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Four pages: overview and verdicts, an interactive city explorer (choose a search
series, remove the season, shift the lag), the Awareness Gap, and data and
methods. It reads only the tracked tables, so it needs no API key.

## Layout

| Path | Contents |
|---|---|
| `src/aqs/` | `config`, collectors, `cleaning`, `features`, `stats`, `awareness`, `causal`, `plots` |
| `notebooks/` | analysis notebooks |
| `app/` | Streamlit dashboard |
| `data/interim/` | weekly PM2.5, weekly search series, sensor selection log |
| `data/processed/` | merged tables, `results_summary.csv`, `awareness_scorecard.csv`, `verdicts.csv` |
| `outputs/figures/` | saved figures (`eda_`, `stats_`, `gap_`, `causal_` prefixes are current) |

## Data conventions

- Weeks run Sunday–Saturday and are labelled by their Sunday, matching Google Trends.
- PM2.5 uses reference-grade monitors present for at least half the study window;
  `data/interim/sensor_selection.csv` lists every candidate sensor and why it was kept or dropped.
- Search interest is pulled for the city's own region (Bangkok `TH-10`, Małopolskie `PL-MA`,
  Victoria `AU-VIC`) in both English and the local language.
