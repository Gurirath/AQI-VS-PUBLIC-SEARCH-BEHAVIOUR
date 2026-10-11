# Air Quality vs. Public Search Behaviour

Does PM2.5 pollution show up in what people search for, and when does the public
stop noticing? Weekly PM2.5 (OpenAQ) against Google Trends search interest for
Bangkok, Kraków, Melbourne and Lahore, 2022–2024.

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

The seven notebooks run in order, top to bottom. They contain code and saved outputs only;
the interpretation is in `outputs/report/findings.md`.

| Notebook | What it does |
|---|---|
| `01_data_collection.ipynb` | Calls the collectors (skipping files that exist), counts sensors and search series per city |
| `02_cleaning_merge.ipynb` | Runs `aqs.cleaning`, sensor selection, weekly coverage, merged table |
| `03_eda.ipynb` | Data quality, distributions, seasonality, correlation, PCA, clustering |
| `04_statistics.ipynb` | Correlations with the season removed for every dense series, peak lag per term |
| `05_awareness_gap.ipynb` | When attention responds, how fast, lag asymmetry, whether it wears out, silent weeks |
| `06_causal_ml_verdict.ipynb` | Granger tests, forecast skill for attention and health, scorecard and verdict tables |
| `07_health_signal.ipynb` | Health searches against PM2.5: the seasonality confound, role contrast, cascade, year by year |

`PYTHONPATH=src python -m aqs.results` rebuilds the three result tables without the notebooks.

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
| `src/aqs/` | `config`, collectors, `cleaning`, `features`, `stats`, `awareness`, `causal`, `health`, `results`, `plots` |
| `notebooks/` | analysis notebooks |
| `app/` | Streamlit dashboard |
| `data/interim/` | weekly PM2.5, weekly search series, sensor selection log |
| `data/processed/` | merged tables, `results_summary.csv`, `awareness_scorecard.csv`, `verdicts.csv` |
| `outputs/figures/` | saved figures, prefixed `eda_`, `stats_`, `gap_`, `causal_`, `health_` by notebook |
| `outputs/report/findings.md` | interpretation of every notebook, with limitations |

## Data conventions

- Weeks run Sunday–Saturday and are labelled by their Sunday, matching Google Trends.
- PM2.5 uses reference-grade monitors present for at least half the study window;
  `data/interim/sensor_selection.csv` lists every candidate sensor and why it was kept or dropped.
- Search interest is pulled for the city's own region (Bangkok `TH-10`, Małopolskie `PL-MA`,
  Victoria `AU-VIC`, Punjab `PK-PB`) in both English and the local language. Health terms are pulled at
  country level, as are Lahore's English baseline terms.
