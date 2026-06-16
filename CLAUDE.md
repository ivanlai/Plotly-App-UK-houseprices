# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Plotly Dash web app that visualizes England & Wales house prices and sales volume by postcode sector (1995–2025), with a choropleth map, time-series charts, and school ranking overlays. Deployed on PythonAnywhere via WSGI/gunicorn.

## Running Locally

```bash
# Install dependencies (uses uv)
uv sync

# Run dev server
uv run python app.py

# Production-style
uv run gunicorn app:server -b 0.0.0.0:8050
```

## Formatting

```bash
uv run ruff format .
uv run ruff check .
```

Ruff config is in `pyproject.toml`: line-length 88, tab indentation, double quotes.

## Architecture

**Single-page Dash app**, no routing:

- `app.py` — App setup, cache init, wires layout and callbacks. Exports `server` (Flask) for gunicorn. Data loaded at module level on startup.
- `layout.py` — Dash layout (dropdowns, map, charts, checklists).
- `callbacks.py` — All Dash callback functions.
- `data.py` — Data loading orchestrator: calls `utils.py` helpers, returns a single dict of dataframes and geo data.
- `utils.py` — Data loading helpers: reads CSVs from `appData/` and GeoJSON from `assets/`. Returns dataframes and geo dicts keyed by year and region.
- `config.py` — Central config dict (`cfg`): year range, region-to-map-region lookup, plotly map centre/zoom settings, cache settings. Detects PythonAnywhere vs local paths.
- `figures_utils.py` — Plotly figure builders: choropleth map (`get_figure`), time-series line charts (`price_ts`, `price_volume_ts`), school scatter overlay (`get_scattergeo`).

**Callback flow** (defined in `callbacks.py`):
- Region/year dropdowns → update postcode dropdown options and choropleth map
- Postcode selection + property type checklist → update price time-series chart
- Map click/select → update postcode dropdown values
- School checklist → overlays school scatter plot, disables postcode selection

**Data files** (not generated at runtime):
- `appData/sector_price_{year}.csv` and `sector_percentage_delta_{year}.csv` — one file per year, regional price/volume data
- `appData/price_volume.csv` — full price/volume matrix used for time-series
- `assets/geodata_{region}.json` — GeoJSON boundary files per region
- `appData/schools_top_500.csv` — school rankings

**Caching**: Flask-Caching filesystem cache (`/tmp/cache` locally) memoizes the price time-series callback. Threshold configured in `config.py`.

## Deployment

PythonAnywhere: `wsgi.py` adds the project to `sys.path` and imports `app.server` as `application`. Paths in `config.py` switch between PythonAnywhere (`/home/ivanlai/...`) and local (`appData/`, `assets/`).

## Key Conventions

- Region names in config (`"North England"`, `"Greater London"`, etc.) are used as keys throughout data dicts and filenames — they must stay consistent.
- "South East" choropleth includes Greater London data (special case in `utils.py:get_regional_data`).
- Update `config["end_year"]` and `config["latest date"]` when adding new year data.
