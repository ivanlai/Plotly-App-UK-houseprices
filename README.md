# Plotly App: UK House Prices

Visualise average England and Wales house prices and sales volume by postcode sector (1995–2025), with top 500 school rankings overlay.

**Live app:** [https://ukhouseprice.project-ds.net/](https://ukhouseprice.project-ds.net/)

![Screenshot](https://github.com/ivanlai/apps-UK_houseprice/blob/master/images/Screenshot-plotly-app.png)

## Running locally

```bash
uv sync
uv run python app.py
# or production-style:
uv run gunicorn app:server -b 0.0.0.0:8050
```

## Annual data update

All settings live in `config.py`. For a new year, edit these values:

```python
# config.py
"end_year": 2026,
"latest date": "31 December 2026",

# pipeline section:
"raw_price_files": ["pp-2026.csv"],
"years_to_process": [2026],
```

Then:

1. Download raw price-paid CSV from [Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) into `input/HousePriceData/Raw/`
2. Run the preprocessing script:
   ```bash
   uv run python scripts/preprocess.py
   ```
   This produces all `appData/` CSVs and summary HTML plots in `output/<year>/`.
3. Verify locally: `uv run python app.py`
4. Deploy to PythonAnywhere.

## Project structure

```
app.py              Dash app layout and callbacks
config.py           Central config (shared by app and pipeline)
figures_utils.py    Plotly figure builders
utils.py            Data loading for the app
scripts/
  preprocess.py     Data preprocessing pipeline (replaces notebook)
appData/            Generated CSVs read by the app
assets/             GeoJSON boundary files
output/             HTML summary plots from preprocessing
input/              Raw data (not in git)
notebooks/          Legacy preprocessing notebook (reference only)
```

## Deployment on PythonAnywhere

In PythonAnywhere bash console:

```bash
mkvirtualenv py38 --python=/usr/bin/python3.8
pip install -r requirements.txt
```

In the "Web" tab:

- Update the WSGI file in the Code section (match `wsgi.py` in repo).
- Set the virtualenv path in the Virtualenv section.
- Inspect log files for debugging.
