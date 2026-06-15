import logging
import sys
import warnings

import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

from config import config as cfg
from data import load_all_data
from layout import create_layout
from callbacks import register_callbacks

warnings.filterwarnings("ignore")

logging.basicConfig(format=cfg["logging_format"], level=logging.INFO)
logging.info(f"System: {sys.version}")

data = load_all_data()

app = dash.Dash(
	__name__,
	meta_tags=[
		{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
	],
	external_stylesheets=[dbc.themes.SUPERHERO],
)

server = app.server
cache = Cache(
	server,
	config={
		"CACHE_TYPE": "filesystem",
		"CACHE_DIR": cfg["cache_dir"],
		"CACHE_THRESHOLD": cfg["cache_threshold"],
	},
)
app.config.suppress_callback_exceptions = True

app.layout = create_layout(app, data)
register_callbacks(app, cache, data)

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == "__main__":
	logging.info(sys.version)

	if "conda-forge" in sys.version:
		app.run_server(debug=True)
	else:
		app.run_server(port=8050, host="0.0.0.0")
