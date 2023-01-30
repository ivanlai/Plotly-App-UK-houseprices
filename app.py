import logging
import random
import sys
import time
import warnings

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from flask_caching import Cache

from config import config as cfg
from figures_utils import (
    get_average_price_by_year,
    get_figure,
    price_ts,
    price_volume_ts,
)
from utils import (
    get_price_volume_df,
    get_regional_data,
    get_regional_geo_data,
    get_regional_geo_sector,
    get_schools_data,
)

warnings.filterwarnings("ignore")


logging.basicConfig(format=cfg["logging format"], level=logging.INFO)
logging.info(f"System: {sys.version}")


""" ----------------------------------------------------------------------------
 App Settings
---------------------------------------------------------------------------- """
regions = [
    "Greater London",
    "South East",
    "South West",
    "Midlands",
    "North England",
    "Wales",
]

colors = {"background": "#1F2630", "text": "#7FDBFF"}

NOTES = """
    **Notes:**
    1. Property type "Other" is filtered from the house price data.
    2. School ranking (2018-2019) is the best of GCSE and A-Level rankings.
    3. GCSE ranking can be misleading - subjects like
    Classics and Latin are excluded from scoring,
    unfairly penalising some schools.

    **Other data sources:**
    - [OpenStreetMap](https://www.openstreetmap.org)
    - [Postcode regions mapping](https://www.whichlist2.com/knowledgebase/uk-postcode-map/)
    - [Postcode boundary data](https://www.opendoorlogistics.com/data/)
    from [www.opendoorlogistics.com](https://www.opendoorlogistics.com)
    - Contains Royal Mail data © Royal Mail copyright and database right 2015
    - Contains National Statistics data © Crown copyright and database right 2015
    - [School 2019 performance data](https://www.gov.uk/school-performance-tables)
    (Ranking scores: [Attainment 8 Score](https://www.locrating.com/Blog/attainment-8-and-progress-8-explained.aspx)
    for GCSE and
    [Average Point Score](https://dera.ioe.ac.uk/26476/3/16_to_18_calculating_the_average_point_scores_2015.pdf)
    for A-Level)
"""

t0 = time.time()

""" ----------------------------------------------------------------------------
Data Pre-processing
---------------------------------------------------------------------------- """
price_volume_df = get_price_volume_df()
regional_price_data = get_regional_data("sector_price")
regional_percentage_delta_data = get_regional_data("sector_percentage_delta")
regional_geo_data, regional_geo_data_paths = get_regional_geo_data()
regional_geo_sector = get_regional_geo_sector(regional_geo_data)
schools_top_500 = get_schools_data()

# ---------------------------------------------

# initial values:
initial_year = max(cfg["Years"])
initial_region = "Greater London"

sectors = regional_price_data[initial_year][initial_region]["Sector"].values
initial_sector = random.choice(sectors)
initial_geo_sector = [regional_geo_sector[initial_region][initial_sector]]

empty_series = pd.DataFrame(np.full(len(cfg["Years"]), np.nan), index=cfg["Years"])
empty_series.rename(columns={0: ""}, inplace=True)


""" ----------------------------------------------------------------------------
 Dash App
---------------------------------------------------------------------------- """
# Select theme from: https://www.bootstrapcdn.com/bootswatch/

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    # external_stylesheets=[dbc.themes.DARKLY]
    external_stylesheets=[dbc.themes.SUPERHERO],
)

server = app.server  # Needed for gunicorn
cache = Cache(
    server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": cfg["cache dir"],
        "CACHE_THRESHOLD": cfg["cache threshold"],
    },
)
app.config.suppress_callback_exceptions = True

# --------------------------------------------------------#

app.layout = html.Div(
    id="root",
    children=[
        # Header -------------------------------------------------#
        html.Div(
            id="header",
            children=[
                html.Div(
                    [
                        html.Div(
                            [html.H1(children="England and Wales House Prices")],
                            style={
                                "display": "inline-block",
                                "width": "74%",
                                "padding": "10px 0px 0px 20px",  # top, right, bottom, left
                            },
                        ),
                        html.Div(
                            [html.H6(children="Created with")],
                            style={
                                "display": "inline-block",
                                "width": "10%",
                                "textAlign": "right",
                                "padding": "0px 20px 0px 0px",  # top, right, bottom, left
                            },
                        ),
                        html.Div(
                            [
                                html.A(
                                    [
                                        html.Img(
                                            src=app.get_asset_url("dash-logo.png"),
                                            style={"height": "100%", "width": "100%"},
                                        )
                                    ],
                                    href="https://plotly.com/",
                                    target="_blank",
                                )
                            ],
                            style={
                                "display": "inline-block",
                                "width": "14%",
                                "textAlign": "right",
                                "padding": "0px 10px 0px 0px",
                            },
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            [
                dcc.Link(
                    f"HM Land Registry Price Paid Data from 01 Jan 1995 to {cfg['latest date']}",  # noqa: E501
                    href="https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads",  # noqa: E501
                    target="_blank",
                    # style={'color': colors['text']}
                )
            ],
            style={"padding": "5px 0px 5px 20px"},
        ),
        # Selection control -------------------------------------#
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id="region",
                            options=[{"label": r, "value": r} for r in regions],
                            value=initial_region,
                            clearable=False,
                            style={"color": "black"},
                        )
                    ],
                    style={
                        "display": "inline-block",
                        "padding": "0px 5px 10px 15px",
                        "width": "15%",
                    },
                    className="one columns",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="year",
                            options=[{"label": y, "value": y} for y in cfg["Years"]],
                            value=initial_year,
                            clearable=False,
                            style={"color": "black"},
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "padding": "0px 5px 10px 0px",
                        "width": "10%",
                    },
                    className="one columns",
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="postcode",
                            options=[
                                {"label": s, "value": s}
                                for s in regional_price_data[initial_year][
                                    initial_region
                                ].Sector.values
                            ],
                            value=[initial_sector],
                            clearable=True,
                            multi=True,
                            style={"color": "black"},
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "padding": "0px 5px 10px 0px",
                        "width": "40%",
                    },
                    className="seven columns",
                ),
                html.Div(
                    [
                        dbc.RadioItems(
                            id="graph-type",
                            options=[
                                {"label": i, "value": i}
                                for i in ["Price", "Volume", "Yr-to-Yr price ±%"]
                            ],
                            value="Price",
                            inline=True,
                        )
                    ],
                    style={
                        "display": "inline-block",
                        "textAlign": "center",
                        "padding": "5px 0px 10px 10px",
                        "width": "33%",
                    },
                    className="two columns",
                ),
            ],
            style={"padding": "5px 0px 10px 20px"},
            className="row",
        ),
        # App Container ------------------------------------------#
        html.Div(
            id="app-container",
            children=[
                # Left Column ------------------------------------#
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="choropleth-container",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H5(id="choropleth-title"),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "width": "64%",
                                            },
                                            className="eight columns",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="school-checklist",
                                                    options=[
                                                        {
                                                            "label": "Show Top 500 Schools",  # noqa: E501
                                                            "value": "True",
                                                        },
                                                    ],
                                                    value=[],
                                                    labelStyle={
                                                        "display": "inline-block"
                                                    },
                                                    inputStyle={"margin-left": "10px"},
                                                )
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "textAlign": "right",
                                                "width": "34%",
                                            },
                                            className="four columns",
                                        ),
                                    ]
                                ),
                                dcc.Graph(id="choropleth"),
                            ],
                        ),
                    ],
                    style={
                        "display": "inline-block",
                        "padding": "20px 10px 10px 40px",
                        "width": "59%",
                    },
                    className="seven columns",
                ),
                # Right Column ------------------------------------#
                html.Div(
                    id="graph-container",
                    children=[
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="property-type-checklist",
                                    options=[
                                        {"label": "F: Flats/Maisonettes", "value": "F"},
                                        {"label": "T: Terraced", "value": "T"},
                                        {"label": "S: Semi-Detached", "value": "S"},
                                        {"label": "D: Detached", "value": "D"},
                                    ],
                                    value=["F", "T", "S", "D"],
                                    labelStyle={"display": "inline-block"},
                                    inputStyle={"margin-left": "10px"},
                                ),
                            ],
                            style={"textAlign": "right"},
                        ),
                        html.Div([dcc.Graph(id="price-time-series")]),
                    ],
                    style={
                        "display": "inline-block",
                        "padding": "20px 20px 10px 10px",
                        "width": "39%",
                    },
                    className="five columns",
                ),
            ],
            className="row",
        ),
        # Notes and credits --------------------------#
        html.Div(
            [
                html.Div(
                    [dcc.Markdown(NOTES)],
                    style={
                        "textAlign": "left",
                        "padding": "0px 0px 5px 40px",
                        "width": "69%",
                    },
                    className="nine columns",
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "© 2020 Ivan Lai "
                            + "[[Blog]](https://www.ivanlai.project-ds.net/) "
                            + "[[Email]](mailto:ivanlai.uk.2020@gmail.com)"
                        )
                    ],
                    style={
                        "textAlign": "right",
                        "padding": "10px 20px 0px 0px",
                        "width": "29%",
                    },
                    className="three columns",
                ),
            ],
            className="row",
        ),
    ],
)

""" ----------------------------------------------------------------------------
 Callback functions:
 Overview:
 region, year, graph-type, school -> choropleth-title
 region, year -> postcode options
 region, year, graph-type, postcode-value, school -> choropleth
 postcode-value, property-type-checklist -> price-time-series
 choropleth-clickData, choropleth-selectedData, region, postcode-State -> postcode-value
---------------------------------------------------------------------------- """

# Update choropleth-title with year and graph-type update
@app.callback(
    Output("choropleth-title", "children"),
    [
        Input("region", "value"),
        Input("year", "value"),
        Input("graph-type", "value"),
        Input("school-checklist", "value"),
    ],
)
def update_map_title(region, year, gtype, school):
    if len(school) > 0:
        return "Top 500 schools (Postcode selection disabled)"
    elif gtype == "Price":
        return f"Avg house price (all property types) by postcode sector in {region}, {year}"  # noqa: E501
    elif gtype == "Volume":
        return (
            f"Sales Volume (all property types) by postcode sector in {region}, {year}"
        )
    else:
        if year == 1995:
            return f"Data from {year-1} to {year} not available"
        else:
            return (
                f"Yr-to-yr average price % change in {region}, from {year-1} to {year}"
            )


# Update postcode dropdown options with region selection
@app.callback(
    Output("postcode", "options"), [Input("region", "value"), Input("year", "value")]
)
def update_region_postcode(region, year):
    return [
        {"label": s, "value": s}
        for s in regional_price_data[year][region].Sector.values
    ]


# Update choropleth-graph with year, region, graph-type update & sectors
@app.callback(
    Output("choropleth", "figure"),
    [
        Input("year", "value"),
        Input("region", "value"),
        Input("graph-type", "value"),
        Input("postcode", "value"),
        Input("school-checklist", "value"),
    ],
)  # @cache.memoize(timeout=cfg['timeout'])
def update_Choropleth(year, region, gtype, sectors, school):
    # Graph type selection------------------------------#
    if gtype in ["Price", "Volume"]:
        df = regional_price_data[year][region]
    else:
        df = regional_percentage_delta_data[year][region]

    # For high-lighting mechanism ----------------------#
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    geo_sectors = dict()

    if "region" not in changed_id:
        for k in regional_geo_data[region].keys():
            if k != "features":
                geo_sectors[k] = regional_geo_data[region][k]
            else:
                geo_sectors[k] = [
                    regional_geo_sector[region][sector]
                    for sector in sectors
                    if sector in regional_geo_sector[region]
                ]

    # Updating figure ----------------------------------#
    fig = get_figure(
        df,
        app.get_asset_url(regional_geo_data_paths[region]),
        region,
        gtype,
        year,
        geo_sectors,
        school,
        schools_top_500,
    )

    return fig


# Update price-time-series with postcode updates and graph-type
@app.callback(
    Output("price-time-series", "figure"),
    [Input("postcode", "value"), Input("property-type-checklist", "value")],
)
@cache.memoize(timeout=cfg["timeout"])
def update_price_timeseries(sectors, ptypes):

    if len(sectors) == 0:
        return price_ts(empty_series, "Please select postcodes", colors)

    if len(ptypes) == 0:
        return price_ts(
            empty_series, "Please select at least one property type", colors
        )

    # --------------------------------------------------#
    df = price_volume_df.iloc[
        np.isin(price_volume_df.index.get_level_values("Property Type"), ptypes),
        np.isin(price_volume_df.columns.get_level_values("Sector"), sectors),
    ]
    df.reset_index(inplace=True)
    avg_price_df = get_average_price_by_year(df, sectors)

    if len(sectors) == 1:
        index = [(a, b) for (a, b) in df.columns if a != "Average Price"]
        volume_df = df[index]
        volume_df.columns = volume_df.columns.get_level_values(0)
        return price_volume_ts(avg_price_df, volume_df, sectors, colors)
    else:
        title = f"Average prices for {len(sectors)} sectors"
        return price_ts(avg_price_df, title, colors)


# ----------------------------------------------------#

# Update postcode dropdown values with clickData, selectedData and region
@app.callback(
    Output("postcode", "value"),
    [
        Input("choropleth", "clickData"),
        Input("choropleth", "selectedData"),
        Input("region", "value"),
        Input("school-checklist", "value"),
        State("postcode", "value"),
        State("choropleth", "clickData"),
    ],
)
def update_postcode_dropdown(
    clickData, selectedData, region, school, postcodes, clickData_state
):

    # Logic for initialisation or when Schoold sre selected
    if dash.callback_context.triggered[0]["value"] is None:
        return postcodes

    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]

    if len(school) > 0 or "school" in changed_id:
        clickData_state = None
        return []

    # --------------------------------------------#

    if "region" in changed_id:
        postcodes = []
    elif "selectedData" in changed_id:
        postcodes = [D["location"] for D in selectedData["points"][: cfg["topN"]]]
    elif clickData is not None and "location" in clickData["points"][0]:
        sector = clickData["points"][0]["location"]
        if sector in postcodes:
            postcodes.remove(sector)
        elif len(postcodes) < cfg["topN"]:
            postcodes.append(sector)
    return postcodes


# ----------------------------------------------------#

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

logging.info(f"Data Preparation completed in {time.time()-t0 :.1f} seconds")

# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    logging.info(sys.version)

    # If running locally in Anaconda env:
    if "conda-forge" in sys.version:
        app.run_server(debug=True)

    # If running on AWS/Pythonanywhere production
    else:
        app.run_server(port=8050, host="0.0.0.0")

""" ----------------------------------------------------------------------------
Terminal cmd to run:
gunicorn app:server -b 0.0.0.0:8050
---------------------------------------------------------------------------- """
