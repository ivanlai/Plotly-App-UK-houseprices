import os
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objs import Scatter, Figure, Layout

import numpy as np
import pandas as pd
import geopandas as gpd

from copy import copy, deepcopy
from collections import OrderedDict

import gdal
import json
from json import dumps

import pickle
import gc

from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
""" Configurations
"""
cfg = dict()

cfg['start_year']       = 1995
cfg['end_year']         = 2020
cfg['Years']            = list(range(cfg['start_year'], cfg['end_year']+1))

cfg['figure years']     = [2000, 2010, 2015, 2018, 2019, 2020]

cfg['in_dir']           = 'input'
cfg['data_dir']         = 'data'
cfg['figures_out_dir']  = os.path.join(cfg['data_dir'], 'plotly_figures')

cfg['load figures']     = False

cfg['regions_lookup'] = {
        'North East'      : 'North England',
        'North West'      : 'North England',
        'East Midlands'   : 'Midlands',
        'West Midlands'   : 'Midlands',
        'Greater London'  : 'Greater London',
        'South East'      : 'South East',
        'South West'      : 'South West',
        'Wales'           : 'Wales',
        'Scotland'        : 'Scotland',
        'Northern Ireland': 'Northern Ireland'
}

cfg['plotly_config'] = {
         'All':            {'centre': [53.2, -2.2], 'maxp': 95, 'zoom': 6},
         'North England':  {'centre': [54.3, -2.0], 'maxp': 99, 'zoom': 7},
         'Wales':          {'centre': [52.4, -3.3], 'maxp': 99, 'zoom': 7.3},
         'Midlands':       {'centre': [52.8, -1.2], 'maxp': 99, 'zoom': 7.3},
         'South West':     {'centre': [51.1, -3.7], 'maxp': 99, 'zoom': 7.2},
         'South East':     {'centre': [51.5, -0.1], 'maxp': 90, 'zoom': 7.8},
         'Greater London': {'centre': [51.5, -0.1], 'maxp': 80, 'zoom': 9.5},
}
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

""" ------------------------------------------
 House Price Data
------------------------------------------ """
price_df  = pd.read_csv(os.path.join(cfg['data_dir'], 'price_ts.csv'), index_col='Year')
volume_df = pd.read_csv(os.path.join(cfg['data_dir'], 'volume_ts.csv'), index_col='Year')

type_df = pd.read_csv(os.path.join(cfg['data_dir'], 'property_type.csv'))
type_df = type_df.set_index(['Year', 'Property Type', 'Sector']).unstack(level=-1)
type_df.columns = type_df.columns.get_level_values(1)
type_df.fillna(value=0, inplace=True)

sector_df = pd.read_csv(os.path.join(cfg['data_dir'], 'sector_houseprice.csv'))
sector_by_year = dict()
for year in cfg['Years']:
    sector_by_year[year] = sector_df[sector_df.Year==year].reset_index(drop=True)

#-------------------------------------------------------#

def get_regional_price_data(sector_df, regions):

    regional_price_data = dict()
    regional_price_data['All'] = deepcopy(sector_df)

    for region in regions:
        if region == 'South East': #Include Greater London in South East graph
            mask = (sector_df.Region==region) | (sector_df.Region=='Greater London')
        else:
            mask = (sector_df.Region==region)

        regional_price_data[region] = sector_df[mask]

    return regional_price_data

#-------------------------------------------------------#

# Breaking price/volume data up by region:
regions = [r for r in sector_df.Region.unique() if isinstance(r, str)]

regional_price_data = dict()
for year in cfg['Years']:
    regional_price_data[year] = get_regional_price_data(sector_by_year[year], regions)

#-------------------------------------------------------#

""" ------------------------------------------
 Post Code Data
------------------------------------------ """

postcode_region_df = pd.read_csv(os.path.join(cfg['in_dir'], 'PostCode Region.csv'))

postcode_region = dict()
for (prefix, region) in postcode_region_df[['Prefix', 'Region']].values:
    postcode_region[prefix] = cfg['regions_lookup'][region]


""" ------------------------------------------
 Geo Data
------------------------------------------ """

def load_geo_data(infile):
    with open(infile, "r") as read_file:
        geo_data = json.load(read_file)
    return geo_data

#---------------------------------------------#

infile = os.path.join(cfg['in_dir'], 'ukpostcode_geojson.json')
geo_data = load_geo_data(infile)

#---------------------------------------------#

def get_regional_geo_data(geo_data, postcode_region, regions):

    pattern = re.compile(r"\d")
    #......................................
    def inner(region):
        Y = dict()
        Y['features'] = []
        for k in geo_data.keys():
            if k != 'features':
                Y[k] = geo_data[k]
            else:
                for i, d in enumerate(geo_data['features']):
                    for k, v in d.items():
                        if k == 'properties':
                            sector = v['name']
                            m = pattern.search(sector)
                            district = sector[:m.start()]

                            if region == 'South East':
                                if postcode_region[district] in [region, 'Greater London']:
                                    Y['features'].append(geo_data['features'][i])
                            else:
                                if postcode_region[district] == region:
                                    Y['features'].append(geo_data['features'][i])
        return Y

    #......................................
    regional_geo_data = dict()
    for r in regions:
        regional_geo_data[r] = inner(r)

    return regional_geo_data

#---------------------------------------------#

regional_geo_data = get_regional_geo_data(geo_data, postcode_region, regions)


""" ------------------------------------------
 Making Graphs
------------------------------------------ """

def get_figure(price_data, geo_data, region):

    _cfg = cfg['plotly_config'][region]
    min_price = np.percentile(np.array(price_data.Price), 5)
    max_price = np.percentile(np.array(price_data.Price), _cfg['maxp'])

    fig = go.Figure(
            go.Choroplethmapbox(
                geojson = geo_data,
                locations = price_data['Sector'],
                featureidkey = "properties.name",
                colorscale = "YlOrRd",
                z = price_data['Price'],
                zmin = min_price,
                zmax = max_price,
                text = price_data['text'], # hover text
                marker_opacity = 0.4,
                marker_line_width = 1,
                colorbar_title = "Average House Price (£)",
          ))

    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_zoom=_cfg['zoom'],
                      autosize=True,
#                       width=1850,
#                       height=800,
                      font=dict(color="#2cfec1"),
                      paper_bgcolor="#1f2630",
                      mapbox_center = {"lat": _cfg['centre'][0] , "lon": _cfg['centre'][1]},
                      margin={'l': 20, 'r': 20, 't': 20, 'b': 20}
                     )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

""" ------------------------------------------
 App Settings
------------------------------------------ """

regions =  ['Greater London',
            'South East',
            'South West',
            'Midlands',
            'North England',
            'Wales']

colors = {
    'background': '#1F2630',
    'text': '#7FDBFF'
}

""" ------------------------------------------
 Dash App
------------------------------------------ """

app = dash.Dash(
     meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
     ],
     external_stylesheets = [dbc.themes.DARKLY]
)

# app = dash.Dash()
#--------------------------------------------------------#

app.layout = html.Div(
    id="root",
    children=[
        # Header -------------------------------------------------#
        html.Div(
            id="header",
            children = [
                html.Div([
                    html.Div([html.H1(children='England and Wales House Prices')],
                              style={'display': 'inline-block',
                                     'width': '69%',
                                     'fontColor': 'orange',
                                     'padding': '10px 0px 0px 20px'}), #padding: top, right, bottom, left

                    html.Div([html.Img(src=app.get_asset_url("dash-logo.png"),
                                       style={'height': '50%',
                                              'width' : '50%'})],
                              style={'display': 'inline-block',
                                     'width': '29%',
                                     'textAlign': 'right',
                                     'padding': '0px 0px 0px 0px'}),
                ]),
                html.Div([
                    html.P(
                        id="description",
                        children="Dash: A web application framework for Python.",
                    )
                ], style={'padding': '0px 0px 0px 25px'})
            ],
        ),

        # App Container ------------------------------------------#
        html.Div(
            id="app-container",
            children=[
                # Left Column ------------------------------------#
                html.Div(
                    id="left-column",
                    children=[
                        html.Div([
                            html.Div(
                                id="dropdown-container",
                                children=[
                                    html.P(
                                        id="dropdown-text",
                                        children="Select region:",
                                    ),
                                    dcc.Dropdown(
                                        id='region',
                                        options=[{'label': r, 'value': r} for r in regions],
                                        value='Greater London',
                                        clearable=False,
                                        style={'color': 'black'}
                                    )
                                ], style={'display': 'inline-block',
                                          'width': '19%'}
                                ),
                            html.Div(
                                id="slider-container",
                                children=[
                                    html.Div([
                                        html.P(
                                            id="slider-text",
                                            children="Drag the slider to change the year:",
                                        )], style={ 'padding': '0px 0px 0px 20px'}
                                    ),
                                    html.Div([
                                        dcc.Slider(
                                            id="years-slider",
                                            min=min(cfg['figure years']),
                                            max=max(cfg['figure years']),
                                            value=max(cfg['figure years']),
                                            marks={
                                                str(year): {
                                                    "label": str(year),
                                                    "style": {"color": "#7fafdf"},
                                                }
                                                for year in cfg['figure years']
                                            },
                                        ),
                                    ]),
                                ], style={'display': 'inline-block', 'width': '79%'}
                            ),
                        ]),

                        html.Div(
                            id="choropleth-container",
                            children=[
                                html.P(
                                    children=f"Average house price by postcode sector in {max(cfg['figure years'])}",
                                    id="choropleth-title",
                                ),
                                dcc.Graph(id="county-choropleth",
                                          clickData={'points': [{'location': 'HA6 3'}]},
                                          figure = get_figure(regional_price_data[2000]['Greater London'],
                                                              regional_geo_data['Greater London'],
                                                              'Greater London')
                                ),
                            ], style={'padding': '20px 0px 0px 0px'},
                        ),
                    ],
                    style={'display': 'inline-block',
                           "width": "64%",
                           'padding': '0px 20px 0px 30px'},
                    className="seven columns"
                ),

                # Right Column ------------------------------------#
                html.Div(
                    id="graph-container",
                    children=[
                        dcc.Graph(id='price-time-series'),
                        dcc.Graph(id='volume-time-series')
                    ],
                    style={'display': 'inline-block',
                           'width': '34%'},
                    className="five columns"
                ),
            ],
          className="row"
        ),
    ],
)

################################################################

@app.callback(
    Output("choropleth-title", "children"),
    [Input("years-slider", "value")])
def update_map_title(year):
    return f"Average house price by postcode sector in {year}"

#----------------------------------------------------#

@app.callback(
    Output("county-choropleth", 'figure'),
    [Input('years-slider', 'value'),
     Input("region", "value")])
def update_graph(year, region):
    return get_figure(regional_price_data[year][region],
                      regional_geo_data[region],
                      region)

#----------------------------------------------------#

def create_time_series(df, title, ylabel):
    fig = px.scatter(df, labels=dict(value=ylabel, variable="PostCode"))
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(#height=225,
                      margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
                      plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'],
                      font_color=colors['text'])
    return fig

#----------------------------------------------------#

@app.callback(
    Output('price-time-series', 'figure'),
    [Input('county-choropleth', 'clickData')])
def update_price_timeseries(clickData):
    graph = None
    count = 0
    while count <= 3:
        try:
            sector = clickData['points'][0]['location']
            title = f'{sector} Average price time-series'
            graph = create_time_series(price_df[sector], title, "Average Price (£)")
            break
        except:
            count += 1
    return graph

#----------------------------------------------------#

@app.callback(
    Output('volume-time-series', 'figure'),
    [Input('county-choropleth', 'clickData')])
def update_price_timeseries(clickData):
    graph = None
    count = 0
    while count <= 3:
        try:
            sector = clickData['points'][0]['location']
            title = f'{sector} Sales volume time-series'
            graph = create_time_series(volume_df[sector], title, "Sales Volume")
            break
        except:
            count += 1
    return graph

#----------------------------------------------------#

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    app.run_server(debug=True)
