import os
import re
import sys
import time
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objs import Scatter, Figure, Layout
from plotly.subplots import make_subplots
# from flask_caching import Cache

import numpy as np
import pandas as pd
import geopandas as gpd

from copy import copy, deepcopy
from collections import OrderedDict

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

cfg['geo_data_dir']     = 'input/geoData'
cfg['app_data_dir']     = 'appData'

cfg['topN']             = 12
cfg['timeout']          = 60*20

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
         # 'All':            {'centre': [53.2, -2.2], 'maxp': 95, 'zoom': 6},
         'North England':  {'centre': [54.3, -2.0], 'maxp': 99, 'zoom': 6.5},
         'Wales':          {'centre': [52.4, -3.3], 'maxp': 99, 'zoom': 6.9},
         'Midlands':       {'centre': [52.8, -1.0], 'maxp': 99, 'zoom': 7},
         'South West':     {'centre': [51.1, -3.7], 'maxp': 99, 'zoom': 6.9},
         'South East':     {'centre': [51.5, -0.1], 'maxp': 90, 'zoom': 7.3},
         'Greater London': {'centre': [51.5, -0.1], 'maxp': 80, 'zoom': 8.9},
}
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
t0 = time.time()

""" ------------------------------------------
 House Price Data
------------------------------------------ """
price_df  = pd.read_csv(os.path.join(cfg['app_data_dir'], 'price_ts.csv'), index_col='Year')
volume_df = pd.read_csv(os.path.join(cfg['app_data_dir'], 'volume_ts.csv'), index_col='Year')

type_df = pd.read_csv(os.path.join(cfg['app_data_dir'], 'property_type.csv'))
type_df = type_df.set_index(['Year', 'Property Type', 'Sector']).unstack(level=-1)
type_df.columns = type_df.columns.get_level_values(1)
type_df.fillna(value=0, inplace=True)

#-------------------------------------------------------#

""" ------------------------------------------------
 Regional Price, percentage delta and Volume Data
-------------------------------------------------"""

def get_regional_data(fname):
    regiona_data = dict()
    for year in cfg['Years']:
        df = pd.read_csv(os.path.join(cfg['app_data_dir'], f'{fname}_{year}.csv'))

        tmp = dict()
        for region in cfg['plotly_config']:
            if region == 'South East': #Include Greater London in South East graph
                mask = (df.Region==region) | (df.Region=='Greater London')
            else:
                mask = (df.Region==region)
            tmp[region] = df[mask]

        regiona_data[year] = deepcopy(tmp)

    return regiona_data

regional_price_data = get_regional_data('sector_price')
regional_percentage_delta_data = get_regional_data('sector_percentage_delta')

""" ------------------------------------------
 Geo Data
------------------------------------------ """
regional_geo_data = dict()
for region in cfg['plotly_config']:
    infile = os.path.join(cfg['app_data_dir'], f'geodata_{region}.csv')
    with open(infile, "r") as read_file:
        regional_geo_data[region] = json.load(read_file)

""" ------------------------------------------
 Making Graphs
------------------------------------------ """

def get_figure(df, geo_data, region, gtype, year):

    _cfg = cfg['plotly_config'][region]

    if gtype == 'Price':
        min_value = np.percentile(np.array(df.Price), 5)
        max_value = np.percentile(np.array(df.Price), _cfg['maxp'])
        z_vec = df['Price']
        text_vec = df['text']
        colorscale = "YlOrRd"
        title = f"{year} Avg Price (£)"
    else:
        min_value = np.percentile(np.array(df['Percentage Change']), 5)
        max_value = np.percentile(np.array(df['Percentage Change']), 95)
        z_vec = df['Percentage Change']
        text_vec = ''
        # colorscale = "Picnic"
        colorscale = "Jet"
        title = "Avg. Price %Change"

    #-------------------------------------------#

    fig = go.Figure(
            go.Choroplethmapbox(
                geojson = geo_data,
                locations = df['Sector'],
                featureidkey = "properties.name",
                colorscale = colorscale,
                z = z_vec,
                zmin = min_value,
                zmax = max_value,
                text = text_vec,
                marker_opacity = 0.4,
                marker_line_width = 1,
                colorbar_title = title,
          ))

    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_zoom=_cfg['zoom'],
                      autosize=True,
                      font=dict(color="#7FDBFF"),
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

sectors = regional_price_data[max(cfg['Years'])]['Greater London']['Sector'].values
initial_sector = random.choice(sectors)

state = dict()
state['last_clickData']    = None
state['last_selectedData'] = None

""" ------------------------------------------
 Dash App
------------------------------------------ """

app = dash.Dash(
    __name__,
     meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
     ],
     external_stylesheets = [dbc.themes.DARKLY]
)
# cache = Cache(app.server, config={'CACHE_TYPE': 'filesystem',
#                                   'CACHE_DIR': 'cache'})
# app.config.suppress_callback_exceptions = True

server = app.server #Needed for gunicorn

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
                    html.Div([
                        html.A([
                            html.Img(src=app.get_asset_url("dash-logo.png"),
                                           style={'height': '50%',
                                                  'width' : '50%'})
                        ], href='https://plotly.com/', target='_blank')
                    ], style={'display': 'inline-block',
                              'width': '29%',
                              'textAlign': 'right',
                              'padding': '0px 0px 0px 0px'}),
                ]),
            ],
        ),

        # Selection control -------------------------------------#
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='region',
                    options=[{'label': r, 'value': r} for r in regions],
                    value='Greater London',
                    clearable=False,
                    style={'color': 'black'}
                )
                ], style={'display': 'inline-block',
                          'padding': '0px 5px 10px 15px',
                          'width': '15%'},
                   className="two columns"
                ),
            html.Div([
                dcc.Dropdown(
                    id="years",
                    options=[{'label': y, 'value': y} for y in cfg['Years']],
                    value=max(cfg['Years']),
                    clearable=False,
                    style={'color': 'black'}
                ),
            ], style={'display': 'inline-block',
                      'padding': '0px 5px 10px 0px',
                      'width': '7%'},
               className="one columns"
            ),
            html.Div([
                dbc.RadioItems(
                    id='graph-type',
                    options=[{'label': i, 'value': i} for i in ['Price', 'Yr-to-Yr ±%']],
                    value='Price',
                    inline=True,
                )
                ], style={'display': 'inline-block',
                          'float': 'right',
                          'padding': '5px 0px 10px 10px',
                          'width': '76%'},
                  className="nine columns"
            ),
        ],  style={'padding': '5px 0px 10px 20px'},
            className="row"
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
                                html.H6(
                                    children=f"Average house price by postcode sector in {max(cfg['Years'])}",
                                    id="choropleth-title",
                                ),
                                dcc.Graph(id="county-choropleth",
                                          clickData={'points': [{'location': initial_sector}]},
                                          figure = get_figure(regional_price_data[max(cfg['Years'])]['Greater London'],
                                                              regional_geo_data['Greater London'],
                                                              'Greater London',
                                                              gtype='Price',year=max(cfg['Years']))
                                ),
                            ],
                        ),
                    ],
                    style={'display': 'inline-block',
                           'padding': '20px 10px 10px 40px',
                           "width": "64%"},
                    className="seven columns"
                ),

                # Right Column ------------------------------------#
                html.Div(
                    id="graph-container",
                    children=[
                        html.H6(
                            children="Legends - F: Flats/Maisonettes; T: Terraced; S: Semi-Detached; D: Detached",
                            id="legend-keys",
                        ),
                        html.Div([dcc.Graph(id='price-time-series')]),
                    ],
                    style={'display': 'inline-block',
                           'padding': '20px 10px 10px 10px',
                           'width': '34%'},
                    className="five columns"
                ),
            ],
            className="row"
        )
    ],
)

################################################################

@app.callback(
    Output("choropleth-title", "children"),
    [Input("years", "value"),
     Input("graph-type", "value")])
def update_map_title(year, gtype):
    if gtype == 'Price':
        return f"Average house price by postcode sector in {year}"
    else:
        if year == 1995:
            return f"Data from {year-1} to {year} not available"
        else:
            return f"Year-to-year price % change by postcode sector, from {year-1} to {year}"

#----------------------------------------------------#

@app.callback(
    Output("county-choropleth", 'figure'),
    [Input('years', 'value'),
     Input("region", "value"),
     Input("graph-type", "value")])     # @cache.memoize(timeout=cfg['timeout'])
def update_graph(year, region, gtype):
    if gtype == 'Price':
        df = regional_price_data
    else:
        df = regional_percentage_delta_data
    return get_figure(df[year][region], regional_geo_data[region],
                      region, gtype, year)

#----------------------------------------------------#

def price_volume_ts(price, volume, sector):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    #- Volume bar graph -------------------------#
    colorsDict = {'D':'#957DAD', 'S':'#AAC5E2', 'T':'#FDFD95', 'F':'#F4ADC6'}
    #colorsDict = {'D':'#4D4BA7', 'S':'#B156B8', 'T':'#E77B42', 'F':'#ECF560'}

    #legend_dict = {'D':'Detached', 'S':'Semi-Detached', 'T':'Terraced', 'F':'Flats/Maosonettes'}
    for ptype in ['D', 'S', 'T', 'F']:
        fig.add_trace(
            go.Bar(x=cfg['Years'],
                   y=volume['Sales Volume'][volume['Property Type']==ptype],
                   marker_color=colorsDict[ptype],
                   name=ptype
                  ),
            secondary_y=False
        )

    #- Price time series ------------------------#
    fig.add_trace(
        go.Scatter(x=price.index, y=price.values, marker_color='cyan',
                   mode='lines+markers', name=f"'{sector}' Avg. Price"),
        secondary_y=True,
    )

    #- Set Axes ---------------------------------#
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Sale Volume", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Avg. Price (£)", secondary_y=True)

    #- Layout------------------------------------#
    fig.update_layout(title=f'{sector}',
                      plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'],
                      autosize=True,
                      barmode='stack',
                      margin={'l': 20, 'b': 30, 'r': 10, 't': 40},
                      legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1,
                                xanchor="right",
                                x=1
                            ),
                      font_color=colors['text'])

    return fig

#----------------------------------------------------#

def price_ts(df, title):
    fig = px.scatter(df, labels=dict(value="Average Price (£)", variable="PostCode"),
                     title=title)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(margin={'l': 20, 'b': 30, 'r': 10, 't': 60},
                      plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'],
                      autosize=True,
                      font_color=colors['text'])
    return fig

#----------------------------------------------------#

@app.callback(
    Output('price-time-series', 'figure'),
    [Input('county-choropleth', 'clickData'),
     Input('county-choropleth', 'selectedData')]) #@cache.memoize(timeout=cfg['timeout'])
def update_price_timeseries(clickData, selectedData):
    # If selectedData:
    if selectedData is not None and len(selectedData['points']) > 0 and \
       selectedData != state['last_selectedData']:
        sector =[_dict['location'] for _dict in selectedData['points']][:cfg['topN']]
        title = f"Average price for {len(sector)} sectors (Up to a maximum of {cfg['topN']} is shown)"
        state['last_selectedData'] = selectedData
        return price_ts(price_df[sector], title)

    # clickData:
    else:
        sector = clickData['points'][0]['location']
        state['last_clickData'] = clickData
        df = type_df[sector].reset_index()
        df.rename(columns={sector: 'Sales Volume'}, inplace=True)

        return price_volume_ts(price_df[sector], df, sector)

#----------------------------------------------------#

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

print(f"Data Preparation completed in {time.time()-t0 :.1f} seconds")
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(
    #     port=8050,
    #     host='0.0.0.0'
    # )

    # Terminal cmd to run:
    # gunicorn app:server -b 0.0.0.0:8050
