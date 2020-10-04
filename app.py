import os
import re
import sys
import time
import random
import logging

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
from flask_caching import Cache

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

""" ------------------------------------------
 Configurations
------------------------------------------ """
cfg = dict()

cfg['start_year']       = 1995
cfg['end_year']         = 2020
cfg['Years']            = list(range(cfg['start_year'], cfg['end_year']+1))

cfg['latest date']     = "31 July 2020"

appDataPath = '/home/ivanlai/apps-UK_houseprice/appData'
if os.path.isdir(appDataPath):
    cfg['app_data_dir'] = appDataPath #For Pythonanywhere
else:
    cfg['app_data_dir'] = 'appData'

cfg['topN']             = 12
cfg['timeout']          = 2*60     # Used in flask_caching
cfg['cache threshold']  = 30000    # corresponds to ~250MB

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

cfg['logging format'] = 'pid %(process)5s [%(asctime)s] ' + \
                        '%(levelname)8s: %(message)s'

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
logging.basicConfig(format=cfg['logging format'], level=logging.INFO)
logging.info(f"System: {sys.version}")

t0 = time.time()

""" ------------------------------------------
 House Price Data
------------------------------------------ """
price_volume_df = pd.read_csv(os.path.join(cfg['app_data_dir'], 'price_volume.csv'))
price_volume_df = price_volume_df.set_index(['Year', 'Property Type', 'Sector']).unstack(level=-1)
price_volume_df.fillna(value=0, inplace=True)

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
    """ ref: https://plotly.com/python/builtin-colorscales/
    """

    _cfg = cfg['plotly_config'][region]

    config = {'doubleClickDelay': 1000} #Set a high delay to make double click easier

    if gtype == 'Price':
        min_value = np.percentile(np.array(df.Price), 5)
        max_value = np.percentile(np.array(df.Price), _cfg['maxp'])
        z_vec = df['Price']
        text_vec = df['text']
        colorscale = "YlOrRd"
        # colorscale = "Sunsetdark"
        title = "Avg Price (£)"
    else:
        min_value = np.percentile(np.array(df['Percentage Change']), 10)
        max_value = np.percentile(np.array(df['Percentage Change']), 90)
        z_vec = df['Percentage Change']
        text_vec = ''
        colorscale = "Picnic"
        # colorscale = "Jet"
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

#---------------------------------------------
# initial values:

initial_year   = max(cfg['Years'])
initial_region = 'Greater London'

sectors = regional_price_data[initial_year][initial_region]['Sector'].values
initial_sector = random.choice(sectors)
empty_series = pd.DataFrame(np.full(len(cfg['Years']), np.nan), index=cfg['Years'])
empty_series.rename(columns={0: ''}, inplace=True)

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
cache = Cache(app.server, config={'CACHE_TYPE': 'filesystem',
                                  'CACHE_DIR': 'cache',
                                  'CACHE_THRESHOLD': cfg['cache threshold']})
app.config.suppress_callback_exceptions = True

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
                                     'width': '75%',
                                     'padding': '10px 0px 0px 20px'}), #padding: top, right, bottom, left
                    html.Div([html.H6(children='Powered by')],
                              style={'display': 'inline-block',
                                     'width': '10%',
                                     'textAlign': 'right',
                                     'padding': '0px 20px 0px 0px'}), #padding: top, right, bottom, left
                    html.Div([
                        html.A([
                            html.Img(src=app.get_asset_url("dash-logo.png"),
                                           style={'height': '100%',
                                                  'width' : '100%'})
                        ], href='https://plotly.com/', target='_blank')
                        ], style={'display': 'inline-block',
                                  'width': '15%',
                                  'textAlign': 'right',
                                  'padding': '0px 25px 0px 0px'}),
                ]),
            ],
        ),

        html.Div([
            dcc.Link(f"HM Land Reigstry Price Paid Data from 01 Jan 1995 to {cfg['latest date']}",
                     href='https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads',
                     target='_blank',
                     #style={'color': colors['text']}
            )
        ], style={'padding': '5px 0px 5px 20px'}),

        # Selection control -------------------------------------#
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='region',
                    options=[{'label': r, 'value': r} for r in regions],
                    value=initial_region,
                    clearable=False,
                    style={'color': 'black'})
                ], style={'display': 'inline-block',
                          'padding': '0px 5px 10px 15px',
                          'width': '15%'},
                   className="one columns"
                ),
            html.Div([
                dcc.Dropdown(
                    id="year",
                    options=[{'label': y, 'value': y} for y in cfg['Years']],
                    value=initial_year,
                    clearable=False,
                    style={'color': 'black'}),
                ], style={'display': 'inline-block',
                          'padding': '0px 5px 10px 0px',
                          'width': '10%'},
                   className="one columns"
            ),
            html.Div([
                dcc.Dropdown(
                    id='postcode-sector',
                    options=[{'label': s, 'value': s} for s in
                             regional_price_data[initial_year][initial_region].Sector.values],
                    value=[initial_sector],
                    clearable=True,
                    multi=True,
                    style={'color': 'black'}),
                ], style={'display': 'inline-block',
                          'padding': '0px 5px 10px 0px',
                          'width': '55%'},
                   className="seven columns"
            ),
            html.Div([
                dbc.RadioItems(
                    id='graph-type',
                    options=[{'label': i, 'value': i} for i in ['Price', 'Yr-to-Yr ±%']],
                    value='Price',
                    inline=True)
                ], style={'display': 'inline-block',
                          'textAlign': 'center',
                          'padding': '5px 0px 10px 10px',
                          'width': '20%'},
                  className="two columns"
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
                                html.H5(
                                    id="choropleth-title",
                                    children=f"Average house prices (all property types) \
                                               by postcode sector in \
                                               {initial_region}, {initial_year}"),

                                dcc.Graph(id="county-choropleth",
                                          clickData={'points': [{'location': initial_sector}]},
                                          figure = get_figure(regional_price_data[initial_year][initial_region],
                                                              regional_geo_data[initial_region],
                                                              initial_region,
                                                              gtype='Price',year=initial_year)
                                ),
                            ],
                        ),
                    ], style={'display': 'inline-block',
                              'padding': '20px 10px 10px 40px',
                              'width': "59%"},
                       className="seven columns"
                ),

                # Right Column ------------------------------------#
                html.Div(
                    id="graph-container",
                    children=[
                        html.Div([
                            html.H6(
                                dcc.Checklist(
                                    id='type-checklist',
                                    options=[
                                        {'label': 'F: Flats/Maisonettes', 'value': 'F'},
                                        {'label': 'T: Terraced', 'value': 'T'},
                                        {'label': 'S: Semi-Detached', 'value': 'S'},
                                        {'label': 'D: Detached', 'value': 'D'}
                                    ],
                                    value=['F', 'T', 'S', 'D'],
                                    labelStyle={'display': 'inline-block'},
                                    inputStyle={"margin-left": "10px"}
                                ),
                            )
                        ], style={'textAlign': 'right'}),

                        html.Div([dcc.Graph(id='price-time-series')]),

                    ], style={'display': 'inline-block',
                              'padding': '20px 10px 10px 10px',
                              'width': '39%'},
                       className="five columns"
                ),
            ],
            className="row"
        ),

        # Notes and credits --------------------------#
        html.Div([
            dcc.Markdown('''
                         **Note:** Property type "Other" have been filtered from the house price data.

                         **Other data sources:**
                         - [OpenStreetMap](https://www.openstreetmap.org)
                         - [Postcode boundary data](https://www.opendoorlogistics.com/data/)
                         from [www.opendoorlogistics.com](https://www.opendoorlogistics.com)
                         - Contains Royal Mail data © Royal Mail copyright and database right 2015
                         - Contains National Statistics data © Crown copyright and database right 2015
                         - [Postcode regions mapping](https://www.whichlist2.com/knowledgebase/uk-postcode-map/)
                         '''
                        )
            ], style={'textAlign': 'left',
                      'padding': '10px 0px 5px 20px'}
        ),

        html.H6([
            dcc.Markdown("© 2020 Ivan Lai [[Email]](mailto:ivanlai.me@gmail.com)")
            ], style={'padding': '5px 0px 10px 20px'}
        )
    ]
)

################################################################

""" Update choropleth-title with year and graph-type update
"""
@app.callback(
    Output('choropleth-title', 'children'),
    [Input('region', 'value'),
     Input('year', 'value'),
     Input('graph-type', 'value')])
def update_map_title(region, year, gtype):
    if gtype == 'Price':
        return f'Average house prices (all property types) by postcode sector in {region}, {year}'
    else:
        if year == 1995:
            return f'Data from {year-1} to {year} not available'
        else:
            return f'Year-to-year price % change by postcode sector in {region}, from {year-1} to {year}'

#----------------------------------------------------#

""" Update postcode dropdown options with region selection
"""
@app.callback(
    Output('postcode-sector', 'options'),
    [Input('region', 'value'),
     Input('year', 'value')])
def update_region_postcode(region, year):
    return [{'label': s, 'value': s} for s in
             regional_price_data[year][region].Sector.values]

#----------------------------------------------------#

""" Update choropleth-graph with year, region and graph-type update
"""
@app.callback(
    Output('county-choropleth', 'figure'),
    [Input('year', 'value'),
     Input('region', 'value'),
     Input('graph-type', 'value')])
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

    for ptype in ['D', 'S', 'T', 'F']:
        fig.add_trace(
            go.Bar(x=cfg['Years'],
                   y=volume.loc[volume['Property Type']==ptype, 'Count'],
                   marker_color=colorsDict[ptype],
                   name=ptype
                  ),
            secondary_y=False
        )

    #- Price time series ------------------------#
    fig.add_trace(
        go.Scatter(x=price.index, y=[p[0] for p in price[sector].values],
                   marker_color='cyan', mode='lines+markers', name=f"Avg. Price"),
        secondary_y=True,
    )

    #- Set Axes ---------------------------------#
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Sales Volume (Bar Chart)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Avg. Price (£)", secondary_y=True)

    #- Layout------------------------------------#
    fig.update_layout(title=f'{sector[0]}',
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
    fig = px.scatter(df, labels=dict(value="Average Price (£)", variable="PostCodes"),
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

def get_average_price_by_year(df, sectors):
    avg_price_df = pd.DataFrame()
    for sector in sectors:
        dot_product = (df[('Count', sector)]*df[('Average Price', sector)]).groupby(df.Year).sum()
        _sum = df[('Count', sector)].groupby(df.Year).sum()
        avg_price_df[sector] =  np.round((dot_product / _sum)/1000) * 1000

    return avg_price_df

#----------------------------------------------------#

""" Update price-time-series with clickData, selectedData or psotcode updates
"""
@app.callback(
    Output('price-time-series', 'figure'),
    [Input('county-choropleth', 'selectedData'),
     Input('postcode-sector', 'value'),
     Input('type-checklist', 'value')])
@cache.memoize(timeout=cfg['timeout'])
def update_price_timeseries(selectedData, sectors, ptypes):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if ('selectedData' in changed_id and selectedData is not None) or \
       ('type-checklist' in changed_id and len(sectors) == 0):
        sectors = [_dict['location'] for _dict in selectedData['points']][:cfg['topN']]
        title = f"Average prices for {len(sectors)} sectors (Up to a maximum of {cfg['topN']} is shown)"
    else:
        title = f"Average prices for {len(sectors)} sectors"

    #--------------------------------------------------#
    if ('type-checklist' not in changed_id) and (len(sectors) == 0 or isinstance(sectors, str)):
        return price_ts(empty_series, 'Please select postcodes')

    if len(ptypes)==0:
        return price_ts(empty_series, 'Please select at least one property type')

    #--------------------------------------------------#
    df = price_volume_df.iloc[np.isin(price_volume_df.index.get_level_values('Property Type'), ptypes),
                              np.isin(price_volume_df.columns.get_level_values('Sector'), sectors)]
    df.reset_index(inplace=True)
    avg_price_df = get_average_price_by_year(df, sectors)

    if len(sectors) == 1:
        index = [(a, b) for (a, b) in df.columns if a != 'Average Price']
        volume_df = df[index]
        volume_df.columns = volume_df.columns.get_level_values(0)
        return price_volume_ts(avg_price_df, volume_df, sectors)
    else:
        return price_ts(avg_price_df, title)

#----------------------------------------------------#

""" Update postcode dropdown values with clickData, selectedData and region
"""
@app.callback(
    Output('postcode-sector', 'value'),
    [Input('county-choropleth', 'clickData'),
     Input('county-choropleth', 'selectedData'),
     Input('region', 'value'),
     State('postcode-sector', 'value')]) #@cache.memoize(timeout=cfg['timeout'])
def update_postcode_dropdown(clickData, selectedData, region, postcode):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'region' in changed_id or 'selectedData' in changed_id:
        return []
    else:
        sector = clickData['points'][0]['location']
        postcode = set(postcode)
        postcode.add(sector)
        return list(postcode)

#----------------------------------------------------#

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

logging.info(f'Data Preparation completed in {time.time()-t0 :.1f} seconds')

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    logging.info(sys.version)

    # If running locally in Anaconda env:
    if 'conda-forge' in sys.version:
        app.run_server(debug=True)

    # If running on AWS production
    else:
        app.run_server(
            port=8050,
            host='0.0.0.0'
        )

"""
Terminal cmd to run:
gunicorn app:server -b 0.0.0.0:8050
"""
