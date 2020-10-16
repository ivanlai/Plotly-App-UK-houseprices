import os
import sys
import time
import json
import random
import logging
from copy import copy, deepcopy

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

import warnings
warnings.filterwarnings("ignore")


""" ----------------------------------------------------------------------------
 Configurations
---------------------------------------------------------------------------- """
cfg = dict()

cfg['start_year']       = 1995
cfg['end_year']         = 2020

cfg['Years']            = list(range(cfg['start_year'], cfg['end_year']+1))
cfg['latest date']      = "31 August 2020"

#When running in Pythonanywhere
appDataPath = '/home/ivanlai/apps-UK_houseprice/appData'
assetsPath  = '/home/ivanlai/apps-UK_houseprice/assets'

if os.path.isdir(appDataPath):
    cfg['app_data_dir'] = appDataPath
    cfg['assets dir']   = assetsPath
    cfg['cache dir']    = 'cache'

#when running locally
else:
    cfg['app_data_dir'] = 'appData'
    cfg['assets dir']   = 'assets'
    cfg['cache dir']    = '/tmp/cache'

cfg['topN']             = 50

cfg['timeout']          = 5*60     # Used in flask_caching
cfg['cache threshold']  = 10000    # corresponds to ~350MB max

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


""" ----------------------------------------------------------------------------
 House Price Data
---------------------------------------------------------------------------- """
price_volume_df = pd.read_csv(os.path.join(cfg['app_data_dir'], 'price_volume.csv'))
price_volume_df = price_volume_df.set_index(['Year', 'Property Type', 'Sector']).unstack(level=-1)
price_volume_df.fillna(value=0, inplace=True)


""" ----------------------------------------------------------------------------
 Regional Price, percentage delta and Volume Data
---------------------------------------------------------------------------- """

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


""" ----------------------------------------------------------------------------
 Geo Data
---------------------------------------------------------------------------- """
regional_geo_data = dict()
regional_geo_data_paths = dict()
for region in cfg['plotly_config']:
    infile = os.path.join(cfg['assets dir'], f'geodata_{region}.csv')
    regional_geo_data_paths[region] = infile

    with open(infile, "r") as read_file:
        regional_geo_data[region] = json.load(read_file)

#------------------------------------------------#

def get_geo_sector(geo_data):
    Y = dict()
    for feature in geo_data['features']:
        sector = feature['properties']['name']
        Y[sector] = feature
    return Y

regional_geo_sector = dict()
for k, v in regional_geo_data.items():
    regional_geo_sector[k] = get_geo_sector(v)


""" ----------------------------------------------------------------------------
 School Data
---------------------------------------------------------------------------- """

schools_top_500 = pd.read_csv(os.path.join(cfg['app_data_dir'], f'schools_top_500.csv'))
schools_top_500['Best Rank'] *= -1 #reverse the rankings solely for display purpose


""" ----------------------------------------------------------------------------
 Making Graphs
---------------------------------------------------------------------------- """

def get_scattergeo(df):
    fig = go.Figure()
    fig.add_trace(
        px.scatter_mapbox(df,
                          lat="Latitude", lon="Longitude",
                          color="Best Rank",
                          # color_discrete_sequence=['White'],
                          size=np.ones(len(df)),
                          size_max=8,
                          opacity=1
        ).data[0]
    )
    fig.update_traces(hovertemplate=df['Info'])

    return fig

#--------------------------------------------#

def get_Choropleth(df, geo_data, arg, marker_opacity,
                   marker_line_width, marker_line_color, fig=None):

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
            go.Choroplethmapbox(
                geojson = geo_data,
                locations = df['Sector'],
                featureidkey = "properties.name",
                colorscale = arg['colorscale'],
                z = arg['z_vec'],
                zmin = arg['min_value'],
                zmax = arg['max_value'],
                text = arg['text_vec'],
                hoverinfo="text",
                marker_opacity = marker_opacity,
                marker_line_width = marker_line_width,
                marker_line_color = marker_line_color,
                colorbar_title = arg['title'],
          )
    )
    return fig

#--------------------------------------------#

def get_figure(df, geo_data, region, gtype, year, geo_sectors, school):
    """ ref: https://plotly.com/python/builtin-colorscales/
    """
    config = {'doubleClickDelay': 1000} #Set a high delay to make double click easier

    _cfg = cfg['plotly_config'][region]

    arg = dict()
    if gtype == 'Price':
        arg['min_value'] = np.percentile(np.array(df.Price), 5)
        arg['max_value'] = np.percentile(np.array(df.Price), _cfg['maxp'])
        arg['z_vec'] = df['Price']
        arg['text_vec'] = df['text']
        arg['colorscale'] = "YlOrRd"
        arg['title'] = "Avg Price (£)"

    elif gtype == 'Volume':
        arg['min_value'] = np.percentile(np.array(df.Volume), 5)
        arg['max_value'] = np.percentile(np.array(df.Volume), 95)
        arg['z_vec'] = df['Volume']
        arg['text_vec'] = df['text']
        arg['colorscale'] = "Plasma"
        arg['title'] = "Sales Volume"

    else:
        arg['min_value'] = np.percentile(np.array(df['Percentage Change']), 10)
        arg['max_value'] = np.percentile(np.array(df['Percentage Change']), 90)
        arg['z_vec'] = df['Percentage Change']
        arg['text_vec'] = df['text']
        arg['colorscale'] = "Picnic"
        arg['title'] = "Avg. Price %Change"

    #-------------------------------------------#
    # Main Choropleth:
    fig = get_Choropleth(df, geo_data, arg, marker_opacity=0.4,
                         marker_line_width=1, marker_line_color='#6666cc')

    #-------------------------------------------#
    # School scatter_geo plot
    if len(school) > 0:
        fig.update_traces(showscale=False)
        school_fig = get_scattergeo(schools_top_500)
        fig.add_trace(school_fig.data[0])

        fig.layout.coloraxis.colorbar.title = 'School Ranking'
        fig.layout.coloraxis.colorscale = px.colors.diverging.Portland
        fig.layout.coloraxis.colorbar.tickvals = [-10, -100, -200, -300, -400, -500]
        fig.layout.coloraxis.colorbar.ticktext = [f'Top {i}' for i in [1, 100, 200, 300, 400, 500]]

    #------------------------------------------#
    """
    mapbox_style options:
    'open-street-map', 'white-bg', 'carto-positron', 'carto-darkmatter',
    'stamen-terrain', 'stamen-toner', 'stamen-watercolor'
    """
    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_zoom=_cfg['zoom'],
                      autosize=True,
                      font=dict(color="#7FDBFF"),
                      paper_bgcolor="#1f2630",
                      mapbox_center = {"lat": _cfg['centre'][0] , "lon": _cfg['centre'][1]},
                      uirevision=region,
                      margin={"r":0,"t":0,"l":0,"b":0}
                     )

    #-------------------------------------------#
    # Highlight selections:
    if geo_sectors is not None and len(school)==0:
        fig = get_Choropleth(df, geo_sectors, arg, marker_opacity=1.0,
                             marker_line_width=3, marker_line_color='aqua', fig=fig)

    return fig


""" ----------------------------------------------------------------------------
 App Settings
---------------------------------------------------------------------------- """

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
initial_geo_sector = [regional_geo_sector[initial_region][initial_sector]]

empty_series = pd.DataFrame(np.full(len(cfg['Years']), np.nan), index=cfg['Years'])
empty_series.rename(columns={0: ''}, inplace=True)


""" ----------------------------------------------------------------------------
 Dash App
---------------------------------------------------------------------------- """
# Select theme from: https://www.bootstrapcdn.com/bootswatch/

app = dash.Dash(
            __name__,
             meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
             ],
             external_stylesheets = [dbc.themes.DARKLY]
             # external_stylesheets = [dbc.themes.CYBORG]
        )

server = app.server #Needed for gunicorn
cache = Cache(server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': cfg['cache dir'],
            'CACHE_THRESHOLD': cfg['cache threshold']
            })
app.config.suppress_callback_exceptions = True

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
                                     'width': '74%',
                                     'padding': '10px 0px 0px 20px'}), #padding: top, right, bottom, left
                    html.Div([html.H6(children='Created with')],
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
                                  'width': '14%',
                                  'textAlign': 'right',
                                  'padding': '0px 10px 0px 0px'}),
                ]),
            ],
        ),

        html.Div([
            dcc.Link(f"HM Land Registry Price Paid Data from 01 Jan 1995 to {cfg['latest date']}",
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
                    id='postcode',
                    options=[{'label': s, 'value': s} for s in
                             regional_price_data[initial_year][initial_region].Sector.values],
                    value=[initial_sector],
                    clearable=True,
                    multi=True,
                    style={'color': 'black'}),
                ], style={'display': 'inline-block',
                          'padding': '0px 5px 10px 0px',
                          'width': '40%'},
                   className="seven columns"
            ),
            html.Div([
                dbc.RadioItems(
                    id='graph-type',
                    options=[{'label': i, 'value': i} for i in ['Price', 'Volume', 'Yr-to-Yr price ±%']],
                    value='Price',
                    inline=True)
                ], style={'display': 'inline-block',
                          'textAlign': 'center',
                          'padding': '5px 0px 10px 10px',
                          'width': '33%'},
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
                                html.Div([
                                    html.Div([
                                        html.H5(id="choropleth-title"),
                                        ], style={'display': 'inline-block',
                                                  'width': '64%'},
                                           className="eight columns"
                                    ),
                                    html.Div([
                                        dcc.Checklist(
                                            id='school-checklist',
                                            options=[
                                                {'label': 'Show Top 500 Schools', 'value': 'True'},
                                            ],
                                            value=[],
                                            labelStyle={'display': 'inline-block'},
                                            inputStyle={"margin-left": "10px"})
                                        ], style={'display': 'inline-block',
                                                  'textAlign': 'right',
                                                  'width': '34%'},
                                           className="four columns"
                                    ),
                                ]),
                                dcc.Graph(id="choropleth"),
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
                            dcc.Checklist(
                                id='property-type-checklist',
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
                        ], style={'textAlign': 'right'}),

                        html.Div([dcc.Graph(id='price-time-series')]),

                    ], style={'display': 'inline-block',
                              'padding': '20px 20px 10px 10px',
                              'width': '39%'},
                       className="five columns"
                ),
            ],
            className="row"
        ),

        # Notes and credits --------------------------#
        html.Div([
            html.Div([
                dcc.Markdown('''
                             **Notes:**

                             1. Property type "Other" is filtered from the house price data.
                             2. School ranking is the best of GCSE and A-Level rankings.
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
                             '''
                            )
                ], style={'textAlign': 'left',
                          'padding': '0px 0px 5px 40px',
                          'width': '69%'},
                   className="nine columns"
            ),

            html.Div([
                    dcc.Markdown("© 2020 Ivan Lai " + \
                                 "[[Blog]](https://www.ivanlai.project-ds.net/) " + \
                                 "[[Email]](mailto:ivanlai.uk.2020@gmail.com)")
                ], style={'textAlign': 'right',
                          'padding': '10px 20px 0px 0px',
                          'width': '29%'},
                   className="three columns"
            )
        ], className="row")
    ]
)

""" ----------------------------------------------------------------------------
 Callback functions:
 Overview:
 region, year, graph-type, school -> choropleth-title
 region, year -> postcode options
 region, year, graph-type, postcode-value, school -> choropleth
 postcode-value, property-type-checklist -> price-time-series
 choropleth-clickData, choropleth-selectedData, region, postcode-State
                                                        -> postcode-value
---------------------------------------------------------------------------- """

""" Update choropleth-title with year and graph-type update
"""
@app.callback(
    Output('choropleth-title', 'children'),
    [Input('region', 'value'),
     Input('year', 'value'),
     Input('graph-type', 'value'),
     Input('school-checklist', 'value')])
def update_map_title(region, year, gtype, school):
    if len(school) > 0:
        return "Top 500 schools (Postcode selection disabled)"
    elif gtype == 'Price':
        return f'Average house prices (all property types) by postcode sector in {region}, {year}'
    elif gtype == 'Volume':
        return f'Sales Volume (all property types) by postcode sector in {region}, {year}'
    else:
        if year == 1995:
            return f'Data from {year-1} to {year} not available'
        else:
            return f'Yr-to-yr price % change in {region}, from {year-1} to {year}'

#----------------------------------------------------#

""" Update postcode dropdown options with region selection
"""
@app.callback(
    Output('postcode', 'options'),
    [Input('region', 'value'),
     Input('year', 'value')])
def update_region_postcode(region, year):
    return [{'label': s, 'value': s} for s in
             regional_price_data[year][region].Sector.values]

#----------------------------------------------------#

""" Update choropleth-graph with year, region, graph-type update & sectors
"""
@app.callback(
    Output('choropleth', 'figure'),
    [Input('year', 'value'),
     Input('region', 'value'),
     Input('graph-type', 'value'),
     Input('postcode', 'value'),
     Input('school-checklist', 'value')]) #@cache.memoize(timeout=cfg['timeout'])
def update_Choropleth(year, region, gtype, sectors, school):

    # Graph type selection------------------------------#
    if gtype in ['Price', 'Volume']:
        df = regional_price_data[year][region]
    else:
        df = regional_percentage_delta_data[year][region]

    # For high-lighting mechanism ----------------------#
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    geo_sectors = dict()

    if 'region' not in changed_id:
        for k in regional_geo_data[region].keys():
            if k != 'features':
                geo_sectors[k] = regional_geo_data[region][k]
            else:
                geo_sectors[k] = [regional_geo_sector[region][sector] for sector in sectors
                                  if sector in regional_geo_sector[region]]

    # Updating figure ----------------------------------#
    fig = get_figure(df, regional_geo_data[region], region, gtype, year,
                     geo_sectors, school)

    return fig

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
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1,
                                  xanchor="right",
                                  x=1),
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

""" Update price-time-series with postcode updates and graph-type
"""
@app.callback(
    Output('price-time-series', 'figure'),
    [Input('postcode', 'value'),
     Input('property-type-checklist', 'value')])
@cache.memoize(timeout=cfg['timeout'])
def update_price_timeseries(sectors, ptypes):

    if len(sectors) == 0:
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
        title = f"Average prices for {len(sectors)} sectors"
        return price_ts(avg_price_df, title)

#----------------------------------------------------#

""" Update postcode dropdown values with clickData, selectedData and region
"""
@app.callback(
    Output('postcode', 'value'),
    [Input('choropleth', 'clickData'),
     Input('choropleth', 'selectedData'),
     Input('region', 'value'),
     Input('school-checklist', 'value'),
     State('postcode', 'value'),
     State('choropleth', 'clickData')])
def update_postcode_dropdown(clickData, selectedData, region, school, postcodes,
                             clickData_state):

    # Logic for initialisation or when Schoold sre selected
    if dash.callback_context.triggered[0]['value'] is None :
        return postcodes

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if len(school) > 0 or 'school' in changed_id:
        clickData_state = None
        return []

    #--------------------------------------------#

    if 'region' in changed_id:
        postcodes = []
    elif 'selectedData' in changed_id:
        postcodes = [D['location'] for D in selectedData['points'][:cfg['topN']]]
    elif clickData is not None and \
         'location' in clickData['points'][0]:
        sector = clickData['points'][0]['location']
        if sector in postcodes:
            postcodes.remove(sector)
        elif len(postcodes) < cfg['topN']:
            postcodes.append(sector)
    return postcodes

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

    # If running on AWS/Pythonanywhere production
    else:
        app.run_server(
            port=8050,
            host='0.0.0.0'
        )

""" ----------------------------------------------------------------------------
Terminal cmd to run:
gunicorn app:server -b 0.0.0.0:8050
---------------------------------------------------------------------------- """
