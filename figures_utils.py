import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from plotly.subplots import make_subplots
from config import config as cfg

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


def get_figure(df, geo_data, region, gtype, year, geo_sectors, school, schools_top_500):
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


def price_volume_ts(price, volume, sector, colors):

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


def price_ts(df, title, colors):
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


def get_average_price_by_year(df, sectors):
    avg_price_df = pd.DataFrame()
    for sector in sectors:
        dot_product = (df[('Count', sector)]*df[('Average Price', sector)]).groupby(df.Year).sum()
        _sum = df[('Count', sector)].groupby(df.Year).sum()
        avg_price_df[sector] =  np.round((dot_product / _sum)/1000) * 1000

    return avg_price_df
