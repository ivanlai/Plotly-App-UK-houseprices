import os
import json
import pandas as pd
import logging
from copy import deepcopy

from config import config as cfg


def get_price_volume_df():
    price_volume_df = pd.read_csv(os.path.join(cfg['app_data_dir'], 'price_volume.csv'))
    price_volume_df = price_volume_df.set_index(['Year', 'Property Type', 'Sector']).unstack(level=-1)
    price_volume_df.fillna(value=0, inplace=True)
    return price_volume_df


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


def get_regional_geo_data():
    regional_geo_data = dict()
    regional_geo_data_paths = dict()
    for region in cfg['plotly_config']:
        fname = f'geodata_{region}.json'
        regional_geo_data_paths[region] = fname

        infile = os.path.join(cfg['assets dir'], fname)
        with open(infile, "r") as read_file:
            regional_geo_data[region] = json.load(read_file)

    return regional_geo_data, regional_geo_data_paths


def get_regional_geo_sector(regional_geo_data):
    regional_geo_sector = dict()
    for k, v in regional_geo_data.items():
        regional_geo_sector[k] = get_geo_sector(v)
    return regional_geo_sector


def get_geo_sector(geo_data):
    Y = dict()
    for feature in geo_data['features']:
        sector = feature['properties']['name']
        Y[sector] = feature
    return Y


def get_schools_data():
    schools_top_500 = pd.read_csv(os.path.join(cfg['app_data_dir'], f'schools_top_500.csv'))
    schools_top_500['Best Rank'] *= -1 #reverse the rankings solely for display purpose
    return schools_top_500
