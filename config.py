import os

appDataPath = "/home/ivanlai/apps-UK_houseprice/appData"
assetsPath  = "/home/ivanlai/apps-UK_houseprice/assets"

if os.path.isdir(appDataPath):
    app_data_dir = appDataPath
    assets_dir = assetsPath
    cache_dir = "cache"
else:
    app_data_dir = "appData"
    assets_dir = "assets"
    cache_dir = "/tmp/cache"


config = {
    "start_year": 1995,
    "end_year": 2022,
    "latest date": '31 December 2022',

    "app_data_dir": app_data_dir,
    "assets dir": assets_dir,
    "cache dir": cache_dir,

    "topN": 50,

    "timeout": 5 * 60,  # Used in flask_caching
    "cache threshold": 10_000,  # corresponds to ~350MB max

    "regions_lookup": {
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
    },

    "plotly_config":{
        'North England':   {'centre': [54.3, -2.0], 'maxp': 99, 'zoom': 6.5},
         'Wales':          {'centre': [52.4, -3.3], 'maxp': 99, 'zoom': 6.9},
         'Midlands':       {'centre': [52.8, -1.0], 'maxp': 99, 'zoom': 7},
         'South West':     {'centre': [51.1, -3.7], 'maxp': 99, 'zoom': 6.9},
         'South East':     {'centre': [51.5, -0.1], 'maxp': 90, 'zoom': 7.3},
         'Greater London': {'centre': [51.5, -0.1], 'maxp': 80, 'zoom': 8.9},
    },

    "logging format": "pid %(process)5s [%(asctime)s] %(levelname)8s: %(message)s"
}

config['Years'] = list(range(config['start_year'], config['end_year']+1))
