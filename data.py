import time
import logging

import numpy as np
import pandas as pd

from config import config as cfg
from utils import (
	get_price_volume_df,
	get_regional_data,
	get_regional_geo_data,
	get_regional_geo_sector,
	get_schools_data,
)


def load_all_data():
	t0 = time.time()

	price_volume_df = get_price_volume_df()
	regional_price_data = get_regional_data("sector_price")
	regional_percentage_delta_data = get_regional_data("sector_percentage_delta")
	regional_geo_data, regional_geo_data_paths = get_regional_geo_data()
	regional_geo_sector = get_regional_geo_sector(regional_geo_data)
	schools_top_500 = get_schools_data()

	initial_year = max(cfg["years"])
	initial_region = "Greater London"

	empty_series = pd.DataFrame(np.full(len(cfg["years"]), np.nan), index=cfg["years"])
	empty_series.rename(columns={0: ""}, inplace=True)

	logging.info(f"Data Preparation completed in {time.time() - t0:.1f} seconds")

	return {
		"price_volume_df": price_volume_df,
		"regional_price_data": regional_price_data,
		"regional_percentage_delta_data": regional_percentage_delta_data,
		"regional_geo_data": regional_geo_data,
		"regional_geo_data_paths": regional_geo_data_paths,
		"regional_geo_sector": regional_geo_sector,
		"schools_top_500": schools_top_500,
		"initial_year": initial_year,
		"initial_region": initial_region,
		"empty_series": empty_series,
	}
