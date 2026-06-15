import numpy as np
import pandas as pd

from figures_utils import get_average_price_by_year


class TestGetAveragePriceByYear:
	def test_single_sector_weighted_average(self):
		"""Weighted avg of 500K (10 sales) and 300K (20 sales) = 366666 → rounds to 367K."""
		df = pd.DataFrame(
			{
				("Count", "SW1A 1"): [10, 20],
				("Average Price", "SW1A 1"): [500000, 300000],
				"Year": [2024, 2024],
			}
		)
		result = get_average_price_by_year(df, ["SW1A 1"])
		expected = np.round(((10 * 500000 + 20 * 300000) / 30) / 1000) * 1000
		assert result["SW1A 1"].iloc[0] == expected

	def test_multiple_sectors_independent(self):
		df = pd.DataFrame(
			{
				("Count", "SW1A 1"): [10],
				("Average Price", "SW1A 1"): [500000],
				("Count", "EC1A 1"): [20],
				("Average Price", "EC1A 1"): [300000],
				"Year": [2024],
			}
		)
		result = get_average_price_by_year(df, ["SW1A 1", "EC1A 1"])
		assert result["SW1A 1"].iloc[0] == 500000
		assert result["EC1A 1"].iloc[0] == 300000

	def test_rounding_to_nearest_thousand(self):
		"""Price of 345,678 across 1 sale → rounds to 346,000."""
		df = pd.DataFrame(
			{
				("Count", "SW1A 1"): [1],
				("Average Price", "SW1A 1"): [345678],
				"Year": [2024],
			}
		)
		result = get_average_price_by_year(df, ["SW1A 1"])
		assert result["SW1A 1"].iloc[0] == 346000

	def test_multiple_years(self):
		df = pd.DataFrame(
			{
				("Count", "SW1A 1"): [10, 15],
				("Average Price", "SW1A 1"): [200000, 300000],
				"Year": [2024, 2025],
			}
		)
		result = get_average_price_by_year(df, ["SW1A 1"])
		assert result.index.tolist() == [2024, 2025]
		assert result["SW1A 1"][2024] == 200000
		assert result["SW1A 1"][2025] == 300000
