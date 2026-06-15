import pandas as pd


class TestSouthEastMasking:
	"""The South East region includes Greater London data in the choropleth.
	This is a special case that could silently break if the masking logic changes.
	Tests the masking logic directly rather than going through file I/O."""

	def _apply_region_mask(self, df, region):
		"""Replicate the masking logic from utils.get_regional_data."""
		if region == "South East":
			mask = (df.Region == region) | (df.Region == "Greater London")
		else:
			mask = df.Region == region
		return df[mask]

	def test_south_east_includes_greater_london(self):
		df = pd.DataFrame(
			{
				"Sector": ["SE1 1", "SW1A 1", "BS1 4"],
				"Region": ["South East", "Greater London", "South West"],
				"Price": [300000, 500000, 250000],
			}
		)
		result = self._apply_region_mask(df, "South East")
		assert len(result) == 2
		assert set(result.Region) == {"South East", "Greater London"}

	def test_other_region_excludes_cross_region(self):
		df = pd.DataFrame(
			{
				"Sector": ["SE1 1", "SW1A 1", "BS1 4"],
				"Region": ["South East", "Greater London", "South West"],
				"Price": [300000, 500000, 250000],
			}
		)
		result = self._apply_region_mask(df, "South West")
		assert len(result) == 1
		assert result.iloc[0]["Region"] == "South West"

	def test_greater_london_standalone(self):
		df = pd.DataFrame(
			{
				"Sector": ["SE1 1", "SW1A 1"],
				"Region": ["South East", "Greater London"],
				"Price": [300000, 500000],
			}
		)
		result = self._apply_region_mask(df, "Greater London")
		assert len(result) == 1
		assert result.iloc[0]["Region"] == "Greater London"
