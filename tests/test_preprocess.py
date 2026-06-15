import pandas as pd

from scripts.preprocess import (
	lookup_region,
	clean_pp_df,
	get_sector_df,
)


class TestLookupRegion:
	def test_normal_postcode(self, postcode_region):
		assert lookup_region(postcode_region, "SW1A 1AA") == "Greater London"

	def test_two_letter_prefix(self, postcode_region):
		assert lookup_region(postcode_region, "BS1 4DJ") == "South West"

	def test_single_letter_prefix(self, postcode_region):
		assert lookup_region(postcode_region, "B1 1BB") == "Midlands"

	def test_prefix_not_in_dict(self, postcode_region):
		assert lookup_region(postcode_region, "ZZ1 1AA") == ""

	def test_no_digit_returns_empty(self, postcode_region):
		assert lookup_region(postcode_region, "INVALID") == ""


class TestCleanPpDf:
	def _make_raw_df(self, rows):
		"""Build a DataFrame mimicking raw Land Registry CSV structure."""
		records = []
		for price, date, postcode, ptype in rows:
			records.append(
				{
					0: "ID",
					1: price,
					2: date,
					3: postcode,
					4: ptype,
					5: "N",
					6: "F",
					7: "1",
					8: "STREET",
					9: "",
					10: "TOWN",
					11: "",
					12: "COUNTY",
					13: "",
				}
			)
		return pd.DataFrame(records)

	def test_sector_extraction(self, postcode_region):
		df = self._make_raw_df([(250000, "2024-06-15", "SW1A 1AA", "D")])
		result = clean_pp_df(df, {}, postcode_region)
		assert result.iloc[0]["Sector"] == "SW1A 1"

	def test_year_month_extraction(self, postcode_region):
		df = self._make_raw_df([(250000, "2024-06-15", "SW1A 1AA", "D")])
		result = clean_pp_df(df, {}, postcode_region)
		assert result.iloc[0]["Year"] == "2024"
		assert result.iloc[0]["Month"] == "06"
		assert result.iloc[0]["Year-Month"] == "2024-06"

	def test_property_type_O_filtered(self, postcode_region):
		df = self._make_raw_df(
			[
				(250000, "2024-06-15", "SW1A 1AA", "D"),
				(300000, "2024-06-15", "SW1A 1AB", "O"),
			]
		)
		result = clean_pp_df(df, {}, postcode_region)
		assert len(result) == 1
		assert result.iloc[0]["Property Type"] == "D"

	def test_low_price_filtered(self, postcode_region):
		df = self._make_raw_df(
			[
				(5000, "2024-06-15", "SW1A 1AA", "D"),
				(250000, "2024-06-15", "SW1A 1AB", "D"),
			]
		)
		result = clean_pp_df(df, {}, postcode_region)
		assert len(result) == 1
		assert result.iloc[0]["Price"] == 250000

	def test_region_assigned(self, postcode_region):
		df = self._make_raw_df([(250000, "2024-06-15", "SW1A 1AA", "D")])
		result = clean_pp_df(df, {}, postcode_region)
		assert result.iloc[0]["Region"] == "Greater London"


class TestGetSectorDf:
	def test_aggregation_and_rounding(self, postcode_region):
		df = pd.DataFrame(
			{
				"Year": [2024, 2024, 2024],
				"Sector": ["SW1A 1", "SW1A 1", "EC1A 1"],
				"Price": [245000, 255000, 310000],
				"Post Code": ["SW1A 1AA", "SW1A 1AB", "EC1A 1AA"],
			}
		)
		result = get_sector_df(df, postcode_region)
		sw = result[result.Sector == "SW1A 1"].iloc[0]
		assert sw["Price"] == 250000  # mean=250000, rounds to 250K
		assert sw["Volume"] == 2

	def test_empty_sector_filtered(self, postcode_region):
		df = pd.DataFrame(
			{
				"Year": [2024, 2024],
				"Sector": ["SW1A 1", ""],
				"Price": [250000, 100000],
				"Post Code": ["SW1A 1AA", ""],
			}
		)
		result = get_sector_df(df, postcode_region)
		assert "" not in result.Sector.values

	def test_text_column_format(self, postcode_region):
		df = pd.DataFrame(
			{
				"Year": [2024],
				"Sector": ["SW1A 1"],
				"Price": [250000],
				"Post Code": ["SW1A 1AA"],
			}
		)
		result = get_sector_df(df, postcode_region)
		text = result.iloc[0]["text"]
		assert "SW1A 1" in text
		assert "250K" in text
