import dash
import numpy as np
from dash.dependencies import Input, Output, State

from config import config as cfg
from figures_utils import (
	get_average_price_by_year,
	get_figure,
	price_ts,
	price_volume_ts,
)
from layout import COLORS


def register_callbacks(app, cache, data):
	price_volume_df = data["price_volume_df"]
	regional_price_data = data["regional_price_data"]
	regional_percentage_delta_data = data["regional_percentage_delta_data"]
	regional_geo_data = data["regional_geo_data"]
	regional_geo_data_paths = data["regional_geo_data_paths"]
	regional_geo_sector = data["regional_geo_sector"]
	schools_top_500 = data["schools_top_500"]
	empty_series = data["empty_series"]

	@app.callback(
		Output("choropleth-title", "children"),
		[
			Input("region", "value"),
			Input("year", "value"),
			Input("graph-type", "value"),
			Input("school-checklist", "value"),
		],
	)
	def update_map_title(region, year, gtype, school):
		if len(school) > 0:
			return "Top 500 schools (Postcode selection disabled)"
		elif gtype == "Price":
			return f"Avg house price (all property types) by postcode sector in {region}, {year}"  # noqa: E501
		elif gtype == "Volume":
			return f"Sales Volume (all property types) by postcode sector in {region}, {year}"
		else:
			if year == cfg["start_year"]:
				return f"Data from {year - 1} to {year} not available"
			else:
				return f"Yr-to-yr average price % change in {region}, from {year - 1} to {year}"

	@app.callback(
		Output("postcode", "options"),
		[Input("region", "value"), Input("year", "value")],
	)
	def update_region_postcode(region, year):
		return [
			{"label": s, "value": s}
			for s in regional_price_data[year][region].Sector.values
		]

	@app.callback(
		Output("choropleth", "figure"),
		[
			Input("year", "value"),
			Input("region", "value"),
			Input("graph-type", "value"),
			Input("postcode", "value"),
			Input("school-checklist", "value"),
		],
	)
	def update_choropleth(year, region, gtype, sectors, school):
		if gtype in ["Price", "Volume"]:
			df = regional_price_data[year][region]
		else:
			df = regional_percentage_delta_data[year][region]

		changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
		geo_sectors = dict()

		if "region" not in changed_id:
			for k in regional_geo_data[region].keys():
				if k != "features":
					geo_sectors[k] = regional_geo_data[region][k]
				else:
					geo_sectors[k] = [
						regional_geo_sector[region][sector]
						for sector in sectors
						if sector in regional_geo_sector[region]
					]

		fig = get_figure(
			df,
			app.get_asset_url(regional_geo_data_paths[region]),
			region,
			gtype,
			year,
			geo_sectors,
			school,
			schools_top_500,
		)

		return fig

	@app.callback(
		Output("price-time-series", "figure"),
		[Input("postcode", "value"), Input("property-type-checklist", "value")],
	)
	@cache.memoize(timeout=cfg["timeout"])
	def update_price_timeseries(sectors, ptypes):
		if len(sectors) == 0:
			return price_ts(empty_series, "Please select postcodes", COLORS)

		if len(ptypes) == 0:
			return price_ts(
				empty_series, "Please select at least one property type", COLORS
			)

		df = price_volume_df.loc[
			np.isin(price_volume_df.index.get_level_values("Property Type"), ptypes),
			np.isin(price_volume_df.columns.get_level_values("Sector"), sectors),
		]
		df.reset_index(inplace=True)
		avg_price_df = get_average_price_by_year(df, sectors)

		if len(sectors) == 1:
			index = [(a, b) for (a, b) in df.columns if a != "Average Price"]
			volume_df = df[index]
			volume_df.columns = volume_df.columns.get_level_values(0)
			return price_volume_ts(avg_price_df, volume_df, sectors, COLORS)
		else:
			title = f"Average prices for {len(sectors)} sectors"
			return price_ts(avg_price_df, title, COLORS)

	@app.callback(
		Output("postcode", "value"),
		[
			Input("choropleth", "clickData"),
			Input("choropleth", "selectedData"),
			Input("region", "value"),
			Input("school-checklist", "value"),
			State("postcode", "value"),
			State("choropleth", "clickData"),
		],
	)
	def update_postcode_dropdown(
		clickData, selectedData, region, school, postcodes, clickData_state
	):
		if dash.callback_context.triggered[0]["value"] is None:
			return postcodes

		changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]

		if len(school) > 0 or "school" in changed_id:
			return []

		if "region" in changed_id:
			postcodes = []
		elif "selectedData" in changed_id:
			postcodes = [D["location"] for D in selectedData["points"][: cfg["topN"]]]
		elif clickData is not None and "location" in clickData["points"][0]:
			sector = clickData["points"][0]["location"]
			if sector in postcodes:
				postcodes.remove(sector)
			elif len(postcodes) < cfg["topN"]:
				postcodes.append(sector)
		return postcodes
