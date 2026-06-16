"""
Data preprocessing pipeline for UK House Prices app.

Converts raw Land Registry price-paid CSVs into the appData/ files
that the Dash app reads at startup. Also produces HTML summary plots
in output/ for reference.

Usage:
    1. Download raw PP files to input/HousePriceData/Raw/
    2. Edit config.py: end_year, latest_date, pipeline.raw_price_files, pipeline.years_to_process
    3. Run: uv run python scripts/preprocess.py
"""

import os
import re
import sys
import time
import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from convertbng.util import convert_lonlat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg

pipe = cfg["pipeline"]

PP_RAW_DIR = os.path.join(pipe["houseprice_dir"], "Raw")
PP_PROCESSED_DIR = os.path.join(pipe["houseprice_dir"], "Processed")

pattern = re.compile(r"\d")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lookup_postcode(postcodes, x):
	if x in postcodes:
		return postcodes[x]
	return ""


def lookup_region(postcode_region, x):
	m = pattern.search(x)
	if m is None:
		return ""
	prefix = x[: m.start()]
	return postcode_region.get(prefix, "")


def load_postcode_lookups():
	postcodes_df = pd.read_csv(os.path.join(pipe["geodata_dir"], "ukpostcodes.csv"))
	postcodes = {
		row.postcode: [row.latitude, row.longitude] for row in postcodes_df.itertuples()
	}

	postcode_region_df = pd.read_csv(
		os.path.join(pipe["geodata_dir"], "PostCode Region.csv")
	)
	postcode_region = {
		row.Prefix: cfg["regions_lookup"][row.Region]
		for row in postcode_region_df.itertuples()
	}
	return postcodes, postcode_region


# ---------------------------------------------------------------------------
# Step 1: Process raw house price data
# ---------------------------------------------------------------------------


def clean_pp_df(df, postcodes, postcode_region):
	col = {
		1: "Price",
		2: "Date",
		3: "Post Code",
		4: "Property Type",
		5: "Old/New",
		6: "Duration",
	}
	df.rename(columns=col, inplace=True)
	df.fillna("", inplace=True)
	df["Address"] = (
		df[7]
		+ " "
		+ df[8]
		+ " "
		+ df[9]
		+ " "
		+ df[10]
		+ " "
		+ df[11]
		+ " "
		+ df[12]
		+ " "
		+ df[13]
	)
	df["Address"] = df["Address"].apply(lambda x: " ".join(x.split()))

	cols_to_drop = [c for c in df.columns if isinstance(c, int)]
	df.drop(cols_to_drop, axis=1, inplace=True)

	df = df.loc[df.Price > pipe["price_threshold"]]
	df = df.loc[df["Property Type"] != "O"]
	df.sort_values(by=["Date"], inplace=True, ignore_index=True)

	df["Post Code Coords"] = df["Post Code"].apply(
		lambda x: lookup_postcode(postcodes, x)
	)
	df["Year-Month"] = df["Date"].apply(lambda s: s[:7])
	df["Year"] = df["Date"].apply(lambda s: s[:4])
	df["Month"] = df["Date"].apply(lambda s: s[5:7])
	df["Sector"] = df["Post Code"].apply(lambda s: s[: s.find(" ") + 2])
	df["Region"] = df["Post Code"].apply(lambda s: lookup_region(postcode_region, s))

	return df


def process_raw_houseprice(postcodes, postcode_region):
	os.makedirs(PP_PROCESSED_DIR, exist_ok=True)

	for infile in pipe["raw_price_files"]:
		print(f"Processing {infile}")
		df = pd.read_csv(os.path.join(PP_RAW_DIR, infile), header=None)
		df = clean_pp_df(df, postcodes, postcode_region)
		print(f"  Transactions: {len(df):,}")

		for year in df.Year.unique():
			if int(year) in pipe["years_to_process"]:
				fname = f"pp-{year}.csv"
				df[df.Year == year].to_csv(
					os.path.join(PP_PROCESSED_DIR, fname), index=False
				)
				print(f"  Saved {fname}")
			else:
				print(f"  Skipping year {year} (not in years_to_process)")


# ---------------------------------------------------------------------------
# Step 2: Load all processed data
# ---------------------------------------------------------------------------


def _load_one_year(year):
	fname = os.path.join(PP_PROCESSED_DIR, f"pp-{year}.csv")
	if os.path.isfile(fname):
		df = pd.read_csv(fname)
		print(f"  {year}: {len(df):,} transactions")
		return df
	return None


def load_processed_data():
	print("Loading processed house price data...")
	with Pool(max(1, cpu_count() - 1)) as p:
		results = p.map(_load_one_year, cfg["years"])

	results = [r for r in results if r is not None]
	house_price_df = pd.concat(results, ignore_index=True)
	print(f"  Total rows: {len(house_price_df):,}")
	return house_price_df


# ---------------------------------------------------------------------------
# Step 3: Sector prices by year (for choropleth)
# ---------------------------------------------------------------------------


def get_sector_df(house_price_df, postcode_region):
	P = (
		house_price_df[["Year", "Sector", "Price"]]
		.groupby(by=["Year", "Sector"])
		.agg(["mean", "count"])
	)
	P.reset_index(inplace=True)
	P.columns = ["Year", "Sector", "Price", "Volume"]
	P = P.loc[P["Sector"] != ""]
	P["Region"] = P["Sector"].apply(lambda s: lookup_region(postcode_region, s))
	P["Price"] = P["Price"].apply(lambda s: int(np.round(s / 1000) * 1000))
	P["Display Price"] = P["Price"].apply(lambda x: f"{int(np.round(x / 1000)):,}K")
	P["text"] = (
		P["Sector"]
		+ "<br>"
		+ "Avg. Price: "
		+ P["Display Price"]
		+ "<br>"
		+ "Sales Volume: "
		+ P["Volume"].astype(str)
	)
	P.drop(columns=["Display Price"], inplace=True)
	return P


def save_sector_prices(sector_df):
	print("Saving sector prices by year...")
	sector_by_year = {}
	for year in cfg["years"]:
		sector_by_year[year] = sector_df[sector_df.Year == year].reset_index(drop=True)
		if pipe["save_output"]:
			fname = os.path.join(cfg["app_data_dir"], f"sector_price_{year}.csv")
			sector_by_year[year].to_csv(fname, index=False)
	print(f"  Saved {len(cfg['years'])} files")
	return sector_by_year


# ---------------------------------------------------------------------------
# Step 4: Percentage deltas by year (for choropleth)
# ---------------------------------------------------------------------------


def save_percentage_deltas(sector_by_year):
	print("Saving percentage deltas by year...")
	sector_price = {}
	for year in cfg["years"]:
		sector_price[year] = {}
		for sector, region, price in sector_by_year[year][
			["Sector", "Region", "Price"]
		].values:
			sector_price[year][sector] = [region, price]

	sector_delta = {}
	sector_delta[cfg["start_year"]] = {
		sector: [0, region]
		for sector, [region, _] in sector_price[cfg["start_year"]].items()
	}

	for y1, y2 in zip(cfg["years"][1:], cfg["years"][:-1]):
		sector_delta[y1] = {}
		for sector, [region, price] in sector_price[y1].items():
			if sector in sector_price[y2]:
				last_year_price = sector_price[y2][sector][1]
				delta = int(np.round(100 * (price - last_year_price) / last_year_price))
				sector_delta[y1][sector] = [delta, region]

	for year in cfg["years"]:
		tmp = pd.DataFrame.from_dict(
			sector_delta[year], orient="index", columns=["Percentage Change", "Region"]
		)
		tmp.reset_index(inplace=True)
		tmp.rename(columns={"index": "Sector"}, inplace=True)
		tmp["text"] = (
			tmp["Sector"]
			+ "<br>"
			+ "Price Change: "
			+ tmp["Percentage Change"].apply(str)
			+ "%"
		)
		if pipe["save_output"]:
			fname = os.path.join(
				cfg["app_data_dir"], f"sector_percentage_delta_{year}.csv"
			)
			tmp.to_csv(fname, index=False)

	print(f"  Saved {len(cfg['years'])} files")


# ---------------------------------------------------------------------------
# Step 5: Price volume by year and property type (for time-series)
# ---------------------------------------------------------------------------


def save_price_volume(house_price_df):
	print("Saving price volume data...")
	cols = ["Year", "Sector", "Property Type"]
	P = house_price_df[cols + ["Price"]].groupby(by=cols).agg(["count", "mean"])
	P.reset_index(inplace=True)
	P.columns = ["Year", "Sector", "Property Type", "Count", "Average Price"]
	P = P.loc[P["Sector"] != ""]

	if pipe["save_output"]:
		fname = os.path.join(cfg["app_data_dir"], "price_volume.csv")
		P.to_csv(fname, index=False)
		print(f"  Saved {fname} ({len(P):,} rows)")


# ---------------------------------------------------------------------------
# Step 6: School data
# ---------------------------------------------------------------------------

regnumber = re.compile(r"\d+")


def num_2_str(x):
	if x is not None and regnumber.match(str(x)):
		return float(x)
	return 0.0


def merge_df(df_x, df_y, key):
	col_x = set(df_x.columns)
	col_y = set(df_y.columns)
	cols = col_x.intersection(col_y)
	cols.discard(key)

	df_z = df_x.merge(df_y, how="outer", on=key)

	for col in cols:
		tmp = []
		for a, b in df_z[[f"{col}_x", f"{col}_y"]].values:
			tmp.append(a if isinstance(a, str) else b)
		df_z[col] = tmp
		df_z.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)

	return df_z


def process_school_data(postcode_region):
	print("Processing school data...")
	school_dir = pipe["school_dir"]

	fields = ["URN", "SCHNAME", "PCODE", "EGENDER", "AGERANGE", "ATT8SCR"]
	gcse_df = pd.read_csv(
		os.path.join(school_dir, "england_ks4final.csv"),
		usecols=fields,
		dtype={"URN": str},
	)
	gcse_df.dropna(inplace=True)
	gcse_df["ATT8SCR"] = gcse_df["ATT8SCR"].apply(num_2_str)
	gcse_df = gcse_df[gcse_df["ATT8SCR"] > 0]
	gcse_df.sort_values(
		by=["ATT8SCR"], ascending=False, ignore_index=True, inplace=True
	)
	gcse_df["GCSE rank"] = gcse_df.index + 1
	gcse_df.rename(columns={"EGENDER": "GENDER"}, inplace=True)

	fields = ["URN", "SCHNAME", "PCODE", "GEND1618", "AGERANGE", "TALLPPE_ALEV_1618"]
	alevel_df = pd.read_csv(
		os.path.join(school_dir, "england_ks5final.csv"),
		usecols=fields,
		dtype={"URN": str},
	)
	alevel_df.dropna(inplace=True)
	alevel_df["TALLPPE_ALEV_1618"] = alevel_df["TALLPPE_ALEV_1618"].apply(num_2_str)
	alevel_df = alevel_df[alevel_df["TALLPPE_ALEV_1618"] > 0]
	alevel_df.sort_values(
		by=["TALLPPE_ALEV_1618"], ascending=False, ignore_index=True, inplace=True
	)
	alevel_df["A-Level rank"] = alevel_df.index + 1
	alevel_df.rename(columns={"GEND1618": "GENDER"}, inplace=True)

	school_df = merge_df(gcse_df, alevel_df, "URN")
	school_df["GENDER"] = school_df["GENDER"].apply(lambda s: s.capitalize())

	tmp = []
	for gcse_rank, alevel_rank in school_df[["GCSE rank", "A-Level rank"]].values:
		tmp.append(int(np.nanmin([gcse_rank, alevel_rank])))
	school_df["Best Rank"] = tmp

	school_info_df = pd.read_csv(
		os.path.join(school_dir, "england_school_information.csv"),
		dtype={"URN": str},
	)
	school_df = school_df.merge(
		school_info_df[["URN", "MINORGROUP"]], how="left", on="URN"
	)
	school_df.rename(columns={"MINORGROUP": "Status"}, inplace=True)

	fields = ["URN", "Easting", "Northing"]
	school_loc_df = pd.read_csv(
		os.path.join(pipe["school_dir"], "edubasealldata20260616.csv"),
		usecols=fields,
		encoding="ISO-8859-1",
		dtype={"URN": str},
	)
	school_df = school_df.merge(
		school_loc_df[["URN", "Easting", "Northing"]], how="left", on="URN"
	)
	school_df = school_df[pd.notna(school_df.Easting) | pd.notna(school_df.Northing)]

	long, lat = convert_lonlat(school_df.Easting.values, school_df.Northing.values)
	school_df["Latitude"] = lat
	school_df["Longitude"] = long
	school_df.drop(columns=["Easting", "Northing"], inplace=True)

	# Hover text
	tmp = []
	info_cols = [
		"SCHNAME",
		"AGERANGE",
		"GENDER",
		"Status",
		"ATT8SCR",
		"GCSE rank",
		"TALLPPE_ALEV_1618",
		"A-Level rank",
	]
	for row in school_df[info_cols].values:
		text = row[0] + "<br>" + row[1] + " " + row[2] + " " + row[3] + " "
		if ~np.isnan(row[4]):
			text += "<br>" + "GCSE: A8S " + str(row[4]) + ", #" + f"{int(row[5]):,}"
		if ~np.isnan(row[6]):
			text += "<br>" + "A-level: APS " + str(row[6]) + ", #" + f"{int(row[7]):,}"
		tmp.append(text)

	school_df["Info"] = tmp
	school_df.drop(
		columns=[
			"SCHNAME",
			"AGERANGE",
			"GENDER",
			"Status",
			"GCSE rank",
			"A-Level rank",
		],
		inplace=True,
	)

	# Save top 500 and regional files
	n = 500
	gcse_top = school_df.sort_values(
		by=["ATT8SCR"], ascending=False, ignore_index=True
	)[:n]
	alevel_top = school_df.sort_values(
		by=["TALLPPE_ALEV_1618"], ascending=False, ignore_index=True
	)[:n]
	school_topN = pd.concat([gcse_top, alevel_top])
	school_topN.drop_duplicates(subset=["URN"], inplace=True)
	school_topN["Region"] = school_topN["PCODE"].apply(
		lambda x: lookup_region(postcode_region, x)
	)

	fname = os.path.join(cfg["app_data_dir"], f"schools_top_{n}.csv")
	school_topN.to_csv(fname, index=False)
	print(f"  Saved {fname} ({len(school_topN)} schools)")

	intermediate_dir = "intermediate_data"
	os.makedirs(intermediate_dir, exist_ok=True)
	regions = [
		"South East",
		"North England",
		"Midlands",
		"South West",
		"Greater London",
		"Wales",
	]
	for region in regions:
		if region != "South East":
			mask = school_topN.Region == region
		else:
			mask = (school_topN.Region == "South East") | (
				school_topN.Region == "Greater London"
			)
		fname = os.path.join(intermediate_dir, f"schools_{region}.csv")
		school_topN[mask].to_csv(fname, index=False)

	print(f"  Saved {len(regions)} regional school files")


# ---------------------------------------------------------------------------
# Step 7: Summary plots (HTML)
# ---------------------------------------------------------------------------


def save_summary_plots(house_price_df):
	print("Saving summary plots...")
	output_dir = os.path.join("output", str(cfg["end_year"]))
	os.makedirs(output_dir, exist_ok=True)

	# Plot 1: Average price and volume by month
	P = (
		house_price_df[["Year-Month", "Price"]]
		.groupby(by=["Year-Month"])
		.agg(["mean", "count"])
	)
	P.reset_index(inplace=True)
	P.columns = ["Year-Month", "Price", "Volume"]

	fig = make_subplots(specs=[[{"secondary_y": True}]])
	fig.add_trace(
		px.line(
			P, x="Year-Month", y="Price", color_discrete_sequence=["cornflowerblue"]
		).data[0],
		secondary_y=True,
	)
	fig.add_trace(
		px.bar(P, x="Year-Month", y="Volume", color_discrete_sequence=["violet"]).data[
			0
		],
		secondary_y=False,
	)
	fig.update_xaxes(showgrid=False)
	fig.update_yaxes(title_text="Avg. Price (£)", secondary_y=True)
	fig.update_yaxes(
		title_text="Sales Volume (Bar Chart)", secondary_y=False, showgrid=False
	)
	fig.update_layout(
		title="England & Wales average house price and sales volume from 1995"
	)
	path1 = os.path.join(output_dir, "price_volume_by_month.html")
	pio.write_html(fig, path1)
	print(f"  Saved {path1}")

	# Plot 2: Price by property type
	P2 = (
		house_price_df[["Year-Month", "Price", "Property Type"]]
		.groupby(by=["Year-Month", "Property Type"])
		.mean()
	)
	P2.reset_index(inplace=True)
	fig2 = px.line(P2, x="Year-Month", y="Price", color="Property Type")
	fig2.update_layout(
		title="England & Wales average house price from 1995 by Property Type"
	)
	path2 = os.path.join(output_dir, "price_by_property_type.html")
	pio.write_html(fig2, path2)
	print(f"  Saved {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
	t0 = time.time()
	print(f"Preprocessing pipeline started at {datetime.datetime.now()}")
	print(f"Year range: {cfg['start_year']}–{cfg['end_year']}")
	print()

	postcodes, postcode_region = load_postcode_lookups()

	if pipe["process_raw_pp"]:
		process_raw_houseprice(postcodes, postcode_region)
		print()

	house_price_df = load_processed_data()
	print()

	sector_df = get_sector_df(house_price_df, postcode_region)
	sector_by_year = save_sector_prices(sector_df)
	save_percentage_deltas(sector_by_year)
	save_price_volume(house_price_df)
	print()

	process_school_data(postcode_region)
	print()

	save_summary_plots(house_price_df)
	print()

	elapsed = time.time() - t0
	print(f"Pipeline completed in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
	main()
