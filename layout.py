import dash_bootstrap_components as dbc
from dash import dcc, html

from config import config as cfg


REGIONS = [
	"Greater London",
	"South East",
	"South West",
	"Midlands",
	"North England",
	"Wales",
]

COLORS = {"background": "#1F2630", "text": "#7FDBFF"}

NOTES_TOOLTIP = dcc.Markdown(
	"""
1. Property type "Other" is filtered from the house price data.
2. School ranking (2018-2019) is the best of GCSE and A-Level rankings.
3. GCSE ranking can be misleading — subjects like Classics and Latin are excluded from scoring, unfairly penalising some schools.
""",
	style={"fontSize": "0.75rem", "lineHeight": "1.3"},
)

DATA_SOURCES = html.Div(
	[
		html.Span("Other data sources:", style={"fontWeight": "bold"}),
		html.Ul(
			[
				html.Li(dcc.Link("OpenStreetMap", href="https://www.openstreetmap.org", target="_blank")),
				html.Li(dcc.Link("Postcode regions mapping", href="https://www.whichlist2.com/knowledgebase/uk-postcode-map/", target="_blank")),  # noqa: E501
				html.Li([
					dcc.Link("Postcode boundary data", href="https://www.opendoorlogistics.com/data/", target="_blank"),  # noqa: E501
					" from ",
					dcc.Link("opendoorlogistics.com", href="https://www.opendoorlogistics.com", target="_blank"),
					" — contains Royal Mail data © Royal Mail copyright and database right 2015; contains National Statistics data © Crown copyright and database right 2015",  # noqa: E501
				]),
				html.Li([
					dcc.Link("School 2019 performance data", href="https://www.gov.uk/school-performance-tables", target="_blank"),  # noqa: E501
					" (",
					dcc.Link("Attainment 8", href="https://www.locrating.com/Blog/attainment-8-and-progress-8-explained.aspx", target="_blank"),  # noqa: E501
					" for GCSE, ",
					dcc.Link("Average Point Score", href="https://dera.ioe.ac.uk/26476/3/16_to_18_calculating_the_average_point_scores_2015.pdf", target="_blank"),  # noqa: E501
					" for A-Level)",
				]),
			],
			style={"margin": "0", "paddingLeft": "20px"},
		),
	],
	style={"fontSize": "0.75rem", "lineHeight": "1.3"},
)


def create_layout(app, data):
	initial_year = data["initial_year"]
	initial_region = data["initial_region"]
	initial_sector = data["initial_sector"]
	regional_price_data = data["regional_price_data"]

	return html.Div(
		id="root",
		style={"height": "100vh", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
		children=[
			# Header -------------------------------------------------#
			html.Div(
				id="header",
				children=[
					html.Div(
						[
							html.Div(
								[html.H1(children="England and Wales House Prices")],
								style={
									"display": "inline-block",
									"width": "74%",
									"padding": "10px 0px 0px 20px",
								},
							),
							html.Div(
								[html.H6(children="Created with")],
								style={
									"display": "inline-block",
									"width": "10%",
									"textAlign": "right",
									"padding": "0px 20px 0px 0px",
								},
							),
							html.Div(
								[
									html.A(
										[
											html.Img(
												src=app.get_asset_url("dash-logo.png"),
												style={"height": "100%", "width": "100%"},
											)
										],
										href="https://plotly.com/",
										target="_blank",
									)
								],
								style={
									"display": "inline-block",
									"width": "14%",
									"textAlign": "right",
									"padding": "0px 10px 0px 0px",
								},
							),
						]
					),
				],
			),
			html.Div(
				[
					dcc.Link(
						f"HM Land Registry Price Paid Data from 01 Jan 1995 to {cfg['latest_date']}",  # noqa: E501
						href="https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads",  # noqa: E501
						target="_blank",
					)
				],
				style={"padding": "5px 0px 5px 20px"},
			),
			# Selection control -------------------------------------#
			html.Div(
				[
					html.Div(
						[
							dcc.Dropdown(
								id="region",
								options=[{"label": r, "value": r} for r in REGIONS],
								value=initial_region,
								clearable=False,
								style={"color": "black"},
							)
						],
						style={
							"display": "inline-block",
							"padding": "0px 5px 10px 15px",
							"width": "15%",
						},
						className="one columns",
					),
					html.Div(
						[
							dcc.Dropdown(
								id="year",
								options=[{"label": y, "value": y} for y in cfg["years"]],
								value=initial_year,
								clearable=False,
								style={"color": "black"},
							),
						],
						style={
							"display": "inline-block",
							"padding": "0px 5px 10px 0px",
							"width": "10%",
						},
						className="one columns",
					),
					html.Div(
						[
							dcc.Dropdown(
								id="postcode",
								options=[
									{"label": s, "value": s}
									for s in regional_price_data[initial_year][
										initial_region
									].Sector.values
								],
								value=[initial_sector],
								clearable=True,
								multi=True,
								style={"color": "black"},
							),
						],
						style={
							"display": "inline-block",
							"padding": "0px 5px 10px 0px",
							"width": "40%",
						},
						className="seven columns",
					),
					html.Div(
						[
							dbc.RadioItems(
								id="graph-type",
								options=[
									{"label": i, "value": i}
									for i in ["Price", "Volume", "Yr-to-Yr price ±%"]
								],
								value="Price",
								inline=True,
							)
						],
						style={
							"display": "inline-block",
							"textAlign": "center",
							"padding": "5px 0px 10px 10px",
							"width": "33%",
						},
						className="two columns",
					),
				],
				style={"padding": "5px 0px 10px 20px"},
				className="row",
			),
			# App Container ------------------------------------------#
			html.Div(
				id="app-container",
				style={"flex": "1", "minHeight": "0", "overflow": "hidden"},
				children=[
					# Left Column ------------------------------------#
					html.Div(
						id="left-column",
						children=[
							html.Div(
								id="choropleth-container",
								style={"flex": "1", "minHeight": "0", "display": "flex", "flexDirection": "column"},
								children=[
									html.Div(
										[
											html.Div(
												[
													html.H5(id="choropleth-title", style={"whiteSpace": "nowrap"}),
												],
												style={
													"display": "inline-block",
													"width": "64%",
												},
												className="eight columns",
											),
											html.Div(
												[
													dcc.Checklist(
														id="school-checklist",
														options=[
															{
																"label": "Show Top 500 Schools",  # noqa: E501
																"value": "True",
															},
														],
														value=[],
														labelStyle={
															"display": "inline-block"
														},
														inputStyle={"margin-left": "10px"},
													)
												],
												style={
													"display": "inline-block",
													"textAlign": "right",
													"width": "34%",
												},
												className="four columns",
											),
										]
									),
									dcc.Graph(id="choropleth", style={"flex": "1", "minHeight": "0"}),
								],
							),
						],
						style={
							"display": "inline-flex",
							"flexDirection": "column",
							"padding": "20px 10px 10px 40px",
							"width": "59%",
							"height": "100%",
						},
						className="seven columns",
					),
					# Right Column ------------------------------------#
					html.Div(
						id="graph-container",
						children=[
							html.Div(
								[
									dcc.Checklist(
										id="property-type-checklist",
										options=[
											{"label": "F: Flats/Maisonettes", "value": "F"},
											{"label": "T: Terraced", "value": "T"},
											{"label": "S: Semi-Detached", "value": "S"},
											{"label": "D: Detached", "value": "D"},
										],
										value=["F", "T", "S", "D"],
										labelStyle={"display": "inline-block"},
										inputStyle={"margin-left": "10px"},
									),
								],
								style={"textAlign": "right"},
							),
							html.Div(
								[dcc.Graph(id="price-time-series", style={"height": "100%"})],
								style={"flex": "1", "minHeight": "0"},
							),
						],
						style={
							"display": "inline-flex",
							"flexDirection": "column",
							"padding": "20px 20px 10px 10px",
							"width": "39%",
							"height": "100%",
						},
						className="five columns",
					),
				],
				className="row",
			),
			# Notes and credits --------------------------#
			html.Div(
				[
					html.Div(
						[
							html.Span(
								"Notes ⓘ",
								id="notes-trigger",
								style={
									"cursor": "pointer",
									"textDecoration": "underline",
									"fontSize": "0.8rem",
									"marginRight": "30px",
								},
							),
							dbc.Tooltip(
								NOTES_TOOLTIP,
								target="notes-trigger",
								placement="top",
								style={
									"maxWidth": "500px",
									"textAlign": "left",
								},
							),
							DATA_SOURCES,
						],
						style={
							"padding": "5px 0px 5px 40px",
							"width": "69%",
							"display": "inline-block",
						},
						className="nine columns",
					),
					html.Div(
						[
							dcc.Markdown(
								"© 2020 Ivan Lai "
								+ "[[Blog]](https://www.ivanlai.project-ds.net/) "
								+ "[[Email]](mailto:ivanlai.uk.2020@gmail.com)"
							)
						],
						style={
							"textAlign": "right",
							"padding": "5px 20px 0px 0px",
							"width": "29%",
							"display": "inline-block",
						},
						className="three columns",
					),
				],
				className="row",
			),
		],
	)
