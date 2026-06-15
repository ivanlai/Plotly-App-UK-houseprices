import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def postcode_region():
	"""Postcode prefix → region lookup dict."""
	return {
		"SW": "Greater London",
		"EC": "Greater London",
		"SE": "South East",
		"BN": "South East",
		"BS": "South West",
		"B": "Midlands",
		"LS": "North England",
		"CF": "Wales",
	}
