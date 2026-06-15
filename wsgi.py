import os
import sys

project_home = "/home/ivanlai/apps-UK_houseprice"
if project_home not in sys.path:
	sys.path = [project_home] + sys.path

os.environ.setdefault("APP_DATA_DIR", f"{project_home}/appData")
os.environ.setdefault("ASSETS_DIR", f"{project_home}/assets")
os.environ.setdefault("CACHE_DIR", "cache")

from app import server as application  # noqa: E402, F401
