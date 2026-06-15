import sys

project_home = "/home/ivanlai/apps-UK_houseprice"
if project_home not in sys.path:
	sys.path = [project_home] + sys.path

from app import server as application  # noqa: E402, F401
