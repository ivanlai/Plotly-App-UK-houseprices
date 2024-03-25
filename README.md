# Plotly App: UK houseprices

- Visualize average England and Wales house prices and sales volume by postcode (sector) from 1995 to 2020
- Show average house price and sales volume trends.
- Show top 500 schools in UK.

Plotly web app ([https://ukhouseprice.project-ds.net/](https://ukhouseprice.project-ds.net/))

![Screenshot](https://github.com/ivanlai/apps-UK_houseprice/blob/master/images/Screenshot-plotly-app.png)

## Deployment on Pythonanywhere

In Pythonanywhere bash console:

- Run setup_ubuntu.sh to setup base environment 

        ./setup_ubuntu.sh

- Setup virtualenv and install libraries:

        mkvirtualenv py38 --python=/usr/bin/python3.8
        pip install -r requirements.txt

In the "Web" tab in the Pythonanywhere webpage after login:

- Update the wsgi file in the Code section (to be the same as wsgi.py in repo).

- Make sure the virtualenv path is set in the Virtualenv section.
- For debugging, inspect the log files in the Log files section.