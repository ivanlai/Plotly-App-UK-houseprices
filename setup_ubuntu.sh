#!/bin/bash

sudo apt-get update
sudo apt-get install git
sudo apt install python3-pip

# sudo pip3 install dash==1.16.0
# sudo pip3 install dash_bootstrap_components==0.10.6
sudo pip3 install gunicorn
sudo pip3 install -r requirements.txt

git clone https://github.com/ivanlai/apps-UK_houseprice.git

