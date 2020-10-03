#!/bin/bash

sudo apt-get update
sudo apt-get install git
sudo apt install python3-pip

sudo pip3 install gunicorn
sudo pip3 install -r requirements.txt

# git clone https://github.com/ivanlai/apps-UK_houseprice.git

