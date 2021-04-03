#!/usr/bin/bash

# Create and start virtual environment
python3 -m venv env-knn
source env-knn/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

# Download and install required packages
pip install -r requirements.txt