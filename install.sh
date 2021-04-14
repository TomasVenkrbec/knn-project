#!/usr/bin/bash

# Create and start virtual environment
python3 -m venv env-knn
source env-knn/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

# Download and install required packages
pip install -r requirements.txt

# Prepare folders
mkdir dataset
mkdir dataset/ImageNet
mkdir dataset/CIFAR-100
mkdir images_to_colorize/
mkdir results/
mkdir snapshots/
