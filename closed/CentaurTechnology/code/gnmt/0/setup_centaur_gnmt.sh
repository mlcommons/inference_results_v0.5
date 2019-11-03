#!/bin/bash
#
# Script to install everything that we need and get an environment setup for running GNMT
set -e

echo "## Setting up a py36 virtualenv..."
python -m venv env

echo "## installing all py36 requirements..."
env/bin/python -m pip install -r requirements/requirements-py3.txt

#echo "## building loadgen binary / bindings..."
#cd ../../../../loadgen;
#CFLAGS="-std=c++14"  ../v0.5/translation/gnmt/tensorflow/env/bin/python setup.py develop
#cd ../v0.5/translation/gnmt/tensorflow

echo "## Downloading data / trained model..."
./download_dataset.sh
./download_trained_model.sh

echo "## Done!"
