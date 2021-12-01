#!/usr/bin/env bash

set -e
# wrong by using:
# the path is not relative to the bootstap.sh
# export FLASK_APP=./main.py
export FLASK_APP=./src/main.py
export FLASK_ENV=development

flask run -h 0.0.0.0 -p 5000
