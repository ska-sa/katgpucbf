#!/bin/bash

python3.8 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-doc.txt

pre-commit install