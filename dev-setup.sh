#!/bin/bash
set -e # So we can stop on a failure instead of bumbling along.

echo "Creating a virtual env for katcbfgpu development"
python3.8 -m venv .venv
echo "Activating the virtual env"
source .venv/bin/activate

pip install wheel  # Makes the process a bit slicker.
echo "Installing package's requirements"
pip install -r requirements.txt
echo "Installing dev requirements"
pip install -r requirements-dev.txt
echo "Installing doc requirements"
pip install -r requirements-doc.txt

echo "Setting up pre-commit"
pre-commit install

echo "Done. Don't forget to source the new venv before you start!"
