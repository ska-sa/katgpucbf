#!/bin/bash
set -e # So we can stop on a failure instead of bumbling along.

echo "Creating a virtual env for katcbfgpu development"
python3.8 -m venv .venv --prompt katgpucbf
echo "Activating the virtual env"
source .venv/bin/activate

pip install --upgrade pip # stops pip from complaining
pip install wheel pip-tools # Makes the process a bit slicker.
echo "Installing requirements for running and development"
pip install -r requirements.txt -r requirements-dev.txt

echo "Installing module in editable mode for development"
pip install -e .

echo "Compiling the C modules needed to run xbengine tests"
make -C test/xbgpu

echo "Building documentation for your convenience"
make -C doc html

echo "Setting up pre-commit"
pre-commit install

# Print a different message based on whether you've sourced the
# file. Works on bash, may not on other shells.
(return 0 2>/dev/null) && echo "Done." || echo "Don't forget to source the new virtual env before you start!"
