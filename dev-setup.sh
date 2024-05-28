#!/bin/bash

################################################################################
# Copyright (c) 2020-2022, 2024 National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

set -e # So we can stop on a failure instead of bumbling along.

echo "Creating a virtual env for katgpucbf development"
python3.10 -m venv .venv --prompt katgpucbf
echo "Activating the virtual env"
source .venv/bin/activate

pip install --upgrade pip # stops pip from complaining
pip install -c requirements-dev.txt wheel # Makes the process a bit slicker.
echo "Installing requirements for running and development"
pip install -r requirements.txt -r requirements-dev.txt

echo "Installing module in editable mode for development"
pip install -e '.[test,doc]'

echo "Building documentation for your convenience"
make -C doc html

echo "Setting up pre-commit"
pre-commit install

# Print a different message based on whether you've sourced the
# file. Works on bash, may not on other shells.
(return 0 2>/dev/null) && echo "Done." || echo "Don't forget to source the new virtual env before you start!"
