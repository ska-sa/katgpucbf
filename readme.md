# MeerKAT GPU X-Engine Repository

TODO: Update as development happens

## Requirements
python3.6 and above

## Installation
1. Create a python 3.6 virtual environment.
2. Install all required python packages using `pip install -r requirements.txt`

## Installing pre-commit workflow
This makes use of [black](https://pypi.org/project/black/) to check your code for formatting before every commit. Not 
using this will make people unhappy and your commits will be rejected with extreme prejudice.

1. Enter a python 3.6 virtual environment
2. Run `pip install -r requirements-dev.txt`

## Test Framework
The test framework has been implemented using [pytest](https://docs.pytest.org).

To run the framework, run the command `python -m pytest` from the katxgpu parent directory.
To install all required packages
