# katxgpu
A tensor-core accelerated GPU-based X-engine.

TODO: Update as development happens

## Requirements
python3.6 and above

## Installation
1. Create a python 3.6 virtual environment.
2. Install all required python packages: `pip install -r requirements.txt`
3. Install the katxgpu package:`pip install -e .`

## Configuring pre-commit workflow
This makes use of [black](https://pypi.org/project/black/), [flake8](https://flake8.pycqa.org/en/latest/), [mypy](https://mypy.readthedocs.io/en/stable/index.html) and [pydocstyle](http://www.pydocstyle.org/en/5.0.2/index.html) to check your code for formatting before every commit. Not using this will make people unhappy and your pull requests will be rejected with extreme prejudice.

1. Enter a python 3.6 virtual environment
2. Run `pip install -r requirements-dev.txt`
3. Run `pre-commit install`

## Test Framework

The test framework has been implemented using [pytest](https://docs.pytest.org).
To run the framework, run the command `pytest` from the katxgpu parent directory.

This assumes the package installation and pre-commit configuration has already been done.

