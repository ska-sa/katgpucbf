# katxgpu
A tensor-core accelerated GPU-based X-engine.

TODO: Update as development happens

## License
The license for this repository still needs to be specified. At the moment this repo is private so its not an issue.
When it eventually goes public, this will need to be specified. We need to check with John Romein (the author of the 
Tensor core X-Engine core kernel that is central to this repo) what license he is using. This will inform the choice of.
license here.

__DO NOT__ make this repo public before specifying the license.

## Requirements
1. python3.6 and above.
2. Nvidia GPU of compute capability 7.5 or above.
3. CUDA version 10.1 or above.
4. Mellanox OFED Drivers for ibverbs functionality (Tested on version 5.1 and above)

## Installation
TODO: update this to reflect SPEAD2 installation instructions
1. Create a python 3.6 virtual environment: `virtualenv -p python3.6 <venv name>`.
2. Activate virtual environment: `source <venv name>/bin/activate`
3. Install all required python packages: `pip install -r requirements.txt`
4. Checkout spead2 submodule: `git submodule update --init --recursive`
5. Install the katxgpu package: `pip install -e .`

SPEAD2 C++ install for simulator:
1. cd /3rdparty/spead2
2. ./bootstrap
3. ./configure
4. make
5. sudo make install

## Configuring pre-commit workflow
This makes use of [black](https://pypi.org/project/black/), [flake8](https://flake8.pycqa.org/en/latest/), [mypy](https://mypy.readthedocs.io/en/stable/index.html) and [pydocstyle](http://www.pydocstyle.org/en/5.0.2/index.html) to check your code for formatting before every commit. Not using this will make people unhappy and your pull requests will be rejected with extreme prejudice.

1. Enter your python 3.6 virtual environment
2. Run `pip install -r requirements-dev.txt`
3. Run `pre-commit install`

## Test Framework

The test framework has been implemented using [pytest](https://docs.pytest.org).
To run the framework, run the command `pytest` from the katxgpu parent directory.

This assumes the package installation and pre-commit configuration has already been done.

## Jenkins CI

Jenkins has been integrated into this repo. On every PR and at least once a day, Jenkins will scan the repo for changes
and if detected, will run the test pipeline. The test will run on the PR branch or the branch where there are detected 
changes (as long as the branch has a Jenkinsfile). The test configuration is specified in the
[Jenkinsfile](./Jenkinsfile) in this repo. The Jenkins build pipeline will execute this pipeline in a docker container.

In theory, any Jenkins server should be able run the pipeline from the Jenkinsfile, however for this repo, there are a 
few additional requirements:
1. The node the docker server run on requires a Nvidia GPU of compute capability 7.5 or above. Basically the GPU is
required to have Tensor cores.
2. The docker engine that Jenkins points to will need to be able to access the host GPU. Nvidia provides a tool
do this, called the [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime). This need to be
installed and then the `--gpus all` flag needs to be added when calling docker run (`docker run --gpus all ...`). The 
jenkins container takes care of addings this flag, but the user needs to ensure that the nvidia runtime container is 
installed.
3. The Nvidia driver installed on the host machine needs to be compatible with the cuda 10.1 as the unit test will run
on an image based on the nvidia/cuda:10.1-devel-ubuntu18.04 container.

This should all be happening automatically on SARAO's servers, but if you fork this repo and want to set up your own CI
server, these steps should help you on your way. A [document](https://docs.google.com/document/d/1iiZk7aEjsAcewM-wDX3Iz9osiTiyOhr3sYYzcmsv4mM/edit?usp=sharing) describes in more detail how Jenkins is configured on SARAO's servers. This document is a work in progress.

#TODO: get Jenkins working with Mellanox OFED

## SPEAD2 Network Side Software

ibverbs
Need to include as submodule as we are using custom C++ code for performance, not the python wrappings
Submodule is included in 3rdparty folder
Wraps C++ code in python susing pybind11. If you want to understand more go to: https://pybind11.readthedocs.io/en/stable/basics.html, however its quite complicated. Avoid the C++ code until you are ready.
Long build times - use ccache(may be worth putting in the install section)
fsim.cpp in /scratch

### F-Engine Packet Simulator


