# katxgpu
A tensor-core accelerated GPU-based X-engine.

## TODOs

katxgpu is still in early development with more modules being added every few weeks. Attempts are made for each of these
modules to be complete, but when there are lingering issues that eventually need to be resolved but are not critical
to development. Many of these TODOs are listed in the relevant files, but sometimes the TODO has no associated file.
In this case the TODO is listed here.
1. A number of google doc links are present in this readme and the [readme](src/README.md) in the src file. These
files must be converted to PDF and the links updated accordingly when this program nears release. The 
[display_fsim_multicast_packets.py](scratch/display_fsim_multicast_packets.py) and [fsim.cpp](scratch/fsim.cpp)  also
have links that must be updated.
2. Move Jenkins file and docker containers to use Ubuntu 20.04 and Python 3.8
3. The scratch folder is getting a bit crowded. Its original purpose was to contain a bunch of misc files that had 
no real place in the repo, but now its contains the fsim and useful python files. The fsim could go in its 
own folder and then another folder called scripts should be added where things like receiver_example.py will go. 
If this scripts folder does get created, I think that a step should be added in the Jenkinsfile to check that the 
scripts run on each change to the repo as there is a chance that some of them will not be run for many months 
before they are needed. Additionally some of the documentation in this readme references the scripts in the scratch 
folder, this will need to be updated to reflect the new links to prevent stale documentation.
4. Figure out the repo license and update the license section below to reflect this. Additionally, there is a 
license classifier in the `setuptools.setup()` function in [setup.py](setup.py) that will need to be updated.
6. Most of the repos documentation is just in the form of readmes and inline commenting. It is worth investigating 
something like [sphinx](http://www.sphinx-doc.org) that can generate a proper readthedocs page for this repo.

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
4. Linux - This system has been tested on Ubuntu 18.04
5. Mellanox OFED Drivers for ibverbs functionality (Tested on version 5.1 and above)
6. The required python packages can be found in the [requirements.txt](./requirements.txt) and the
[requirements-dev.txt](./requirements-dev.txt) files.
7. The SPEAD2 submodule requires a number of system packages to be installed using `apt-get`. These are: autoconf,
libboost-all-dev, libibverbs-dev,librdmacm-dev, and libpcap-dev.

## Installation
In order to install the katxgpu module, the following commands must be run:
1. Install required C++ libraries for SPEAD2: `apt-get install autoconf libboost-all-dev libibverbs-dev librdmacm-dev libpcap-dev`
2. Create a python 3.6 virtual environment: `virtualenv -p python3.6 <venv name>`.
3. Activate virtual environment: `source <venv name>/bin/activate`
4. Install all required python packages: `pip install -r requirements.txt`
5. Checkout spead2 submodule: `git submodule update --init --recursive`
6. Install the katxgpu package: `pip install .`

NOTE: Due to the underlying complexity of turning the SPEAD2 C++ code into a python module, installing the katxgpu 
module can take quite a while as the SPEAD2 software is installed each time. Build times over a minute long are quite
normal. To reduce these build times look at using the [ccache](https://ccache.dev/) utility. 

If the F-Engine simulator needs to be run, the SPEAD2 C++ library needs to be installed. The following steps must be
followed in order to do this:
1. `sudo apt-get install autoconf libboost-all-dev libibverbs-dev librdmacm-dev libpcap-dev`
2. `pip install pycparser jinja2`
3. `cd /3rdparty/spead2`
4. `./bootstrap`
5. `./configure`
6. `make`
7. `sudo make install`

The SPEAD2 C++ install and the katxgpu module install run different programs and as such the once can be installed 
without the other.

## Configuring pre-commit workflow
This makes use of [black](https://pypi.org/project/black/), [flake8](https://flake8.pycqa.org/en/latest/), 
[mypy](https://mypy.readthedocs.io/en/stable/index.html) and [pydocstyle](http://www.pydocstyle.org/en/5.0.2/index.html)
to check your code for formatting before every commit. Not using this will make people unhappy and your pull requests
will be rejected with extreme prejudice.

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
4. The node the server runs on requires a a Mellanox ConnectX-5 (or newer) NIC. The Mellanox OFED drivers enabling 
ibverbs functionality on the NIC must also be installed.

This should all be happening automatically on SARAO's servers, but if you fork this repo and want to set up your own CI
server, these steps should help you on your way. A 
[document](https://docs.google.com/document/d/1iiZk7aEjsAcewM-wDX3Iz9osiTiyOhr3sYYzcmsv4mM/edit?usp=sharing) describes
in more detail how Jenkins is configured on SARAO's servers. This document is a work in progress.

## Running within a Docker container

This repository contains a Dockerfile for building a docker image that can launch katxgpu.

In order to build the container, the following command needs to be run from the top level katxgpu directory: 
`docker image build -t katxgpu .`

To run the container and open a terminal within the container run:
```
docker run \
    --gpus all \
    --network host \
    --ulimit=memlock=-1 \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --rm \
    -it \
    katxgpu
```


To launch the [receiver_example.py](scratch/receiver_example.py) in a container run the following command:
```
docker run \
    --gpus all \
    --network host \
    --ulimit=memlock=-1 \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --rm \
    -d \
    --name=katxgpu_container \
    katxgpu \
    python scratch/receiver_example.py
```

To view the output from the receiver script run the following command: `watch docker logs -t --tail 10 katxgpu_containe`

__NOTE__: There is quite a bit of overlap between the commands in the [Dockerfile](./Dockerfile) and the
[Jenkinsfile](./Jenkinsfile) and the requirements for running a docker container are the same requirements as mentioned
in the Jenkins CI section above. The explanation for the different arguments required to launch the docker container can
be found in the Jenkinsfile.

## SPEAD2 Network Side Software

This software uses the high performance SPEAD2 networking library for all high speed data transmission and reception.
The SPEAD2 library has been extended in C++ and this has been turned into a project submodule. This module can be
imported using `import katxgpu._katxgpu`.

The makeup of this module is quite complex. This [file](src/README.md) within this repo describes the entire module in
great detail. A simple example of how to use the receiver network code is shown in
[receiver_example.py](scratch/receiver_example.py) in the katxgpu/scratch directory. This example is probably the
quickest way to figure out how the receiver works.

The `katxgpu._katxgpu` module uses the SPEAD2 C++ bindings (not the python bindings) and as such requires the SPEAD2
submodule to be included in this repository. This module is located in the katxgpu/3rdparty directory.

__NOTE:__ The F-Engine 

### F-Engine Packet Simulator

In order to test the X-Engine code, data from the F-Engines needs to be received. In general an X-Engine needs to 
receive data for a subset of channels from N F-Engines where N is the telescope array size. This is complicated to 
configure and requires many F-Engines. In order to bypass this, an F-Engine simulator has been created that simulates
packets recieved at the X-Engine (i.e Packets from multiple F-Engines destined for the same X-Engine.) This simulator
requires a server with a Mellanox NIC to run. This fsim simulates the exact packet format from the SKARAB F-Engines. 
The SKARAB X-Engines ingest data from 4 different multicast streams. This simulator only simulates data from a single
multicast stream - if more streams are required, more instances of this simulator need to be run in parallel.

In order to build the fsim, navigate to the [scratch](./scratch) folder and run `make`

The minimum command to run the fsim from the [scratch](./scratch) folder is: 
`sudo ./fsim --interface <interface_address> <multicast_address>[+y]:<port>`

Where:
 * `<interface_address>` is the ip address of the network interface to transmit the data out on.
 * `<multicast_address>` is the multicast address all packets are destined to. The optional `[+y]` argument will create 
`y` additional multicast streams with the same parameters each on a different multicast addresses consecutivly after the
base `<multicast_address>`.
 * `<port>` is the UDP port to transmit data to.

The above command will transmit data at about `7.8 * (y+1)` Gbps by default. 

See the fsim [source code](./scratch/fsim.cpp) for a  detailed description of how the F-Engine simulator works and the 
useful configuration arguments.

A rough description of the ingest packet format is described 
[here](https://docs.google.com/drawings/d/1lFDS_1yBFeerARnw3YAA0LNin_24F7AWQZTJje5-XPg/edit). The
[display_fsim_multicast_packets.py](scratch/display_fsim_multicast_packets.py) in katxgpu/scratch will capture packets
out of the F-Engine and display the most useful packet information. This will give an intuitive understanding of the
packet formats.

### Eliminating Packet Drops

The receiver software has been made to receive data at up to 68 Gbps per port on a dual-port Mellanox ConnectX-6 NIC
without dropping any packets. A standard server will likely be unable to do to accomplish this with its default
configuration. The following are recommended steps that can be taken to improve server performance - all these steps
need to be followed to eliminate packet drops:
1. P-states - P-states (or ACPI P states) optomise CPU frequency and voltage during operation. These p-states must be
set to enable maximum performance at all times. This is normally done within the system bios by disabling p-states
entirely. (NOTE: The setting is not always called p-states, look out for an ACPI setting instead. It may require a bit
of a search to get right.).
2. C-states - C-states allow the CPU to enter a power saving mode when idle. This often results in the CPU reducing its
frequency to conserve power. This does not interact well with SPEAD2 as the CPU will often reduces its frequency while
the NIC is receiving packets. When the CPU needs to process the packets, the reduced frequency results in the packets
not being processed fast enough and being dropped. C-states must be disabled. One way to do this is to run the following
command in the terminal: `cpupower frequency-set --governor performance`. It is left to the user to ensure that this
command is working correctly. 
3. NUMA boundaries - Dual socket servers (and even some single socket servers) have different memory access regions.
Additionally different sockets have different PCIe root complexes. It is advisable to keep the GPU and NIC within the
same NUMA region and use some utility like `numactl` to ensure that the corresponding CPU and memory region are used
with the GPU and NIC. If this is not done, there is a chance that the data will move between CPU sockets, leading to
bottlenecks at the bus between the two sockets.

