# katxgpu

This repository implements a GPU-based XB-Engine for the MeerKAT Extension Project. 

This repo makes use of the [SPEAD2](https://spead2.readthedocs.io/en/latest/index.html) library for high-performance
networking. SPEAD2 is designed to send and recieve multicast UDP data that conforms to the SPEAD protocol. More
information can be found [here](src/README.md). With the correct configuration, this engine is able to receive packets
at high data rates without dropping any.

Nvidia Tensor Cores have been used to greatly accelerate the correlation algorithm. 

The SARAO [katsdpsigproc](https://katsdpsigproc.readthedocs.io/) package has been used to implement the GPU-side processing.
This wraps both OpenCL and CUDA allowing for generic operations to be defined that can operate in both CUDA and OpenCL.
Currently all kernels have been written in CUDA. Additionally in order to make use of Tensor Cores, CUDA needs to be
used. Due to these reasons, the framework only compiles things down to CUDA, not OpenCL.

Both katsdpsigproc and SPEAD2 are compatible with [asyncio](https://docs.python.org/3/library/asyncio.html). This
XB-Engine has been written to run across multiple functions using asyncio to coordinate all the moving parts.

The majority of this code is written in python. The Python portion of this code is only used for controlling and
coordinating the engine. All heavy processing is handled on seperate C++ threads (in the SPEAD2 case) and on the GPU.
This allows the pipeline to process at very high data rates. (With the correct configuration, by running 10 instances
of this pipeline, a single Nvidia RTX 3080 GPU and Mellanox 100 GbE NIC should be able to process 70 Gbps quite easily
without dropping any packets.)  

TODO: The specific tools used for control and monitoring, metrics and logging need to be mentioned here when they
have been decided upon.

NOTE: Currently the B-Engine part of this engine has not been implemented.

## TODOs
The X-Engine in its current form works as expected. The B-Engine and control and monitoring still needs to be
implemented and there are some quality-of-life improvements that can be made. The recommeded improvements are as
follows:

katxgpu is still in early development with more modules being added every few weeks. Attempts are made for each of these
modules to be complete, but when there are lingering issues that eventually need to be resolved but are not critical
to development. Many of these TODOs are listed in the relevant files, but sometimes the TODO has no associated file.
In this case the TODO is listed here.
1. A number of google doc links are present in this readme and the [readme](src/README.md) in the src file. These
files must be converted to PDF and the links updated accordingly when this program nears release. The 
[display_fsim_multicast_packets.py](scratch/display_fsim_multicast_packets.py), [fsim.cpp](scratch/fsim.cpp) and 
[display_xengine_multicast_packets.py](scratch/display_xengine_multicast_packets.py), [xsend.py](katxgpu/xsend.py) 
also have links that must be updated. This README.md also has these links. 
2. Move Jenkins file and docker containers to use Ubuntu 20.04 and Python 3.8. Once this port has been done. Change the
[send_example.py](scratch/send_example.py) example to use the updated `asyncio.gather()` syntax instead of the the 
`loop.run_until_complete(run())` syntax. This syntax also needs to be changed in the 
[xbengine.py](katxgpu/xbengine.py)
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
There is currently a [docs](./docs) directory in this repo where some documentation has been stored. This will need
to be incorporated into the formal documentation at some stage.
7. The [unit tests](test/tensorcore_xengine_core_test.py) for the TensorCoreXEngineCore are executed in python and take
a long time to verify. This verification should be moved to C as is done in some of the other unit tests.
8. The [spead2_send_test.py](test/spead2_send_test.py) file has some TODOs that can improve the test coverage. These
should be implemented.
9. The `default_spead_flavour` variable is duplicated in two places ([xsend.py](katxgpu/xsend.py) and
[test/spead2_receiver_test.py](test/spead2_receiver_test.py)) with the potential to be duplicated in more places
if a bsend.py file is created. It may be worth looking at putting all this information in the same place.
10. When the Jenkins CI server runs its unit tests, sometimes a test/spead2_receiver_test.py::test_recv_simple test
will just hang (not error out) forever. This causes the test to never end and the Jenkins server is then unable to
launch more tests due to this stall. This can be fixed by manually restarting the tests but this is far from ideal.
This only occurs about once every ten runs, but when multiple branches are being PR'd and merged, this tends to
kick off many tests. This problem needs to be investigate. I think this is due to the async function here:
https://github.com/ska-sa/katxgpu/blob/6ad82705394052b62065da3cfeac7953f1a45dd7/test/spead2_receiver_test.py#L451-L496 
but I dont know for sure.
11. There are a list of TODOs in the [xbengine](katxgpu/xbengine.py). These should be implemented. The most
pressing of these is the implementation of a clean exit and the addition of control and monitoring.
12. In the [xbengine](katxgpu/xbengine.py) a number of print statements are in place of proper logging
messages. These should be replaced with log messages. In addition, it must be decided what logging and metric measuring
tools must be used. (There was talk on using something like logstash for centralised logging and Prometheus for
managing metrics.)
13. The receiver requires a few different classes to set up correctly. The recv.Stream, recv.Ringbuffer and 
katxgpu.ringbuffer.AsyncRingbuffer objects. See [receiver_example.py](scratch/receiver_example.py) for an example. It
may be worth creating some top level class that encapsulates all of these classes as there is no real value added by
having them seperate and it is a bit confusing to have to work with so many classes to do essentially one function.
14. The command-line parameters in [main.py](katxgpu/main.py) could be made more intuitive, for example instead of
having mcast addresses and port numbers as seperate arguments, accept something formatted as
`<ip address>:<port number>` and parse the argument to seperate out the parameters. Additionally checks need to be put
in place to ensure the command-line parameters are correct - is the port number valid, is the IP address a multicast
address, is the array size >0, etc. As a first step for this, I would look at the
[main.py](https://github.com/ska-sa/katfgpu/blob/master/katfgpu/main.py) file in katfgpu as this uses some useful
parsing functions that could be of use here.
15. There is no B-Engine. It should eventually be implemented.
16. The current Tensor Core kernel is designed to work on the Nvidia RTX 20xx series of GPUs. The newer ranges of
cards (RTX 30xx and above) may not be compatible with this kernel. This needs to be tested as soon as possible on 
a newer card to see if it works. If this does not work, there are a few options. Either the
[tensorcore_xengine_core.py](katxgpu/tensorcore_xengine_core.py) might require some tweaking in which case the changes
are quite well contained. A complication may be that the Tensor Core kernel needs to be changed so much that the input
and output data formats change. In this case, the [precorrelation_reorder.py](katxgpu/precorrelation_reorder.py) may
need to be changed too. The entirety of the `async def _gpu_proc_loop(self)` function in
[xbengine.py](katxgpu/xbengine.py) would then need to be modified. If you begin modifying other
functions in xbengine.py to get the new Tensor Cores working then I suspect you have done something wrong as
only the `_gpu_proc_loop` function launches GPU kernels. Nvidia has some cuBLAS functions that could potentially
perform the operation we want after a bit of reordering 
(see [here](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-syrk) - but you may need to dig deeper into
the cuBLAS options available) - I am just not certain that this uses Tensor Cores under the hood. You will need to
investigate and profile this further.
17. The katxgpu._katxgpu module only exists in the C++ realm. IDEs (and I suspect documentation generators) do not pickup
up these C++ python modules very well. It would be nice if these modules were detected by IDEs. In katfgpu, there is a
solution to this that involves using .pyi files (stub files). The folder with these stub files is
[here](https://github.com/ska-sa/katfgpu/tree/master/katfgpu/_katfgpu). The 
[recv.pyi](https://github.com/ska-sa/katfgpu/blob/master/katfgpu/_katfgpu/recv.pyi) file in this folder defines stubs
of the functions in [py_recv.cpp](https://github.com/ska-sa/katfgpu/blob/master/src/py_recv.cpp). We should do something
similar to this in katxgpu. Addtitionally we should move the comments that were placed in py_recv.cpp to a py_recv.pyi
file in katxgpu/_katxgpu/recv.pyi so that it can get checked by mypy and used in your IDE of choice.
18. Currently to run katxgpu, three arguments are given: `--receiver-thread-affinity, --receiver-comp-vector-affinity`, 
and `--sender-thread-affinity`. It is likely that all of these wil be set to use the same core. It may be worth
creating a new argument in [main.py](katxgpu/main.py) to assign all these values to a specific core. We must then
decide if we want to give the user access to the above three arguments. If this is done, the "Launching the XB-Engine"
section below will need to be updated to reflect this new command.
19. The [verification_functions_lib.c](test/verification_functions_lib.c) has different regions for functions relating
to different unit tests. It may be worth splitting out these regions into different C files. The
[Makefile](test/Makefile) would need to be adjusted to build these different files and the .py unit tests calling these
functions would need to be modified to call the correct .so file. (Unless we combine the different C files into a single
.so file).
20. Currently there is quite a bit of redundancy when creating a docstring for an `__init__()`. Much of the information
already sits in the class docstring. We want to change our commenting style so that the `__init__()` function does not
have a docstring or a minimal one for information that does not need to be said about the class as a whole. The 
"Parameters" comments should also be put in the class docstring. See:
https://github.com/ska-sa/katsdpsigproc/blob/master/katsdpsigproc/accel.py#L1191-L1217 for an example. 

    The following
lines of code must be added to the `.pydocstyle.ini` file to tell pydocstyle to ignore the docstring on the `__init()__`
function in the pre-commit flow (This repo does not have this file - it likely fits in the
[pyproject.toml](pyproject.toml) file):
    ```
    [pydocstyle]
    # D107 tells pydocstyle to ignore __init__ functions. Since we document the class, this is fine.
    ignore = D107
    ```
    This change needs to be applied across all modules in the repo. Dont forget to look at the 
[py_recv.cpp](https://github.com/ska-sa/katfgpu/blob/master/src/py_recv.cpp) file. The longer we wait to apply this
change the more work it will require as more modules are added to the repos, so dont put it off.


## License
The license for this repository still needs to be specified. At the moment this repo is private so its not an issue.
When it eventually goes public, this will need to be specified. We need to check with John Romein (the author of the 
Tensor Core X-Engine core kernel that is central to this repo) what license he is using. This will inform the choice of.
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

## Launching the XB-Engine.

The [main.py](katxgpu/main.py) file launches the entire XB-Engine pipeline. When installing the katxgpu package, the
main.py file is wrapped in a script called `xgpu` that can be called from the command line. The pipeline can be
launched using the following command: 

```
xgpu \
    --receiver-thread-affinity <CPU core index> \
    --receiver-comp-vector-affinity <CPU core index> \
    --src-interface-address <interface IP> \
    --sender-thread-affinity <CPU core index> \
    --dest-interface-address <interface IP> \
    <src mcast address> <src port> \
    <dest mcast address> <dest port>
```

An example with the fields populated is: 
`xgpu --receiver-thread-affinity 0 --receiver-comp-vector-affinity 0 --src-interface-address 10.100.44.1 --sender-thread-affinity 0 --dest-interface-address 10.100.44.1 239.10.10.10 7149 239.10.10.11 7149`

The command above launches the XB-Engine with the minimum number of arguments required to run. The interface with
address 10.100.44.1 is used to send and receive data. Data is received from the 239.10.10.10:7149 address and sent out
on the 239.10.10.11:7149 address. All the different affinities are set to use core 0. This XB-Engine uses the default 
configuration of a 64 antenna, 32 768 channels, L-Band array. Running `xgpu --help` will list all other arguments that
can be used to configure the array.

This pipeline requires three core indices to be specified 
(`--receiver-thread-affinity, --receiver-comp-vector-affinity, --sender-thread-affinity`). It is recommended that these
all be assigned to the same core. The reason for keeping them seperate is to be explicit and in case performance issues
occur.

## Theory of Operation

Documentation describing how the XB-Engine works is under construction. It can currently be found
[here](./docs/THEORY.md).

## Test Framework

The test framework has been implemented using [pytest](https://docs.pytest.org).
To run the framework, run the command `pytest` from the katxgpu parent directory.

This assumes the package installation and pre-commit configuration has already been done.

__NB:__ Some of the tests in pytest verify their output using a function written in the 
[verification_functions_lib.c](test/verification_functions_lib.c) C file in order to reduce the test runtime.
This file needs to be compiled or else the test will throw an error. Navigate to the [test](./test) directory and run
`make` to compile it.

## Jenkins CI

Jenkins has been integrated into this repo. On every PR and at least once a day, Jenkins will scan the repo for changes
and if detected, will run the test pipeline. The test will run on the PR branch or the branch where there are detected 
changes (as long as the branch has a Jenkinsfile). The test configuration is specified in the
[Jenkinsfile](./Jenkinsfile) in this repo. The Jenkins build pipeline will execute this pipeline in a docker container.

In theory, any Jenkins server should be able run the pipeline from the Jenkinsfile, however for this repo, there are a 
few additional requirements:
1. The node the docker server run on requires a Nvidia GPU of compute capability 7.5 or above. Basically the GPU is
required to have Tensor Cores.
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

This repository contains a Dockerfile for building a docker image that can launch katxgpu. As with the Jenkins CI
server, the nvidia-container-runtime needs to be installed when running the image.

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

To launch the entire XB-Engine in a container run the following command:

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
    xgpu \
    --receiver-thread-affinity <CPU core index> \
    --receiver-comp-vector-affinity <CPU core index> \
    --src-interface-address <interface IP> \
    --sender-thread-affinity <CPU core index> \
    --dest-interface-address <interface IP> \
    <src mcast address> <src port> \
    <dest mcast address> <dest port>
```

To view the output from the receiver script run the following command: `watch docker logs -t --tail 10 katxgpu_container`

The `--network host`, `--ulimit=memlock=-1`, `--device=/dev/infiniband/rdma_cm`, `--device=/dev/infiniband/uverbs0`
flags are required to pass the ibverbs devices to the container for high-performance networking.

The `--gpus all` flag passes the GPUs from the host machine into the docker container

__NOTE__: There is quite a bit of overlap between the commands in the [Dockerfile](./Dockerfile) and the
[Jenkinsfile](./Jenkinsfile) and the requirements for running a docker container are the same requirements as mentioned
in the Jenkins CI section above. The explanation for the different arguments required to launch the docker container can
be found in the Jenkinsfile.

## SPEAD2 Network Side Software

This software uses the high-performance SPEAD2 networking library for all high speed data transmission and reception.
The SPEAD2 library has been extended in C++ and this has been turned into a project submodule. This module can be
imported using `import katxgpu._katxgpu`.

The makeup of this module is quite complex. This [file](docs/networking.md) within this repo describes the entire module in
great detail. A simple example of how to use the receiver network code is shown in
[receiver_example.py](scratch/receiver_example.py) in the katxgpu/scratch directory. This example is probably the
quickest way to figure out how the receiver works. Likewise the [send_example.py](scratch/send_example.py) file in the 
katxgpu/scratch folder demonstrates network transmit code works.

The `katxgpu._katxgpu` module uses the SPEAD2 C++ bindings (not the python bindings) and as such requires the SPEAD2
submodule to be included in this repository. This module is located in the katxgpu/3rdparty directory.

The `katxgpu._katxgpu` module only provides functionality for receiving data. The normal SPEAD2 python module is used
for sending X-Engine output data. This is because the X-Engine accumulates data and transmits it at a much lower data 
rate than it is received meaning that the chunking approach is not necessary. The [xsend.py](katxgpu/xsend.py) module
handles the transmission of correlated data. The high level description of this module can also be found
in the same [file](docs/networking.md) that describes the data receiver module.

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
4. Larger Packet Sizes - The current MeerKAT F-Engine output packets are 1 KiB in size. A number of packets make up a
heap with each packet in the heap representing a single channel. The packet size is thus equal to the value set by
the `--samples-per-channel` flag multiplied by 4 (as each sample is a dual pol, complex 8-bit sample, so 4 bytes per
sample). For the 1 KiB packets, `--samples-per-channel` is equal to 256. 1 KiB packets require quite a bit of 
computation to assemble into heaps. By switching to 2 KiB packets, the total CPU processing requirements can be reduced
to 2/3 of the 1 KiB packets. Packet sizes of 4 and 8 KiB also improve on the 2 KiB packet size but the improvement is
not as drastic. By increasing the packet sizes, the less chance there is of your CPU being overloaded and dropping
packets. The [fsim.cpp](scratch/fsim.cpp) and [main.py](katxgpu/main.py) both have a `--samples-per-channel` flag.
Using these two files the thread performance at different packet sizes can be analysed.