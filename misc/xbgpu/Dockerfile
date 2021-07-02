# This file builds a docker container that can run the katxgpu software.
#
# In order to build the following command must be run from the katxgpu 
# directory: `sudo docker image build -t katxgpu .`
#    
# In order to run this container the following command needs to be run:
# `sudo docker run --gpus all --network host --ulimit=memlock=-1 
#    --device=/dev/infiniband/rdma_cm  --device=/dev/infiniband/uverbs0 
#    -it katxgpu `
# See the Jenkinsfile or README.md file in this repo for an explanation of the 
# different flags above. All these flags are required. Additionally the user
# within the container must be root for ibverbs programs to run.
#
# TODO: Look at best practices for the "COPY . ." command. It seems a bit clunky
# at the moment. 

# 1. Use the nvidia development image as a base. This gives access to all 
# nvidia and cuda runtime and development tools.
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 2. Install all packages that are independent of katxgpu. This will ensure
# that these commands are not re-run by docker when SPEAD2 is updated.

# 2.1 Install all packages required for running python3
# (curl is needed for launching under katsdpcontroller).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 \
    python3-pip \
    python3-wheel \
    python3.8-dev \
    python-is-python3 \
    git \
    curl

# 2.2 Install all packages required for using SPEAD2.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    autoconf \
    pkg-config \
    libboost-all-dev \
    libibverbs-dev \
    librdmacm-dev \
    libpcap-dev

# 2.3 Install all python modules required for compiling katxgpu.
# NOTE: numpy is installed here as if it is not already installed the wrong
# version of pycuda gets installed when running the 
# `pip install -r requirements.txt` command below. If the required version of
# numpy changes, this command will need to be updated.
RUN pip3 install numpy==1.19.5 pycparser jinja2 pybind11

# 3. Install katxgpu including all python packages specified in the requirements
# files.
WORKDIR /katxgpu

# Install requirements.
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# NOTE: These packages do not need to be installed. I currently use it when I
# have to open the container and verify that things are running by running
# pytest. The line below can be safetly removed if needed.
COPY requirements-dev.txt .
RUN pip3 install -r requirements-dev.txt 

# Copy entire docker build context into katxgpu folder with the container.
COPY . .

RUN pip3 install .

# Build the Fsim in case someone wants to use it.
WORKDIR /katxgpu/3rdparty/spead2
RUN ./bootstrap.sh
RUN ./configure
RUN make
RUN make install

WORKDIR /katxgpu/scratch
RUN make

# Change the workdir back
WORKDIR /katxgpu
