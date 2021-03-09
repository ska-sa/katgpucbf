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
FROM nvidia/cuda:10.1-devel-ubuntu18.04

# 2. Install all packages that are independent of katxgpu. This will ensure
# that these commands are not re-run by docker when SPEAD2 is updated.
RUN apt-get update

# 2.1 Install all packages required for running python3.
RUN apt-get install -y \
    python3.6 \
    python3-pip \
    python-pybind11 \
    python3.6-dev \
    git

# 2.2 Install all packages required for using SPEAD2.
RUN apt-get install -y \
    autoconf \
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
WORKDIR katxgpu

# Copy entire docker build context into katxgpu folder with the container.
COPY . . 
RUN pip3 install -r requirements.txt

# NOTE: These packages do not need to be installed. I currently use it when I
# have to open the container and verify that things are running by running
# pytest. The line below can be safetly removed if needed.
RUN pip3 install -r requirements-dev.txt 

# Hack to force python3 use by default. Needed as the install of SPEAD2 calls 
# `python` while expecting it to be `python3`. This is probably not advisable
# in most situations.
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python 

RUN pip3 install -e .