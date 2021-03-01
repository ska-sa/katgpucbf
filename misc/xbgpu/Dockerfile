#Comment on flags to pass into this container
#sudo docker run --gpus all --network host --ulimit=memlock=-1 --device=/dev/infiniband/rdma_cm  --device=/dev/infiniband/uverbs* -it katxgpu 
#sudo docker image build -t katxgpu .

FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3.6 python3-pip python-pybind11 python3.6-dev git # Required for python
RUN apt-get install -y autoconf libboost-all-dev libibverbs-dev librdmacm-dev libpcap-dev #Required for installing SPEAD2.

WORKDIR katxgpu

# Copy entire docker build context into katxgpu folder with the container.
COPY . . 

RUN pip3 install numpy==1.19.5 pycparser jinja2 pybind11
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements-dev.txt # Does not need to be installed - using pytest for quick verification
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python # Hack to force python 3 use. Not advisable in most situations.

RUN pip3 install -e .
RUN pip3 install -e . # Needs to be duplicated for some unknown reason. Working on eliminating this
