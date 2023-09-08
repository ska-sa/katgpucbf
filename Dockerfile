# syntax = docker/dockerfile:1

################################################################################
# Copyright (c) 2021-2023, National Research Foundation (SARAO)
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

# Use the nvidia development image as a base. This gives access to all
# nvidia and cuda runtime and development tools. pycuda needs nvcc, so
# the development tools are necessary.

FROM nvidia/cuda:12.0.1-base-ubuntu22.04 as base

# This "base" layer is modified to better support running with Vulkan. That's
# needed by both build-base (used by Jenkins to run unit tests) and the final
# image. Additionally, for the Vulkan drivers to work one needs
# libvulkan1, libegl1 and libxext6.
#
# Some development packages are also installed that are needed for pycuda,
# as well as libcufft, needed for fgpu.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# See also https://github.com/NVIDIA/nvidia-container-toolkit/issues/16
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-0 \
    cuda-profiler-api-12-0 \
    libcurand-dev-12-0 \
    libcufft-12-0 \
    libvulkan1 \
    libegl1 \
    libxext6

FROM base as build-base

# Install system packages:
# - git is needed for setuptools_scm
# - wget is used to download spead2
# - pkg-config and lib* are needed for spead2
# DEBIAN_FRONTEND=noninteractive prevents apt-get from asking configuration
# questions.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    git \
    pkg-config \
    libboost-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libibverbs-dev \
    librdmacm-dev \
    libpcap-dev \
    libcap-dev \
    libdivide-dev \
    libvulkan-dev \
    wget

# Create a virtual environment
RUN python -m venv /venv
# Activate it
ENV PATH=/venv/bin:$PATH
# Install up-to-date versions of installation tools, for the benefits of
# packages not using PEP 517/518.
RUN pip install pip==22.3.1 setuptools==65.6.3 wheel==0.38.4 meson==1.2.1

# Install spead2 C++ bindings. We use requirements.txt just to get the
# version, so that when we want to update we only have to do it in one place.
WORKDIR /tmp/katgpucbf
COPY requirements.txt .
WORKDIR /tmp
RUN SPEAD2_VERSION=$(grep ^spead2== katgpucbf/requirements.txt | cut -d= -f3) && \
    wget "https://github.com/ska-sa/spead2/releases/download/v$SPEAD2_VERSION/spead2-$SPEAD2_VERSION.tar.gz" && \
    tar -zxf "spead2-$SPEAD2_VERSION.tar.gz" && \
    cd "spead2-$SPEAD2_VERSION" && \
    meson setup build && \
    cd build && \
    meson compile && \
    meson install

# Install and immediately uninstall pycuda. This causes pip to cache the
# wheel it built, making it fast to install later (we uninstall so that the
# Jenkins image has a clean environment to start from).
RUN pip install --no-deps "$(grep '^pycuda==' /tmp/katgpucbf/requirements.txt)" && \
    pip uninstall -y pycuda

#######################################################################

# Image used by Jenkins
FROM build-base as jenkins

# docker so that Jenkins can build a Docker image
# All the TeX stuff for building the documentation and qualification report
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    docker.io \
    latexmk \
    lmodern \
    pdf2svg \
    tex-gyre \
    texlive-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-science

#######################################################################

# The above image is independent of the contents of this package (except
# for requirements.txt), and is used to form the image for Jenkins
# testing. We now install the package in a new build stage.

FROM build-base as build-py

# Install requirements (already copied to build-base image).
WORKDIR /tmp/katgpucbf
RUN pip install -r requirements.txt

# Install the package itself. Using --no-deps ensures that if there are
# requirements that aren't pinned in requirements.txt, the subsequent
# pip check will complain.
COPY . .
RUN pip install --no-deps . && pip check

#######################################################################

# Separate stage to build the C++ tools. This is in a separate build stage
# so that changes to either the C++ code or the Python code don't invalidate
# the build cache for the other.

FROM build-base as build-cxx

# Build utilities.
# We use make clean to ensure that an existing build from the build context
# won't accidentally get used instead.
WORKDIR /tmp/tools
COPY src/tools .
RUN make clean && make -j fsim

RUN wget https://raw.githubusercontent.com/ska-sa/katsdpdockerbase/master/docker-base-runtime/schedrr.c && \
    gcc -Wall -O2 -o schedrr schedrr.c

#######################################################################

# The above builds everything. Now install it into a lighter-weight runtime
# image, without all the stuff needed for the build itself.
FROM base
LABEL maintainer="MeerKAT CBF team <cbf@ska.ac.za>"

# curl is needed for running under katsdpcontroller
# numactl allows CPU and memory affinity to be controlled.
# libboost-program-options is for spead2's C++ command-line tools - not
# strictly needed but useful for debugging.
# netbase provides /etc/protocols, which libpcap depends on in some cases.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    curl \
    numactl \
    libboost-program-options1.74.0 \
    libboost-system1.74.0 \
    libibverbs1 \
    librdmacm1 \
    ibverbs-providers \
    libpcap0.8 \
    libcap2 \
    libcap2-bin \
    netbase

COPY --from=build-py /venv /venv
COPY --from=build-cxx /tmp/tools/fsim /usr/local/bin
COPY --from=build-cxx /tmp/tools/schedrr /usr/local/bin
COPY docker/tuning.db /root/.cache/katsdpsigproc/tuning.db
RUN setcap cap_sys_nice+ep /usr/local/bin/schedrr
ENV PATH=/venv/bin:$PATH KATSDPSIGPROC_TUNE_MATCH=nearest
