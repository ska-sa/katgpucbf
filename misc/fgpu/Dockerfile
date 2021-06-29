FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-build as build

USER root
RUN apt-get -y update && apt-get --no-install-recommends -y install \
        libpcap-dev
USER kat

ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN install_pinned.py -r /tmp/install/requirements.txt

COPY --chown=kat:kat . /tmp/install/katfgpu
RUN pip install --no-deps /tmp/install/katfgpu
RUN pip check

# Compile and install spead2 so that dsim can be built
WORKDIR /tmp/install/katfgpu/3rdparty/spead2
# The bootstrap script needs these Python modules to work.
RUN pip install jinja2==2.10.1 pycparser==2.19 && ./bootstrap.sh && ./configure && make -j
USER root
RUN make install
# Compile dsim
USER kat
WORKDIR /tmp/install/katfgpu/scratch
RUN make clean && make -j dsim

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-runtime

USER root
RUN apt-get -y update && apt-get --no-install-recommends -y install \
        libpcap0.8
USER kat

COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
# Install dsim binary
COPY --from=build --chown=root:root /tmp/install/katfgpu/scratch/dsim /usr/local/bin/dsim
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
