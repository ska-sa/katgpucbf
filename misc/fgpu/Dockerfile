FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-build as build

USER root
RUN apt-get -y update && apt-get --no-install-recommends -y install \
        libpcap-dev
USER kat

ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

COPY --chown=kat:kat . /tmp/install/katfgpu
RUN pip install --no-deps /tmp/install/katfgpu
RUN pip check

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-runtime

USER root
RUN apt-get -y update && apt-get --no-install-recommends -y install \
        libpcap0.8
USER kat

COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
