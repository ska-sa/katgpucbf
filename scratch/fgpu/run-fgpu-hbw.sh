#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

src_affinity="0,4"
src_comp=$src_affinity
dst_affinity="8"
dst_comp=$dst_affinity
other_affinity="12"
srcs="239.102.0.64+7:7148 239.102.0.72+7:7148"
dst="239.102.200.0+15:7148"
katcp_port="7140"
prom_port="7150"
feng_id="0"

export CUDA_VISIBLE_DEVICES="$cuda1"

set -x
exec spead2_net_raw taskset -c $other_affinity fgpu \
    --src-chunk-samples 67108864 \
    --dst-chunk-jones 33554432 \
    --src-interface $iface1,$iface2 --src-ibv \
    --dst-interface $iface1,$iface2 --dst-ibv \
    --src-affinity $src_affinity --src-comp-vector=$src_comp \
    --dst-affinity $dst_affinity --dst-comp-vector=$dst_comp \
    --adc-sample-rate ${adc_sample_rate:-7000000000} \
    --spectra-per-heap ${spectra_per_heap:-256} \
    --katcp-port $katcp_port \
    --prometheus-port $prom_port \
    --sync-epoch 0 \
    --array-size ${array_size:-8} \
    --feng-id "$feng_id" \
    --wideband "name=wideband,channels=${channels:-32768},dst=$dst" \
    $srcs "$@"
