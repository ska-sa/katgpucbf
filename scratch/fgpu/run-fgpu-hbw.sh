#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

recv_affinity="0,1,4,5"
recv_comp=$recv_affinity
send_affinity="8"
send_comp=$send_affinity
other_affinity="12"
src="239.102.0.64+15:7148"
dst="239.102.200.0+63:7148"
nb_dst="239.102.216.0+7:7148"
katcp_port="7140"
prom_port="7150"
feng_id="0"

export CUDA_VISIBLE_DEVICES="$cuda1"

set -x
exec spead2_net_raw taskset -c $other_affinity fgpu \
    --recv-chunk-samples 134217728 \
    --send-chunk-jones 33554432 \
    --recv-buffer=268435456 \
    --recv-interface $iface1,$iface2 --recv-ibv \
    --send-interface $iface1,$iface2 --send-ibv \
    --recv-affinity $recv_affinity --recv-comp-vector=$recv_comp \
    --send-affinity $send_affinity --send-comp-vector=$send_comp \
    --adc-sample-rate ${adc_sample_rate:-7000000000} \
    --jones-per-batch ${jones_per_batch:-1048576} \
    --katcp-port $katcp_port \
    --prometheus-port $prom_port \
    --sync-time 0 \
    --array-size ${array_size:-8} \
    --feng-id "$feng_id" \
    --wideband "name=wideband,channels=${channels:-32768},dst=$dst" \
    --narrowband "name=narrow0,channels=${nb_channels:-32768},decimation=${nb_decimation:-8},centre_frequency=284e6,${nb_ddc_taps:+ddc_taps=$nb_ddc_taps,}dst=$nb_dst" \
    "$src" "$@"
