#!/bin/bash

set -e -u

iface1=enp130s0f0
iface2=enp130s0f1
channels_per_substream=${channels_per_substream:-512}

affinity="$(($1 * 6))"
rx_affinity=$affinity
rx_comp=$rx_affinity
tx_affinity=$(($affinity + 1))
other_affinity=$tx_affinity
src_mcast="239.10.10.$((10 + $1)):7148"
dst_mcast="239.10.11.$((10 + $1)):7148"
port="$((7150 + $1))"
channel_offset=$(($channels_per_substream * $1))

case "$1" in
    0|1)
        iface="$iface1"
        export CUDA_VISIBLE_DEVICES=0
        ;;
    2|3)
        iface="$iface2"
        export CUDA_VISIBLE_DEVICES=0
        ;;
    *)
        echo "Pass 0-3" 1>&2
        exit 2
        ;;
esac

set -x

exec spead2_net_raw numactl -C $other_affinity xbgpu \
    --src-affinity $rx_affinity \
    --src-comp-vector $rx_comp \
    --dst-affinity $tx_affinity \
    --dst-interface $iface \
    --src-interface $iface \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --array-size ${array_size:-64} \
    --spectra-per-heap ${spectra_per_heap:-256} \
    --channels ${channels:-32768} \
    --channels-per-substream $channels_per_substream \
    --channel-offset-value $channel_offset \
    --heap-accumulation-threshold ${heap_accumulation_threshold:-52} \
    --katcp-port $port \
    $src_mcast $dst_mcast
