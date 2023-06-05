#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

channels=${channels:-32768}
channels_per_substream=${channels_per_substream:-512}

nproc=$(nproc)
step=$(($nproc / 4))
affinity="$(($1 * step))"
rx_affinity=$affinity
rx_comp=$rx_affinity
tx_affinity=$(($affinity + 1))
tx_comp=$tx_affinity
other_affinity=$tx_affinity
src_mcast="239.10.10.$((10 + $1)):7148"
dst_mcast="239.10.11.$((10 + $1)):7148"
port="$((7150 + $1))"
channel_offset=$(($channels_per_substream * $1))

case "$1" in
    0|1)
        iface="$iface1"
        export CUDA_VISIBLE_DEVICES="$cuda1"
        ;;
    2|3)
        iface="$iface2"
        export CUDA_VISIBLE_DEVICES="$cuda2"
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
    --dst-comp-vector $tx_comp \
    --src-interface $iface \
    --dst-interface $iface \
    --src-ibv --dst-ibv \
    --corrprod=name=bcp1,heap_accumulation_threshold=${heap_accumulation_threshold:-52},dst=$dst_mcast \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --array-size ${array_size:-64} \
    --spectra-per-heap ${spectra_per_heap:-256} \
    --channels $channels \
    --channels-per-substream $channels_per_substream \
    --samples-between-spectra ${samples_between_spectra:-$((channels*2))} \
    --channel-offset-value $channel_offset \
    --sync-epoch 0 \
    --katcp-port $port \
    --tx-enabled \
    $src_mcast
