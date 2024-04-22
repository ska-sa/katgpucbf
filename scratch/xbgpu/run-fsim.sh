#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

mcast="239.10.10.$((10 + $1 * 2))+1:7148"

case "$1" in
    0)
        iface=$iface1
        ;;
    1)
        iface=$iface2
        ;;
    *)
        echo "Pass 0-1" 1>&2
        exit 2
        ;;
esac

exec spead2_net_raw fsim \
    --interface $iface \
    --ibv \
    --array-size ${array_size:-80} \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --channels ${channels:-32768} \
    --channels-per-substream ${channels_per_substream:-512} \
    --jones-per-batch ${jones_per_batch:-1048576} \
    $mcast
