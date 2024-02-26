#!/bin/bash

set -e -u

function interface_ipv4_addr()
{
    ip -4 -o addr show $1 | awk '/inet /{split($4, a, "/"); print a[1]}'
}

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

iface_ip1=$(interface_ipv4_addr $iface1)
iface_ip2=$(interface_ipv4_addr $iface2)
mcast="239.10.10.$((10 + $1 * 2))+1:7148"

case "$1" in
    0)
        iface_ip=$iface_ip1
        ;;
    1)
        iface_ip=$iface_ip2
        ;;
    *)
        echo "Pass 0-1" 1>&2
        exit 2
        ;;
esac

exec spead2_net_raw ../../src/tools/fsim \
    --interface $iface_ip \
    --ibv \
    --array-size ${array_size:-64} \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --channels ${channels:-32768} \
    --channels-per-substream ${channels_per_substream:-512} \
    --spectra-per-heap ${spectra_per_heap:-256} \
    $mcast
