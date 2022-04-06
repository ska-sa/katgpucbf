#!/bin/bash

set -e -u

# Load variables for machine-specific config
. config/$(hostname -s).sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 0|1|2|3" 1>&2
    exit 2
fi

katcp_port="$(($1+7140))"
prom_port="$(($1+7150))"

case "$1" in
    0|1)
        iface="$iface1"
        ;;
    2|3)
        iface="$iface2"
        ;;
    *)
        echo "Pass an integer from 0-3" 1>&2
        exit 2
        ;;
esac

nproc="$(nproc)"
cpu1="$(($nproc * $1 / 4))"
cpu2="$(($cpu1 + 1))"
addresses="239.102.$1.64+7:7148 239.102.$1.72+7:7148"

set -x
exec spead2_net_raw taskset -c $cpu2 dsim \
    --affinity $cpu1 \
    --ibv \
    --interface $iface \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --ttl 2 \
    --katcp-port $katcp_port \
    --prometheus-port $prom_port \
    $addresses
