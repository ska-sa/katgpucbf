#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

nproc="$(nproc)"
cpu=("0" "$(($nproc / 4))")
iface=("$iface1" "$iface2")
addresses=("239.102.0.64+7:7148" "239.102.0.72+7:7148")
sync_time="$(date +%s)"

set -x
for i in 0 1; do
    sudo `which spead2_net_raw` taskset -c $((cpu[$i] + 1)) `which dsim` \
        --affinity "${cpu[$i]}" \
        --ibv \
        --interface "${iface[$i]}" \
        --adc-sample-rate ${adc_sample_rate:-7000000000} \
        --ttl 2 \
        --katcp-port $(($i + 7140)) \
        --prometheus-port $(($i + 7150)) \
        --sync-time "$sync_time" \
        --first-id $i \
        "${addresses[$i]}" "$@" &
done
wait
