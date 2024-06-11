#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

nodes=$(lscpu | grep 'NUMA node.*CPU' | wc -l)
nproc=$(nproc)
step=$(($nproc / $nodes))
hstep=$(($step / 2))
src_idx=$(($1 % 4))

src_affinity="$(($step*$1))"
src_comp=$src_affinity
dst_affinity="$(($step*$1+$hstep))"
dst_comp=$dst_affinity
other_affinity="$(($step*$1+$hstep+1))"
src="239.102.$src_idx.64+15:7148"
dst="239.102.$((200+$1)).0+63:7148"
nb_dst="239.102.$((216+$1)).0+7:7148"
katcp_port="$(($1+7140))"
prom_port="$(($1+7150))"
feng_id="$1"

case "$1" in
    0|1)
        iface="$iface1"
        export CUDA_VISIBLE_DEVICES="$cuda1"
        ;;
    2|3)
        iface="$iface2"
        export CUDA_VISIBLE_DEVICES="$cuda2"
        ;;
    4|5)
        iface="$iface3"
        export CUDA_VISIBLE_DEVICES="$cuda3"
        ;;
    6|7)
        iface="$iface4"
        export CUDA_VISIBLE_DEVICES="$cuda4"
        ;;
    *)
        echo "Pass 0-7" 1>&2
        exit 2
        ;;
esac
shift

set -x
exec spead2_net_raw taskset -c $other_affinity fgpu \
    --src-interface $iface --src-ibv \
    --dst-interface $iface --dst-ibv \
    --src-affinity $src_affinity --src-comp-vector=$src_comp \
    --dst-affinity $dst_affinity --dst-comp-vector=$dst_comp \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --jones-per-batch ${jones_per_batch:-1048576} \
    --katcp-port $katcp_port \
    --prometheus-port $prom_port \
    --sync-time 0 \
    --array-size ${array_size:-8} \
    --feng-id "$feng_id" \
    --wideband "name=wideband,channels=${channels:-32768},dst=$dst" \
    --narrowband "name=narrow0,channels=${nb_channels:-32768},decimation=${nb_decimation:-8},centre_frequency=284e6,${nb_ddc_taps:+ddc_taps=$nb_ddc_taps,}dst=$nb_dst" \
    "$src" "$@"
