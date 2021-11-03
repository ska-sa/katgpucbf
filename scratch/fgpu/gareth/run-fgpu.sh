#!/bin/bash
set -e -u

# This NIC seems to be dropping a lot of incoming packets
# (maybe contending with GPU on the same host bridge?)
# iface1=enp129s0f0np0
# iface2=enp129s0f1np1

iface1=enp130s0f0
iface2=enp130s0f1
cuda1=0
cuda2=0

src_affinity="$((6*$1)),$((6*$1+1))"
src_comp=$src_affinity
dst_affinity="$((6*$1+3))"
dst_comp=$dst_affinity
other_affinity="$((6*$1+4))"
srcs="239.102.$1.64+7:7148 239.102.$1.72+7:7148"
dst="239.102.$((200+$1)).0+15:7148"
port="$(($1+7140))"

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
exec spead2_net_raw taskset -c $other_affinity fgpu \
    --src-interface $iface --src-ibv \
    --dst-interface $iface --dst-ibv \
    --src-affinity $src_affinity --src-comp-vector=$src_comp \
    --dst-affinity $dst_affinity --dst-comp-vector=$dst_comp \
    --adc-sample-rate ${adc_sample_rate:-1712000000} \
    --channels ${channels:-32768} \
    --spectra-per-heap ${spectra_per_heap:-256} \
    --quant-gain 0.0001 \
    --katcp-port $port \
    --sync-epoch 0 \
    $srcs $dst
