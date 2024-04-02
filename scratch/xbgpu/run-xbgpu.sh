#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

channels=${channels:-32768}
channels_per_substream=${channels_per_substream:-512}
int_time=${int_time:-0.5}
adc_sample_rate=${adc_sample_rate:-1712000000.0}
spectra_per_heap=${spectra_per_heap:-256}
samples_between_spectra=${samples_between_spectra:-$((channels*2))}
heap_accumulation_threshold=$(python -c "print(round($int_time * $adc_sample_rate / $samples_between_spectra / $spectra_per_heap))")

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
channel_offset=$(($channels_per_substream * $1))
katcp_port="$((7140 + $1))"
prom_port="$((7150 + $1))"

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

if ! command -v schedrr > /dev/null; then
    cat 1>&2 <<'EOF'
schedrr not found.
- Download it from https://raw.githubusercontent.com/ska-sa/katsdpdockerbase/master/docker-base-runtime/schedrr.c
- Compile it with gcc -o schedrr schedrr.c -Wall -O2
- Run `sudo setcap cap_sys_nice+ep schedrr`
- Add it to your $PATH
EOF
    exit 1
fi

set -x

exec schedrr spead2_net_raw numactl -C $other_affinity xbgpu \
    --src-affinity $rx_affinity \
    --src-comp-vector $rx_comp \
    --dst-affinity $tx_affinity \
    --dst-comp-vector $tx_comp \
    --src-interface $iface \
    --dst-interface $iface \
    --src-ibv --dst-ibv \
    --corrprod=name=bcp1,heap_accumulation_threshold=${heap_accumulation_threshold},dst=$dst_mcast \
    --beam=name=beam_0x,pol=0,dst=239.10.12.$((0 + $1 * 8)):7148 \
    --beam=name=beam_0y,pol=1,dst=239.10.12.$((1 + $1 * 8)):7148 \
    --beam=name=beam_1x,pol=0,dst=239.10.12.$((2 + $1 * 8)):7148 \
    --beam=name=beam_1y,pol=1,dst=239.10.12.$((3 + $1 * 8)):7148 \
    --beam=name=beam_2x,pol=0,dst=239.10.12.$((4 + $1 * 8)):7148 \
    --beam=name=beam_2y,pol=1,dst=239.10.12.$((5 + $1 * 8)):7148 \
    --beam=name=beam_3x,pol=0,dst=239.10.12.$((6 + $1 * 8)):7148 \
    --beam=name=beam_3y,pol=1,dst=239.10.12.$((7 + $1 * 8)):7148 \
    --adc-sample-rate ${adc_sample_rate} \
    --array-size ${array_size:-64} \
    --spectra-per-heap ${spectra_per_heap} \
    --heaps-per-fengine-per-chunk ${heaps_per_fengine_per_chunk:-5} \
    --channels $channels \
    --channels-per-substream $channels_per_substream \
    --samples-between-spectra $samples_between_spectra \
    --channel-offset-value $channel_offset \
    --sync-epoch 0 \
    --katcp-port $katcp_port \
    --prometheus-port $prom_port \
    --tx-enabled \
    $src_mcast
