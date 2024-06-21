#!/bin/bash

set -e -u

# Load variables for machine-specific config
. ../config/$(hostname -s).sh

channels=${channels:-32768}
channels_per_substream=${channels_per_substream:-512}
int_time=${int_time:-0.5}
adc_sample_rate=${adc_sample_rate:-1712000000.0}
jones_per_batch=${jones_per_batch:-1048576}
spectra_per_batch=$((jones_per_batch / channels))
samples_between_spectra=${samples_between_spectra:-$((channels*2))}
heap_accumulation_threshold=$(python -c "print(round($int_time * $adc_sample_rate / $samples_between_spectra / $spectra_per_batch))")

index="$1"
nproc=$(nproc)
step=$((nproc / 4))
affinity="$((index * step))"
rx_affinity=$affinity
rx_comp=$rx_affinity
tx_affinity=$((affinity + 1))
tx_comp=$tx_affinity
other_affinity=$tx_affinity
src="239.10.10.$((10 + index)):7148"
dst="239.10.11.$((10 + index)):7148"
channel_offset=$((channels_per_substream * index))
katcp_port="$((7140 + index))"
prom_port="$((7150 + index))"

declare -a beam_args
beams=${beams:-4}
i=0
while [ "$i" -lt "$beams" ]; do
    beam_args+=("--beam=name=beam_${i}x,pol=0,dst=239.10.$((12 + index)).$((i * 2)):7148")
    beam_args+=("--beam=name=beam_${i}y,pol=1,dst=239.10.$((12 + index)).$((i * 2 + 1)):7148")
    i="$((i + 1))"
done

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
    --recv-affinity $rx_affinity \
    --recv-comp-vector $rx_comp \
    --send-affinity $tx_affinity \
    --send-comp-vector $tx_comp \
    --recv-interface $iface \
    --send-interface $iface \
    --recv-ibv --send-ibv \
    --corrprod=name=bcp1,heap_accumulation_threshold=${heap_accumulation_threshold},dst=$dst \
    "${beam_args[@]}" \
    --adc-sample-rate ${adc_sample_rate} \
    --array-size ${array_size:-64} \
    --jones-per-batch ${jones_per_batch} \
    --heaps-per-fengine-per-chunk ${heaps_per_fengine_per_chunk:-32} \
    --channels $channels \
    --channels-per-substream $channels_per_substream \
    --samples-between-spectra $samples_between_spectra \
    --channel-offset-value $channel_offset \
    --sync-time 0 \
    --katcp-port $katcp_port \
    --prometheus-port $prom_port \
    --send-enabled \
    $src
