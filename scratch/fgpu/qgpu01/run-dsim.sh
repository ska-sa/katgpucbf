#!/bin/bash

set -e -u

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 0|1|2|3" 1>&2
    exit 2
fi

case "$1" in
    0|1|2|3)
        ;;
    *)
        echo "Pass an integer from 0-3" 1>&2
        exit 2
        ;;
esac

cpu="$1"
addresses="239.102.$1.64+7:7148 239.102.$1.72+7:7148"
ip="10.100.$((41+2*($1%2))).1"
exec cap_net_raw numactl -C $cpu ../../../src/tools/dsim --ibv --interface $ip --adc-sample-rate 1712000000 --ttl 2 $addresses
