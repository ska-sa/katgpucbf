#!/bin/bash

set -e -u

iface_ip1=10.100.43.1
iface_ip2=10.100.41.1
mcast="239.10.10.$((10 + $1 * 2))+1:7148"

case "$1" in
    0|1)
        iface_ip=$iface_ip1
        ;;
    2|3)
        iface_ip=$iface_ip2
        ;;
    *)
        echo "Pass 0-3" 1>&2
        exit 2
        ;;
esac

exec spead2_net_raw ../../../src/tools/fsim --interface $iface_ip --channels-per-substream 256 $mcast
