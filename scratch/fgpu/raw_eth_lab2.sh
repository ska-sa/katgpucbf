#!/bin/bash
set -e -u
dev=$(ibdev2netdev | grep " ==> $iface " | cut -d' ' -f1)
numactl -N0 raw_ethernet_bw -d $dev --client -m 5120 -B $mac -E 01:00:5e:65:c8:00 -j $ip -J 239.101.200.0 -K 7148 -k 8888 --rate_limit=68 --rate_limit_type=SW -D 10000
