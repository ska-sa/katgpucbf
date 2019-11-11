#!/bin/bash
set -e -u
numactl -C8 raw_ethernet_bw -B $mac -E 01:00:5e:66:c8:64 -J 239.102.200.100  -j $ip -K 7148 -k 8888  -P -m 8192 --rate_limit=54 --rate_limit_type=SW -D 10000
