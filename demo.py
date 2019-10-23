#!/usr/bin/env python3

import numpy as np

import katfgpu._katfgpu as katfgpu

ring = katfgpu.Receiver.Ringbuffer(2)
recv = [katfgpu.Receiver(i, 4096, 2**25, ring) for i in range(2)]
for pol in range(len(recv)):
    for i in range(4):
        buf = np.empty(2**25 * 10 // 8, np.uint8)
        recv[pol].add_chunk(katfgpu.InChunk(buf))
    recv[pol].add_udp_pcap_file_reader('/mnt/data/bmerry/pcap/dig1s.pcap')

while True:
    try:
        chunk = ring.pop()
    except katfgpu.Stopped:
        break
    print(f'Received chunk: timestamp={chunk.timestamp} pol={chunk.pol}')
    recv[chunk.pol].add_chunk(chunk)
print('Done!')
