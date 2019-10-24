#!/usr/bin/env python3

import numpy as np
import katsdpsigproc.accel as accel

import katfgpu._katfgpu as katfgpu


CHUNK_SAMPLES = 2**25
CHUNK_BYTES = CHUNK_SAMPLES * 10 // 8

ctx = accel.create_some_context()
queue = ctx.create_command_queue()
dev_samples = accel.DeviceArray(ctx, (CHUNK_BYTES,), np.uint8)

ring = katfgpu.Receiver.Ringbuffer(2)
recv = [katfgpu.Receiver(i, 4096, CHUNK_SAMPLES, ring) for i in range(2)]
for pol in range(len(recv)):
    for i in range(4):
        buf = accel.HostArray((CHUNK_BYTES,), np.uint8, context=ctx)
        recv[pol].add_chunk(katfgpu.InChunk(buf))
    recv[pol].add_udp_pcap_file_reader('/mnt/data/bmerry/pcap/dig1s.pcap')

while True:
    try:
        chunk = ring.pop()
    except katfgpu.Stopped:
        break
    print(f'Received chunk: timestamp={chunk.timestamp} pol={chunk.pol}')
    try:
        buf = chunk.base
        dev_samples.set(queue, buf)
    finally:
        recv[chunk.pol].add_chunk(chunk)
print('Done!')
