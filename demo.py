#!/usr/bin/env python3

import numpy as np
import katsdpsigproc.accel as accel

import katfgpu._katfgpu as katfgpu


SAMPLE_BITS = 10
CHUNK_SAMPLES = 2**28
PACKET_SAMPLES = 4096
CHUNK_BYTES = CHUNK_SAMPLES * 10 // 8

ctx = accel.create_some_context()
queue = ctx.create_command_queue()

ring = katfgpu.Receiver.Ringbuffer(2)
recv = [katfgpu.Receiver(i, SAMPLE_BITS, PACKET_SAMPLES, CHUNK_SAMPLES, ring) for i in range(2)]
dev_samples = accel.DeviceArray(ctx, (recv[0].chunk_bytes,), np.uint8)
for pol in range(len(recv)):
    for i in range(4):
        buf = accel.HostArray((recv[pol].chunk_bytes,), np.uint8, context=ctx)
        recv[pol].add_chunk(katfgpu.InChunk(buf))
    recv[pol].add_udp_pcap_file_reader('/mnt/data/bmerry/pcap/dig1s.pcap')

lost = 0
while True:
    try:
        chunk = ring.pop()
    except katfgpu.Stopped:
        break
    total = len(chunk.present)
    good = sum(chunk.present)
    lost += total - good
    print('Received chunk: timestamp={chunk.timestamp} pol={chunk.pol} ({good}/{total}, lost {lost})'.format(
        chunk=chunk, good=good, total=total, lost=lost))
    try:
        buf = chunk.base
        dev_samples.set(queue, buf)
    finally:
        recv[chunk.pol].add_chunk(chunk)
print('Done!')
