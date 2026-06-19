#! /usr/bin/env python3

# Analyze a VLBI dump file and print the number of framesets and frames in each frameset.

import io
import struct

from baseband.vdif import VDIFFrame, VDIFFrameSet


def main() -> None:
    """Main function."""
    vtp_packets = dict[int, VDIFFrame]()
    packet_count = 0
    for size in range(5000, 65000):
        with open("vlbidump", "rb") as f:
            data = f.read(3 * size)
            s0, s1, s2 = (struct.unpack("<Q", data[i : i + 8])[0] for i in (0, size, 2 * size))
            if s1 == s0 + 1 and s2 == s1 + 1:
                print(f"likely packet size: {size}")
                break
    with open("vlbidump", "rb") as f:
        while True:
            packet = f.read(5040)
            packet_count += 1
            if not packet:
                break
            new_seq_id = struct.unpack("<Q", packet[:8])[0]
            try:
                frame = VDIFFrame.fromfile(io.BytesIO(packet[8:]))
                if vtp_packets.get(new_seq_id) is not None:
                    print("duplicate seq id")
                vtp_packets[new_seq_id] = frame
            except Exception:
                print(f"Error decoding frame {new_seq_id}")

    print(f"total packets: {packet_count}")
    seq_ids = sorted(vtp_packets.keys())
    prev_frames = []
    framesets = []
    print(f"first seq_id{seq_ids[0]}")
    print(f"last seq_id{seq_ids[-1]}")
    print(f"total expected seq_id{seq_ids[-1] - seq_ids[0]}")
    packet_count = 0
    for seq_id in seq_ids:
        packet_count += 1
        if vtp_packets[seq_id].header["frame_nr"] == 0:
            if len(prev_frames) != 0:
                frameset = VDIFFrameSet(prev_frames, prev_frames[0].header)
                framesets.append(frameset)
                prev_frames.clear()
        else:
            prev_frames.append(vtp_packets[seq_id])
    print(f"Number of framesets: {len(framesets)}")
    print(f"Number of packets: {packet_count}")
    #    print(f"data duration: {framesets[-1].get_time().to_value("unix") - framesets[0].get_time().to_value("unix")}")

    prev_seq_id = seq_ids[0]
    total_missing_frames = 0
    for seq_id in seq_ids[1:]:
        if seq_id != prev_seq_id + 1:
            print(f"Missing sequence IDs between {prev_seq_id} and {seq_id}")
            total_missing_frames += seq_id - prev_seq_id - 1
        prev_seq_id = seq_id
    print(f"Total missing frames: {total_missing_frames}")


if __name__ == "__main__":
    main()
