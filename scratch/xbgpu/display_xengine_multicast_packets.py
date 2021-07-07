"""
Script that receives SPEAD packets out of the x-engine. Prints all the important SPEAD heap information in each packet.

This script does not use any optomised networking code. If the X-Engine transmits at a data rate that is too high, then
overflows will happen. Its up to the user to reduce the X-Engine data rates if required.

See https://docs.google.com/drawings/d/1d3CMrMl8wTQfVlyX5NXztMGGHhak-37mt4IoK7Idc_I for a description of F-Engine
output/X-Engine input packet format.

TODO: It would be useful to make this script display the xeng_raw data graphically.
"""

import argparse
import socket
import struct

# 2. Address and ports
parser = argparse.ArgumentParser(description="Script for displaying key information from the xengine output packets.")
parser.add_argument("--mcast_src_ip", default="239.10.11.10", help="IP address of multicast stream to subscribe to.")
parser.add_argument("--mcast_src_port", default="7149", type=int, help="Port of multicast stream to subscribe to.")

args = parser.parse_args()
mcast_group = args.mcast_src_ip
mcast_port = args.mcast_src_port
is_all_group = True

# 3. Opens socket listening for multicast data on mcast_group:PORT
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((mcast_group, mcast_port))
mreq = struct.pack("4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY)

sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# 4. Infinite loop interating waiting for recieved packets. NOTE: Expect buffer overflows at any reasonable data rate.
i = 0
while True:
    # 4.1. Wait for packet to be received from socket.
    data = sock.recv(10240)

    # 4.2. Print packet information
    print("Packet:", i, "Length:", len(data))
    i += 1

    print("Header             :", data[0:8].hex())
    print("Heap id            :", data[8:16].hex(), int.from_bytes(data[10:16], "big"))
    print("heap size          :", data[16:24].hex(), int.from_bytes(data[18:24], "big"))
    print("heap offset        :", data[24:32].hex(), int.from_bytes(data[26:32], "big"))
    print("payload size       :", data[32:40].hex(), int.from_bytes(data[34:40], "big"))
    print("frequency          :", data[40:48].hex(), int.from_bytes(data[42:48], "big"))
    print("timestamp          :", data[48:56].hex(), int.from_bytes(data[50:56], "big"))
    print("descriptor         :", data[56:64].hex(), int.from_bytes(data[58:64], "big"))
    print("xeng_raw[0-7]      :", data[64:72].hex())
    print("xeng_raw[8-15]     :", data[72:80].hex())
    print("xeng_raw[2032-2039]:", data[2096:2104].hex())
    print("xeng_raw[2040-2047]:", data[2104:2112].hex())

    print()
