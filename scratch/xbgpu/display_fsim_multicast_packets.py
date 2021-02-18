"""
Script that receives SPEAD packets from the fsim. It will print all the important SPEAD heap information in each packet.

This script does not use any optomised networking code. If the F-Engine transmits at a data rate that is too high, then
overflows will happen. Its up to the user to reduce the fsim data rates.

This script is hardcoded to expect multicast data on address 239.10.10.10 and port 7149.

See https://docs.google.com/drawings/d/1lFDS_1yBFeerARnw3YAA0LNin_24F7AWQZTJje5-XPg for a description of F-Engine
output/X-Engine input packet format.

TODO: It would be useful to make this script display the feng_raw data graphically.
"""

# 1. Imports
import socket
import struct

# 2. Hardcoded address and ports
MCAST_GRP = "239.10.10.10"
MCAST_PORT = 7149
IS_ALL_GROUPS = True

# 3. Opens socket listening for multicast data on MCAST_GRP:PORT
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if IS_ALL_GROUPS:
    # on this port, receives ALL multicast groups
    sock.bind(("", MCAST_PORT))
else:
    # on this port, listen ONLY to MCAST_GRP
    sock.bind((MCAST_GRP, MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)

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
    print("timestamp          :", data[40:48].hex(), int.from_bytes(data[42:48], "big"))
    print("feng_id            :", data[48:56].hex(), int.from_bytes(data[50:56], "big"))
    print("frequency          :", data[56:64].hex(), int.from_bytes(data[58:64], "big"))
    print("descriptor         :", data[64:72].hex(), int.from_bytes(data[66:72], "big"))
    print("padding 0          :", data[72:80].hex())
    print("padding 1          :", data[80:88].hex())
    print("padding 2          :", data[88:96].hex())
    print("feng_raw[0-7]      :", data[96:104].hex())
    print("feng_raw[8-15]     :", data[104:112].hex())
    print("feng_raw[1008-1015]:", data[1104:1112].hex())
    print("feng_raw[1016-1023]:", data[1112:1120].hex())

    print()
