import socket

mgrp = "239.192.35.162"
server = "", 7148
interface = "10.100.88.1"

s = socket.socket(type=socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(server)
mreq = socket.inet_aton(mgrp) + socket.inet_aton(interface)
s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
data = s.recv(1024)
print(f"{data}")

s.close()
