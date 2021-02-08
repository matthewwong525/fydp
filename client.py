#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""PyBluez simple example rfcomm-client.py
Simple demonstration of a client application that uses RFCOMM sockets intended
for use with rfcomm-server.
Author: Albert Huang <albert@csail.mit.edu>
$Id: rfcomm-client.py 424 2006-08-24 03:35:54Z albert $
"""
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))


while True:
    full_msg = ''
    while True:
        msg = s.recv(8)
        if len(msg) <= 0:
            break
        full_msg += msg.decode("utf-8")

    if len(full_msg) > 0:
        print(full_msg)
