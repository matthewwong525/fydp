#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import socket
import errno
import sys
import pickle

HEADER_LENGTH = 10

IP = '192.168.0.106'
PORT = 1234
my_username = socket.gethostname()

# Create a socket
# socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
# socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to a given ip and port
client_socket.connect((IP, PORT))

# Set connection to non-blocking state, so .recv() call won;t block, just return some exception we'll handle
client_socket.setblocking(False)

# Prepare username and header and send them
# We need to encode username to bytes, then count number of bytes and prepare header of fixed size, that we encode to bytes as well
username = my_username.encode('utf-8')
username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
client_socket.send(username_header + username)
action = ''
print('Waiting for server to start...')

while True:
    # If message is not empty - send it
    if action == 's':
        # Wait for user to input a message
        # TODO: change this to read sensor data
        message = input('Send Sensor Data Here:')
        data = {'msg': message }
        
        # Encode message to bytes, prepare header and convert to bytes, like for username above, then send
        data = pickle.dumps(data)
        data_header = f"{len(data):<{HEADER_LENGTH}}".encode('utf-8')
        client_socket.send(data_header + data)

    try:
        # Now we want to loop over received messages (there might be more than one) and print them
        while True:
            action = client_socket.recv(1)

            # If we received no data, server gracefully closed a connection, for example using socket.close() or socket.shutdown(socket.SHUT_RDWR)
            if not len(action):
                print('Connection closed by the server')
                sys.exit()

            # Receive and decode username
            action = action.decode('utf-8')

            # Print message
            print(f'Received Action: {action}')

    except IOError as e:
        # This is normal on non blocking connections - when there are no incoming data error is going to be raised
        # Some operating systems will indicate that using AGAIN, and some using WOULDBLOCK error code
        # We are going to check for both - if one of them - that's expected, means no incoming data, continue as normal
        # If we got different error code - something happened
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            sys.exit()

        # We just did not receive anything
        continue

    except Exception as e:
        # Any other exception - something happened, exit
        print('Reading error: '.format(str(e)))
        sys.exit()