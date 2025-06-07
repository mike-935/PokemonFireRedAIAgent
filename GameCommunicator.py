#!/usr/bin/env -S python3 -u

import argparse, socket, json, select, sys

HOST = "127.0.0.1"
PORT = 65432

# Communicates with our lua socket to handle messages from the game and send back responses.
class GameCommunicator:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.port = self.socket.getsockname()[1]
        print(f'Bound to port {self.port}')

    # Send a message to the server
    def send_message(self, connection, message):
        print(f'Sent Message: {message}')
        connection.sendall(message.encode("utf-8"))

    # Read a message given from the lua code
    def recieve_message(self, connection):
        message = ""

        while not message.endswith('\r\n'):
            data = connection.recv(1028).decode()
            if not data:
                break
            message += data

        return message
    
    def send_button(self, connection, value):
        self.send_message(connection, f'PRESS_KEY {value}')
        # return self.recieve_message()
    
    def send_button_release(self, connection, value):
        self.send_message(connection, f'RELEASE_KEY {value}')

    # Run the server to listen for incoming connections and handle messages
    def run(self):
        try:
            self.socket.listen()
            connection, address = self.socket.accept()
            with connection:
                print(f'Connected by {address}')
                while True:
                    data = self.recieve_message(connection)
                    if data:
                        command, value = input("Enter command and value: ").split()
                        if command == "PRESS_KEY":
                            self.send_button(connection, value + "\r\n")
                        if command == "RELEASE_KEY":
                            self.send_button_release(connection, value + "\r\n")
                        print('We read:', data)

        except (KeyboardInterrupt, SystemExit) as e:
            print("Exiting gracefully...")
            self.socket.close()
            return
        
    def parse_input(self, data):
        pass
        
        

