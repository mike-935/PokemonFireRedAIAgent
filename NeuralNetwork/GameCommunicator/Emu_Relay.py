#!/usr/bin/env -S python3 -u

import socket
from http.client import responses

import torch
from .GameTranslator import GameTranslator
from .GameTranslatorPandas import GameTranslatorPandas
import Network
import os
import time

HOST = "127.0.0.1"
PORT = 65432

# Communicates with our lua socket to handle messages from the game and send back responses.
class EmuRelay:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.port = self.socket.getsockname()[1]
        self.GameTranslatorPandas = GameTranslatorPandas()
        torch.manual_seed(37)
        self.neural_network = Network.Network()
        #self.neural_network.train_test_network()
        #self.GameTranslatorPandas.update_old_csv_with_status()
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
        return self.send_message(connection, f'PRESS_KEY {value}')
        # return self.recieve_message()
    
    def send_button_release(self, connection, value):
        return self.send_message(connection, f'RELEASE_KEY {value}')

    # Run the server to listen for incoming connections and handle messages
    def run(self):
        try:
            self.socket.listen()
            connection, address = self.socket.accept()
            with connection:
                print(f'Connected by {address}')
                while True:
                    print("Waiting for data...")
                    data = self.recieve_message(connection)
                    print('Received data:')
                    if data:
                        print('We read:', data)
                        response = self.parse_input(data, connection)
                        self.send_message(connection, response)

        except (KeyboardInterrupt, SystemExit) as e:
            print("Exiting gracefully...")
            self.close()
            return

    def close(self):
        print("Closing socket...")
        self.socket.close()
        
    def parse_input(self, data, connection):
        print("Parsing input data:")
        split_data = data.strip().split(",")
        response = "ERROR"
        print('split data: ', split_data[0])
        # print('split data: ', split_data[1])
        match split_data[0]:
            case "REQUEST_AI_MOVE":
                print('IN REQUEST_AI_MOVE')
                formatted_data = self.GameTranslatorPandas.translate(split_data)
                print("This is the data for the move request:", formatted_data)
                ai_action = self.neural_network.generate_ai_decision(formatted_data)
                print('ai_action:', ai_action)
                response = f"SELECT_MOVE {ai_action}"
                # response = f"Decision: {str(self.neural_network.generate_ai_decision(formatted_data))}"
                #formatted_data = self.GameTranslator.translate(split_data)
                #tensor_data = torch.tensor(formatted_data, dtype=torch.float32)
            case "SAVE_MOVE":
                print('In SAVE_MOVE')
                formatted_data = self.GameTranslatorPandas.translate(split_data)
                #formatted_data = self.GameTranslator.translate(split_data, True)
                print("Here is the formatted data to save:", formatted_data)
                response = "SAVED_TURN_DATA"
            case "PRESS_KEY":
                value = split_data[1]
                response = f'PRESS_KEY {value}'
            case "KEY_PRESSED":
                time.sleep(0.2)
                value = split_data[1]
                response = f'RELEASE_KEY {value}'
            case "KEY_RELEASED":
                time.sleep(0.4)
                response = 'KEY_PRESSED'
            case _:
                print("Unsupported command:", split_data[0])
        return response
        
        

