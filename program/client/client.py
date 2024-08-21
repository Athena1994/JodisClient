import logging
from socketio import SimpleClient
import requests


class Client:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._socket.disconnect()

    def __init__(self, server: str, port: int) -> None:
        self._server = server
        self._port = port

        self._socket = SimpleClient()

    def register(self, name: str) -> None:
        target = f"{self._server}:{self._port}"
        response = requests.post(f'http://{target}/client/register',
                                 params={'name': name})

        if response.status_code != 200:
            logging.error(f"Server returned {response.status_code}")
            raise ValueError(f"Server returned {response.status_code}")

        response_data = response.json()
        id = response_data['id']
        logging.info(f"Registered with client id {id}")
        return id

    def connect(self, client_id: int) -> None:
        self._socket.connect(f'http://{self._server}:{self._port}',
                             namespace='/client')
        self._socket.emit('claim_client', client_id)
        print(self._socket.receive(10))
