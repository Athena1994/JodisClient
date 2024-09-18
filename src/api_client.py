import logging
import threading
import requests

from utils.socket_io_wrapper import SocketIOWrapper


class NotConnectedError(Exception):
    pass


class APIClient:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._socket.disconnect()

    def __init__(self,
                 server: str, port: int,
                 timeout_s: float) -> None:
        self._server = server
        self._port = port
        self._timeout_s = timeout_s

        self._socket = SocketIOWrapper(timeout_s)
        self._socket.register_handler(
            'request_release',
            lambda: threading.Thread(target=self.on_release_requested).start())
        self._socket.register_handler(
            'request_activation',
            lambda: threading.Thread(target=self.on_activation_requested)
                             .start())
        self._socket.register_handler(
            'pause_job',
            lambda: threading.Thread(target=self.on_drop_active_job_requested)
                             .start())
        self._socket.register_handler(
            'cancel_job',
            lambda: threading.Thread(target=self.on_cancel_active_job_requested)
                             .start())

    def on_release_requested(self) -> None:
        pass

    def on_activation_requested(self) -> None:
        pass

    def on_drop_active_job_requested(self) -> None:
        pass

    def on_cancel_active_job_requested(self) -> None:
        pass

    def _assert_socket_connection(self) -> None:
        if not self._socket.is_connected():
            raise NotConnectedError("Not connected")

    def register(self, name: str) -> int:
        self._assert_socket_connection()

        target = f"{self._server}:{self._port}"
        response = requests.post(f'http://{target}/client/register',
                                 json={'name': name})

        if response.status_code != 200:
            logging.error(f"Server returned {response.status_code}")
            raise ValueError(f"Server returned {response.status_code}")

        response_data = response.json()
        id = response_data['id']
        logging.info(f"Registered with client id {id}")
        return id

    def get_client_list(self) -> list:
        self._assert_socket_connection()
        self._socket.emit('get_clients', namespace='/client')
        response = self._socket.receive()
        return response.data

    def connect(self) -> bool:
        try:
            self._socket.connect(f'http://{self._server}',
                                 self._port, '/client')
        except TimeoutError:
            logging.error("Connection attempt timedout!")
            return False
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            return False

        logging.info(f"Socket connection established "
                     f"({self._server}:{self._port})")
        return True

    def claim_client(self, client_id: int) -> dict:
        self._assert_socket_connection()
        self._socket.emit('claim_client', client_id,
                          namespace='/client')
        response = self._socket.receive()

        result = response.event == 'claim_successfull'

        if result:
            logging.info(f"Connected as client {client_id}")
            return response.data
        else:
            logging.error(f"Failed to connect ({response})")
            return None

    def drop_claim(self) -> None:
        self._assert_socket_connection()
        self._socket.emit('drop_claim', namespace='/client')

    def disconnect(self) -> None:
        self._socket.disconnect()
        logging.info("Disconnected")

    def claim_next_job(self) -> dict:
        self._assert_socket_connection()
        self._socket.emit('claim_next_job', namespace='/client')
        response = self._socket.receive()

        result = response.event == 'job_claimed'

        if result:
            logging.info(f"Claimed job {response.data['id']}")
            return response.data
        else:
            logging.error(f"Failed to claim job ({response})")
            return None

    def _set_state(self, active: bool) -> str | None:
        self._assert_socket_connection()
        self._socket.emit('set_state', active, namespace='/client')
        response = self._socket.receive()
        if response.event == 'success':
            new_state = response.data['state']
            logging.info(f"State set to {new_state}")
            return new_state
        else:
            logging.error(f"Failed to set state ({response})")
            return None

    def release_active_state(self) -> str | None:
        return self._set_state(active=False)

    def claim_active_state(self) -> str | None:
        return self._set_state(active=True)

    def drop_active_job(self) -> None:
        self._assert_socket_connection()

        self._socket.emit('get_active_job', namespace='/client')
        response = self._socket.receive()
        if response.event != 'success':
            logging.warning("no active job to drop")
            return

        job_id = response.data['id']

        response \
            = requests.post(f'http://{self._server}:{self._port}/jobs/unassign',
                            json={'jobIds': [job_id], 'force': True})

        if response.status_code != 200:
            logging.error(f"Failed to drop active job ({response.content})")
        else:
            logging.info(f"Dropped active job {job_id}")

    def cancel_active_job(self) -> None:
        self._assert_socket_connection()

        self._socket.emit('get_active_job', namespace='/client')
        response = self._socket.receive()
        if response.event != 'success':
            logging.warning("no active job to drop")
            return

        job_id = response.data['id']

        requests.post(f'http://{self._server}:{self._port}/jobs/delete',
                      json={'ids': [job_id], 'force': True})
