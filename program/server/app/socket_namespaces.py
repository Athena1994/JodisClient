import logging
from flask import request
from flask_socketio import Namespace

from program.server.server import Server


class ClientEventNamespace(Namespace):

    def __init__(self, server: Server):
        super().__init__()
        self._server = server
        print(server)

    def on_connect(self):
        logging.info(f'Socket with id {request.sid} connected')

    def on_disconnect(self):
        client_id = self._server.deregister_socket(request.sid)
        if client_id is not None:
            logging.info(f'Client {client_id} disconnected '
                         f'(socket {request.sid})')
        else:
            logging.info(f'Unassigned socket {request.sid} disconnected')

    def on_claim_client(self, client_id: int):
        if not isinstance(client_id, int):
            self.emit('error', {'message': 'Client id must be an integer'})
            logging.warning('Claim attempt with non-integer client id')
            return

        try:
            self._server.register_socket(request.sid, client_id)
            logging.info(f'Socket {request.sid} claimed client {client_id}')
            self.emit('success')
        except ValueError as e:
            logging.warning(f'Claim failed! {e}')
            self.emit('error', {'message': str(e)})
