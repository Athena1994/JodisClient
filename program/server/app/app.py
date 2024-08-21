import json
import logging
from flask import Flask
from flask_cors import CORS
from flask_injector import FlaskInjector, singleton
from flask_socketio import SocketIO

from program.server.app.socket_namespaces import ClientEventNamespace
from program.server.server import Server
from program.server.app.endpoints.clients import clients_pb
from program.server.app.endpoints.jobs import jobs_pb


logging.basicConfig(level=logging.INFO)

PORT = 5000


def configure(binder):
    with open('program/server/sql_test_cfg.json', 'r') as f:
        cfg = Server.Config.from_dict(json.load(f))

    binder.bind(Server, to=Server(cfg), scope=singleton)


app = Flask(__name__)
app.register_blueprint(clients_pb)
app.register_blueprint(jobs_pb)

CORS(app)

socketio = SocketIO(app)
socketio.on_namespace(ClientEventNamespace('/client'))

FlaskInjector(app=app, modules=[configure])

if __name__ == '__main__':
    socketio.run(app, use_reloader=True, debug=True, port=PORT)
