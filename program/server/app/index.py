
import json
from flask import Flask, request
from ..server import Server


with open('program/server/sql_test_cfg.json', 'r') as f:
    cfg = Server.Config.from_dict(json.load(f))
server = Server(cfg)

app = Flask(__name__)


@app.route('/client/register', methods=['POST'])
def register_client():
    name = request.args.get('name')
    id = server.add_client(name)
    return json.dumps({
        'status': 'ok',
        'message': 'Client registered',
        'id': id})
