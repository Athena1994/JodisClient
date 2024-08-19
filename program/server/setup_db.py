import json
from server import Server

with open('program/server/sql_cfg.json', 'r') as f:
    cfg = Server.Config.from_dict(json.load(f))

s = Server(cfg)

s.create_tables()
