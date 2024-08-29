import json
from server import Server

with open('program/server/sql_test_cfg.json', 'r') as f:
    cfg = Server.Config.from_dict(json.load(f))

s = Server(cfg)

s.create_tables()
s.add_job({'test': 'test'}, 'test_job', 'Test job to assure everything is '
                                        'working')
s.add_job({'test': 'test2'}, 'foo', 'bar')
s.add_client('test')
