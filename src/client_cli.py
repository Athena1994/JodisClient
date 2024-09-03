from dataclasses import dataclass
import dataclasses
import json
import logging
import os
import socket
import time

from client_manager import ClientManager
from utils.config_utils import assert_fields_in_dict

logging.basicConfig(level=logging.INFO)


CONFIG_FILE = 'client_cfg.json'


@dataclass
class Config:
    server: str
    port: int

    client_id: int

    @staticmethod
    def from_dict(cfg: dict):
        assert_fields_in_dict(cfg, ['server', 'port', 'client_id'])

        return Config(cfg['server'], cfg['port'], cfg['client_id'])

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(dataclasses.asdict(self), f)

    @staticmethod
    def load():
        with open(CONFIG_FILE, 'r') as f:
            return Config.from_dict(json.load(f))


def connect_client(client: ClientManager, cfg: Config):
    if cfg.client_id == -1:
        cfg.client_id = client.register(socket.gethostname())
        cfg.save()
    client.connect(cfg.client_id)


def main():
    if os.path.exists(CONFIG_FILE):
        cfg = Config.load()
    else:
        cfg = Config('localhost', 5000, -1)

    with ClientManager(cfg.server, cfg.port) as client:
        connect_client(client, cfg)
        time.sleep(10)


if __name__ == '__main__':
    main()
