

from dataclasses import dataclass
import json
import os
from jsonschema import validate


@dataclass
class Config:
    server: str = 'localhost'
    port: int = 5000

    client_id: int = -1


schema = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "default": "localhost"},
        "port": {
            "type": "integer",
            "default": 5000,
            "minimum": 1, "maximum": 65535},
        "client_id": {
            "type": "integer",
            "default": -1, "minimum": -1},
    },
    "required": ["server", "port", "client_id"],
}


class ConfigService:
    def __init__(self, file: str):
        self._config: Config = None
        self._file = file

        if not os.path.exists(file):
            json_data = {}
        else:
            with open(file, 'r') as f:
                json_data = json.load(f)

        for key, value in schema["properties"].items():
            if key not in json_data and "default" in value:
                json_data[key] = value["default"]

        validate(json_data, schema)

        self._config = Config(**json_data)

    @property
    def config(self) -> Config:
        return self._config

    def save(self):

        validate(self._config.__dict__, schema)

        with open(self._file, 'w') as f:
            json.dump(self.config, f)
