import logging
from pathlib import Path

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config
from jodisutils.config.helper import load_config, save_config


@config
class AppConfig:
    server: str = Attribute(default="localhost")
    port: int = Attribute(default=5000)

    root: Path = Attribute(required=True, json_type=str)

    client_id: int = Attribute("client-id", default=-1)


class ConfigService:
    def __init__(self, file: Path):
        self._config: AppConfig = None
        self._file = file

        try:
            self._config = load_config(file, AppConfig)
        except FileNotFoundError:
            logging.warning(f"Config file {file} not found. Loading default "
                            "values.")
            self._config = AppConfig()
        print(self._config)

    def __call__(self, *args, **kwds):
        return self._config

    @property
    def config(self) -> AppConfig:
        return self._config

    def save(self):
        save_config(self._config, self._file)
