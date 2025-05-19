

from enum import Enum
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Type, TypeVar
import uuid

from utils.config.helper import load_config, save_config
from utils.config.attribute import Attribute
from utils.config.decorator import config


def _default_serializer(value: object) -> str:
    return json.dumps(value.__dict__)


def _default_deserializer(json_str: str) -> object:
    return json.loads(json_str)


class Cache:

    def __init__(self, base_path: Path,
                 working_dir: str = "./cache",
                 config_file: str = "config.json"):

        self._working_dir = base_path.joinpath(working_dir)
        self._config_file = config_file

        self._cfg: CacheConfig = None
        self._tmp_fields: Dict[str, object] = {}

        if not base_path.exists():
            raise FileNotFoundError(f"Base directory {base_path} does not"
                                    " exist.")

        if not self._working_dir.exists():
            logging.info(f"Creating working directory: {self._working_dir}")
            self._working_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_cache()

    # --- properties ---

    @property
    def config_file(self) -> Path:
        return self._at_working_dir(self._config_file)

    # --- public methods ---

    T = TypeVar('T')

    def check_integrity(self):
        for key, value in self._cfg.files.items():
            path = self._at_working_dir(value)
            if not path.exists():
                raise FileNotFoundError(f"Corrupted cache at key {key}! File "
                                        f"'{path}' not found!")

    def clear(self):
        """
        Clear cache from the working directory.
        :return: None
        """
        def clear_dir(path: Path):
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    clear_dir(child)
                    child.rmdir()

        clear_dir(self._working_dir)
        self._tmp_fields.clear()
        self._cfg = None

        logging.info("Cleared cache.")

        self._initialize_cache()

    def has(self, key: str | Enum) -> bool:
        if isinstance(key, Enum):
            key = key.value
        return key in self._cfg.fields or key in self._tmp_fields

    def has_file(self, key: str | Enum) -> bool:
        if isinstance(key, Enum):
            key = key.value

        return key in self._cfg.files \
            and self._at_working_dir(self._cfg.files[key]).is_file()

    def has_dir(self, key: str | Enum) -> bool:
        if isinstance(key, Enum):
            key = key.value

        return key in self._cfg.files \
            and self._at_working_dir(self._cfg.files[key]).is_dir()

    def set(self, key: str | Enum, value: object, tmp: bool,
            serializer: Callable[[object], str] = _default_serializer):
        if isinstance(key, Enum):
            key = key.value

        target = self._tmp_fields if tmp else self._cfg.fields
        non_target = self._cfg.fields if tmp else self._tmp_fields

        if key in non_target:
            raise KeyError(f"Key '{key}' already exists in "
                           f"{'permanent' if tmp else 'temporary'} cache.")
        target[key] = serializer(value)
        logging.info(f"Cached '{key}'.")

        if not tmp:
            self._save_config()

    def get(self, key: str | Enum,
            type: Type[T] = None,
            raise_on_missing: bool = True,
            deserializer: Callable[[str], object] = _default_deserializer) -> T:
        if isinstance(key, Enum):
            key = key.value

        value = self._get(key, True, raise_on_missing)

        if value is not None:
            value = deserializer(value)
            if type is not None:
                value = self.T(value)

        return value

    def create_file(self, key: str | Enum) -> Path:
        """
        Create a file in the cache.
        :param key: The key to associate with the file.
        :return: The path to the created file.
        """
        if isinstance(key, Enum):
            key = key.value

        return self._add_location(key, False)

    def create_dir(self, key: str | Enum) -> Path:
        """
        Create a directory in the cache.
        :param key: The key to associate with the directory.
        :return: The path to the created directory.
        """
        if isinstance(key, Enum):
            key = key.value

        return self._add_location(key, True)

    def get_file(self, key: str | Enum, raise_on_missing: bool = True) \
            -> Path | None:
        if isinstance(key, Enum):
            key = key.value

        path = self._get_location(key, raise_on_missing)
        if path is not None and not path.is_file():
            raise IsADirectoryError(f"Key '{key}' is a directory, not a "
                                    "file.")
        return path

    def get_dir(self, key: str | Enum, raise_on_missing: bool = True)\
            -> Path | None:
        if isinstance(key, Enum):
            key = key.value

        path = self._get_location(key, raise_on_missing)
        if path is not None and not path.is_dir():
            raise NotADirectoryError(f"Key '{key}' is a file, not a "
                                     "directory.")
        return path

    # --- private methods ---

    def _get_location(self, key: str, raise_on_missing: bool = True) \
            -> Path | None:
        file = self._get(key, False, raise_on_missing)
        if file is None:
            return None
        path = self._at_working_dir(file)
        if not path.exists():
            raise FileNotFoundError(f"Corrupted cache at key {key}! File "
                                    f"'{path}' not found!")
        return path

    def _add_location(self, key: str, dir: bool) -> Path:
        if key in self._cfg.files:
            raise KeyError(f"Key '{key}' already exists in cache.")

        file_name = uuid.uuid4().hex
        file = self._at_working_dir(file_name)

        if dir:
            file.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created cache dir '{file_name}' for key '{key}'.")
        else:
            file.touch()
            logging.info(f"Created cache file '{file_name}' for key '{key}'.")

        self._cfg.files[key] = file_name
        self._save_config()

        return self._at_working_dir(file_name)

    def _get(self, key: str,
             retrieve_field: bool,
             raise_on_missing: bool) -> object | None:

        targets = [self._tmp_fields, self._cfg.fields] if retrieve_field \
            else [self._cfg.files]

        for target in targets:
            if key in target:
                return target[key]

        if raise_on_missing:
            raise KeyError(f"Key '{key}' not found in cache.")

        return None

    def _at_working_dir(self, path: Path):
        """
        Join the working directory with the given path.
        :param path: The path to join.
        :return: The joined path.
        """
        return self._working_dir / path

    def _initialize_cache(self):
        """
        Load the cache configuration from the working directory.
        """
        if self.config_file.exists():
            try:
                self._cfg = load_config(self.config_file, CacheConfig)
                self.check_integrity()
                logging.info("Successfully retrieved cache.")
            except Exception as e:
                logging.error(f"Failed to load cache: {e}")

        if self._cfg is None:
            self._cfg = CacheConfig()
            self._save_config()
            logging.info("Initialized empty cache.")

    def _save_config(self):
        save_config(self._cfg, self.config_file)


@config
class CacheConfig:
    files: Dict[str, str] = Attribute(
        default={}, schema={'additionalProperties': {'type': 'string'}})

    fields: Dict[str, str] = Attribute(
        default={}, schema={'additionalProperties': {'type': 'string'}})
