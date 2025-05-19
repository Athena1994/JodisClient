

import enum
import logging
from pathlib import Path
import traceback
from typing import List
from apps.cli_client.services.pip_service import PipService

from core.modules.module_control import ClientModuleControl
from errors.invalid_state_error import InvalidStateError
from utils.config.attribute import Attribute
from utils.config.decorator import config
from utils.dynamic_module_loader import instanciate_class, load_module
from utils.hash import md5_hash_dir
from utils.id_helper import get_module_id
from utils.injector import inject
from utils.error import Error
from utils.config.helper import load_config


@config
class EntryPoint:
    file: Path = Attribute('file', required=True, json_type=str)
    class_name: str = Attribute('class', required=True)


@config
class ModuleConfig:

    name: str = Attribute(required=True)
    version: str = Attribute(required=True)

    src_dir: Path = Attribute('src-dir', default='./src', json_type=str)
    runtime_dir: Path = Attribute('runtime-dir', default='./rt', json_type=str)

    requirements_file: Path = Attribute(
        'requirements', default='./requirements.txt', json_type=str)

    entry_point: EntryPoint = Attribute('entry-point', required=True)


class ClientModule:

    # todo: state locking
    class State(enum.Enum):
        """
        Enum for the state of the module.
        """
        UNINITIALIZED = 'uninitialized'
        INITIALIZED = 'initialized'  # config loaded
        INSTALLED = 'installed'  # dependencies installed
        READY = 'ready'  # module ready to run
        BUSY = 'busy'  # module running
        UPDATING = 'updating'  # module updating

        ERROR = 'error'

        def assert_state(self, *states: List['ClientModule.State']):
            if self not in states:
                raise ValueError(f"State Error! Invalid state {self} "
                                 f"(expected: {states})")

    def _src_path(self, path: Path) -> Path:
        return self.src_dir.joinpath(path)

    def _module_path(self, path: Path) -> Path:
        return self._base_path.joinpath(path)

    def __init__(self, config_path: Path):

        self._state = ClientModule.State.UNINITIALIZED
        self._error: Error = None

        self._id: str = None

        self._config_path: Path = config_path
        self._base_path: Path = None

        self._config: ModuleConfig = None

        self._control: ClientModuleControl = None
        self._src_hash: str = None

    def __str__(self):
        if self._state == ClientModule.State.UNINITIALIZED:
            return f"ClientModule({self._config_path})"
        else:
            return f"ClientModule({self.name}, {self.version}, {self.state})"

    # --- Public methods ---

    def restore_src(self):
        if not self.is_updating:
            raise InvalidStateError("Can only restore while updating.")

        if not self.has_valid_config:
            raise InvalidStateError("Can only restore if module successfully"
                                    " initialized.")

        archived_path = self.src_dir.with_suffix('.old')
        if archived_path.exists():
            archived_path.rename(str(archived_path)[:-4])
        else:
            logging.warning(f"{archived_path} not found for restoration.")
        archived_path = Path(str(self._config_path) + '.old')
        if archived_path.exists():
            archived_path.rename(str(archived_path)[:-4])
        else:
            logging.warning(f"{archived_path} not found for restoration.")

    def archivate_src(self):
        if not self.is_updating:
            raise InvalidStateError("Can only archivate while updating.")

        if not self.has_valid_config:
            raise InvalidStateError("Can only archivate if module successfully"
                                    " initialized.")

        if not self.src_dir.exists():
            logging.warning(f"Source directory {self.src_dir} does not exist "
                            "and can therefore not be archived.")
            return

        p = Path(f"{self.src_dir}.old")
        if p.exists():
            for child in p.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    for sub_child in child.rglob('*'):
                        if sub_child.is_file():
                            sub_child.unlink()
                        else:
                            sub_child.rmdir()
                    child.rmdir()
            p.rmdir()

        self.src_dir.replace(p)
        self._config_path.replace(f"{self._config_path}.old")
        self._src_hash = None

    async def setup_module(self):
        self.state.assert_state(ClientModule.State.UNINITIALIZED)

        self.initialize()
        await self.install()
        self.load()

    def initialize(self):
        """
        Initialize the module with the given configuration.
        """
        self.state.assert_state(ClientModule.State.UNINITIALIZED)
        logging.info(f"Initializing module {self}")

        self._config = load_config(self._config_path, ModuleConfig)
        self._base_path = self._config_path.parent

        if not self.src_dir.exists():
            raise ValueError(f"Source directory {self._config.src_dir}"
                             "does not exist.")
        if not self.runtime_dir.exists():
            self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.state = ClientModule.State.INITIALIZED

    @inject
    async def install(self, ps: PipService):
        self.state.assert_state(ClientModule.State.INITIALIZED)
        logging.info(f"Installing module dependencies for {self.name}.")

        req_path = self._module_path(self.config.requirements_file)

        try:
            await ps.install(req_path, block=True, raise_on_error=True)
        except Exception as e:
            self.error = Error("Install Failed",
                               "Installing PIP requirements failed", e)
            raise ValueError(f"Installing module {self.name} failed:\n"
                             f"{e}\n{traceback.format_exc()}")

        self.state = ClientModule.State.INSTALLED

    def load(self):
        self.state.assert_state(ClientModule.State.INSTALLED)

        self._control = instanciate_class(
            load_module(self._src_path(self._config.entry_point.file),
                        'client_module'),
            self._config.entry_point.class_name)

        self.state = ClientModule.State.READY

    async def stop(self):
        if self.state == ClientModule.State.BUSY:
            await self.control.abort()
            self.state = ClientModule.State.READY

    def check_version(self, version: str, hash: str = None) -> bool:
        """
        Check if the module version matches given version.
        :param version: The version to check.
        :param hash: The hash to check against (optional)
        :return: True if the module version is compatible, False otherwise.
        """
        if hash is not None:
            return self.src_hash == hash
        return self.version == version

    # --- Properties ---

    @property
    def has_valid_config(self) -> bool:
        """
        Check if the module is initialized.
        :return: True if the module is initialized, False otherwise.
        """
        return self._config is not None

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error: Error):
        self._error = error
        logging.error(error)

    @property
    def is_ready(self) -> bool:
        return self.state == ClientModule.State.READY

    @property
    def state(self) -> State:
        if self._error is not None:
            return ClientModule.State.ERROR

        if self._state == ClientModule.State.READY:
            if self.control.is_busy:
                return ClientModule.State.BUSY

        return self._state

    @state.setter
    def state(self, state: State):
        if self._state == state:
            return

        logging.info(f"Setting module state to {state}")
        if self._state == ClientModule.State.UPDATING:
            if state != ClientModule.State.UNINITIALIZED:
                raise InvalidStateError("UPDATING state can only be set to "
                                        "UNINITIALIZED.")

        self._state = state

    @property
    def is_updating(self) -> bool:
        """
        Check if the module is updating.
        :return: True if the module is updating, False otherwise.
        """
        return self.state == ClientModule.State.UPDATING

    @property
    def is_active(self) -> bool:
        """
        Check if the module is active.
        :return: True if the module is active, False otherwise.
        """
        return self.state in [ClientModule.State.READY,
                              ClientModule.State.BUSY]

    @property
    def src_hash(self) -> str:
        if self._src_hash is None:
            self._src_hash = md5_hash_dir(self.src_dir)
        return self._src_hash.hexdigest()

    @property
    def src_dir(self) -> Path:
        return self._base_path.joinpath(self._config.src_dir)

    @property
    def runtime_dir(self) -> Path:
        return self._base_path.joinpath(self._config.runtime_dir)

    @property
    def id(self) -> str:
        if self._id is None:
            self._id = get_module_id(self.name, self.version)
        return self._id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def control(self) -> ClientModuleControl:
        if self._control is None:
            raise ValueError("Invalid module state! Module control not loaded.")
        return self._control

    @property
    def config(self) -> ModuleConfig:
        if self._config is None:
            raise ValueError("Invalid module state! Module config not loaded.")

        return self._config
