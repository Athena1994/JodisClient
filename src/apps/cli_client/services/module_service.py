

from io import BytesIO
import logging
from pathlib import Path
import zipfile

from core.http_api.http_api import HttpAPI
from core.modules.client_module import ClientModule
from errors.invalid_state_error import InvalidStateError
from utils.injector import inject
from utils.web_ressource import WebRessource

CONFIG_NAME = "module.json"


class ModuleService:

    @staticmethod
    def _find_modules(base_path: Path) -> list[Path]:
        """
        Find all modules in the base path.
        :return: List of module configs.
        """
        modules = []
        for path in base_path.iterdir():
            if path.is_dir() and path.name != "__pycache__":
                config_file = path.joinpath(CONFIG_NAME)
                if config_file.exists():
                    modules.append(config_file)

        return modules

    def get_modules(self) -> list[ClientModule]:
        """
        Get all loaded modules.
        :return: List of loaded modules.
        """
        return list(self._module_by_id.values())

    def __init__(self, root_path: Path):
        if not root_path.exists():
            raise FileNotFoundError(f"Root path {root_path} does not exist.")

        self._base_path = Path(root_path, "./modules")
        if not self._base_path.exists():
            self._base_path.mkdir(parents=True)

        self._module_by_id = {}  # hashed by id
        self._module_id_by_name = {}  # names must be unique

    @inject
    async def install_or_update(self, module_id: str,
                                api: HttpAPI,
                                reinstall: bool = False):

        # load server module data
        try:
            server_module = api.modules.get_module(module_id)
        except Exception as e:
            raise KeyError(f"Failed to get module {module_id} from server: {e}")

        local_module = None

        # check if module is already installed and needs to be updated
        if self.has_module(server_module.name):
            local_module = self.get_module_by_name(server_module.name)

            # assert module is not in use
            if local_module.is_active:
                raise InvalidStateError(
                    f"Cannot update active module {server_module.name}.\n"
                    f"Current state: {local_module.state}.\n")

            # check if module is already up to date
            if not reinstall \
                and local_module.check_version(server_module.version,
                                               server_module.hash):
                logging.info(
                    f"Local module {server_module.name}:{server_module.version}"
                    f" is already up to date with server module {module_id}.\n")
                return

            local_module.state = ClientModule.State.UPDATING

        # download module
        res = WebRessource(server_module.client_url)
        await res.download()

        # remane old files/folders to ./*.old
        if local_module and local_module.has_valid_config:
            local_module.archivate_src()

        try:
            # extract new module data
            module_path = self._base_path.joinpath(server_module.name)
            if not module_path.exists():
                module_path.mkdir(parents=True)
            with zipfile.ZipFile(BytesIO(res.content)) as reader:
                reader.extractall(module_path)
        except Exception as e:
            # remove old files/folders
            if local_module and local_module.has_valid_config:
                local_module.restore_src()
            raise ValueError(f"Failed to extract module {module_id}:\n{e}")

        logging.info(
            f"Module {server_module.name} downloaded and extracted to "
            f"{module_path}. Reinitializing module...\n")

        if local_module:
            await self.unload_module(local_module, block=True)

        await self.load_module(ClientModule(module_path.joinpath(CONFIG_NAME)))

    async def unload_module(self, module: ClientModule, block: bool):
        if module.is_active:
            if block:
                await module.stop()
            else:
                raise InvalidStateError(
                    f"Cannot unload active module {module.name}.\n"
                    f"Current state: {module.state}.\n")

        del self._module_by_id[module.id]
        del self._module_id_by_name[module.name]

    async def load_module(self, module: ClientModule):
        try:
            await module.setup_module()
        except ValueError as e:
            raise ValueError(f"Module setup failed!\n{e}")

        if self.has_module(module.name):
            raise ValueError(f"Module {module.name} already registered.")

        self._module_by_id[module.id] = module
        self._module_id_by_name[module.name] = module.id

    async def load_all_modules(self):
        module_configs = self._find_modules(self._base_path)
        for config in module_configs:
            await self.load_module(ClientModule(config))

    def get_module(self, module_id: str) -> ClientModule:
        """
        Get a module by its id.
        :param module_id: The id of the module.
        :return: The module.
        """
        if module_id not in self._module_by_id:
            raise ValueError(f"Module {module_id} not found.")
        return self._module_by_id[module_id]

    def get_module_by_name(self, module_name: str) -> ClientModule:
        """
        Get a module by its name.
        :param module_name: The name of the module.
        :return: The module.
        """
        if module_name not in self._module_id_by_name:
            raise ValueError(f"Module {module_name} not found.")
        return self._module_by_id[self._module_id_by_name[module_name]]

    def has_module(self, module_name: str) -> bool:
        """
        Check if a module with the given name exists.
        :param module_name: The name of the module.
        :return: True if the module exists, False otherwise.
        """
        return module_name in self._module_id_by_name

    @inject
    def is_module_compatible(self, module_name: str, server_module_id: str,
                             api: HttpAPI) -> bool:
        if not self.has_module(module_name):
            raise ValueError(f"Module {module_name} not found.")

        module = self.get_module_by_name(module_name)

        return api.modules.is_module_compatible(
            server_module_id, module.version, module.src_hash)
