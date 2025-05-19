
import enum
from typing import List

from requests import HTTPError
from core.http_api.api_objects import ModuleDO
from core.http_api.http_api_base import HttpApiBase
from utils.web_ressource import WebRessource


class Endpoints(enum.Enum):
    """
    Enum for HTTP API endpoints.
    """
    MODULES_CHECK_COMPATIBILITY = "/jobs/meta/compatibility"
    MODULES_GET_ALL = "/jobs/meta/modules"
    MODULES_GET = "/jobs/meta/module"
    MODULES_DOWNLOAD = "modules/client"


class HttpModulesAPI(HttpApiBase):

    def __init__(self, base_url: str):
        super().__init__(base_url)

    def get_module(self, module_id: str) -> ModuleDO:
        try:
            response = self.get(Endpoints.MODULES_GET,
                                rel_path=module_id)
            return ModuleDO(**response)

        except HTTPError as e:
            raise ValueError(f"Server returned {e}")

    def get_module_versions(self, module_name: str) -> List[ModuleDO]:
        """
        Get the versions of a module from the server.
        :param module_name: The name of the module.
        :return: List of versions.
        """
        try:
            return [ModuleDO.from_json(module) for
                    module in self.get(Endpoints.MODULES_GET,
                                       rel_path=module_name)]
        except HTTPError as e:
            raise ValueError(f"Server returned {e}")

    def get_server_modules(self) -> List[ModuleDO]:
        """
        Get the list of modules from the server.
        :return: List of modules.
        """
        try:
            return [ModuleDO.from_json(module) for module in
                    self.get(Endpoints.MODULES_GET_ALL)]
        except HTTPError as e:
            raise ValueError(f"Server returned {e}")

    def is_module_compatible(self, server_module_id: str,
                             local_version: str,
                             src_hash: str = None) -> bool:
        """
        Check if the module is compatible with the current version.
        :param server_module_id: The ID of the module on the server.
        :param local_version: The version of the module on the local machine.
        :param src_hash: The hash of the source code directory. (for dev
                         purposes only)
        :return: True if the module is compatible, False otherwise.
        """
        try:
            params = {
                'client-version': local_version,
                'module-id': server_module_id
            }

            if src_hash:
                params.update({'src-hash': src_hash})

            result = self.get(Endpoints.MODULES_CHECK_COMPATIBILITY, params)

        except HTTPError as e:
            raise ValueError(f"Server returned {e}")

        return result['compatible']

    def get_module_download(self, module_id: str) -> WebRessource:
        """
        Download the module from the server.
        :param module_id: The ID of the module to download.
        """
        url = self._base_url/Endpoints.MODULES_DOWNLOAD.value/module_id

        return WebRessource(url)
