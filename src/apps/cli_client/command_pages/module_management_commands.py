

from apps.cli_client.services.module_service import ModuleService
from core.cli_state_machine.cli_command_decorator import CliCommand
from core.cli_state_machine.cli_command_page import CLICommandPage
from core.http_api.http_api import HttpAPI
from utils.injector import inject


class ModuleManagementCommands(CLICommandPage):

    def __init__(self):
        super().__init__()

    @CliCommand('installed-modules')
    @inject
    def _get_modules(self, ms: ModuleService):
        modules = ms.get_modules()
        print("Loaded modules:")
        for module in modules:
            print(f"\t{module.name}: {module.version} ({module.src_hash})")

    @CliCommand('server-modules')
    @inject
    def _get_server_modules(self, api: HttpAPI):
        modules = api.modules.get_server_modules()
        print("server-side job modules:")
        for module in modules:
            print(f"\t{module.name}: {module.version} "
                  f"({module.id, module.hash})")

    @CliCommand('is-compatible')
    @inject
    def _check_compatability(self, ms: ModuleService,
                             module_name: str, remote_module_id: str):

        if not ms.has_module(module_name):
            print(f"Module {module_name} not found.")
            return

        if ms.is_module_compatible(module_name, remote_module_id):
            print(f"Module {module_name} is compatible with "
                  f"{remote_module_id}.")
        else:
            print(f"Module {module_name} is NOT compatible with "
                  f"{remote_module_id}.")

    @CliCommand('update')
    @inject
    async def _update_module(self, ms: ModuleService, api: HttpAPI,
                             module: str, force: bool = False):
        if not ms.has_module(module):
            print(f"Module {module} not found.")
            return

        versions = api.modules.get_module_versions(module)

        if len(versions) == 0:
            print(f"Module {module} not found on server.")
            return

        await ms.install_or_update(versions[0].id, reinstall=force)

    @CliCommand('install')
    @inject
    async def _install_module(self, ms: ModuleService, api: HttpAPI,
                              module: str):
        if ms.has_module(module):
            print(f"Module {module} already instlaled.")
            return

        versions = api.modules.get_module_versions(module)

        if len(versions) == 0:
            print(f"Module {module} not found on server.")
            return

        await ms.install_or_update(versions[0].id)
