

from apps.cli_client.command_pages.general_client_commands \
    import GeneralClientCommands
from apps.cli_client.command_pages.module_management_commands \
    import ModuleManagementCommands
from apps.cli_client.command_pages.server_job_commands \
    import ServerJobCommands
from apps.cli_client.states.active import ActiveState
from core.cli_state_machine.cli_command_decorator import CliCommand
from core.socket_api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState
from utils.injector import inject


class SuspendedCommandPage:
    @CliCommand('activate')
    @inject
    def _claim_active(self, client: APIClient) -> BaseState:
        new_state = client.claim_active_state()
        if new_state:
            print(f"Claimed active state ({new_state})")
            return ActiveState.instance()
        else:
            print("Failed to claim active state")


class SuspendedState(BaseState):
    _instance: 'SuspendedState' = None

    @classmethod
    def instance(cls) -> 'SuspendedState':
        if cls._instance is None:
            cls._instance = SuspendedState()
        return cls._instance

    def __init__(self):
        super().__init__(SuspendedCommandPage(), GeneralClientCommands(),
                         ModuleManagementCommands(), ServerJobCommands())

    def prompt_prefix(self) -> str:
        return (
            f'connected ({self.context["id"]}: {self.context["name"]}, '
            'suspended)'
        )
