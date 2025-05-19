from apps.cli_client.command_pages.module_management_commands \
    import ModuleManagementCommands
from apps.cli_client.command_pages.server_job_commands import \
    ServerJobCommands
from core.cli_state_machine.base_state import BaseState
from core.cli_state_machine.cli_command_decorator import CliCommand
from core.cli_state_machine.cli_command_page import CLICommandPage
from core.socket_api.api_client import APIClient
from utils.injector import inject


class UnconnectedStateCommands(CLICommandPage):

    def __init__(self):
        super().__init__()

    @CliCommand('connect')
    @inject
    async def connect(self, client: APIClient) -> BaseState:
        if await client.connect():
            from apps.cli_client.states.connected import ConnectedState
            return ConnectedState.instance()


class UnconnectedState(BaseState):
    _instance: 'UnconnectedState' = None

    def __init__(self):
        super().__init__(UnconnectedStateCommands(),
                         ModuleManagementCommands(), ServerJobCommands())

    @classmethod
    def instance(cls) -> 'UnconnectedState':
        if cls._instance is None:
            cls._instance = UnconnectedState()
        return cls._instance

    def prompt_prefix(self) -> str:
        return 'unconnected'
