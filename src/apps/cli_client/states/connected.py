
from socket import gethostname
from apps.cli_client.command_pages.general_client_commands \
    import GeneralClientCommands
from apps.cli_client.command_pages.module_management_commands \
    import ModuleManagementCommands
from apps.cli_client.command_pages.server_job_commands \
    import ServerJobCommands
from apps.cli_client.services.config_service import ConfigService
from core.cli_state_machine.cli_command_decorator import CliCommand
from core.cli_state_machine.cli_command_page import CLICommandPage
from core.socket_api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState
from utils.injector import inject


class ConnectedStateCommands(CLICommandPage):
    def __init__(self):
        super().__init__()

    @CliCommand('register')
    @inject
    def _register(self, client: APIClient, cs: ConfigService,
                  name: str = '', save: bool = False):

        if name == '':
            name = gethostname()

        cs.config.client_id = client.register(name)
        if save:
            cs.save()

    @CliCommand('claim')
    @inject
    async def _claim(self, client: APIClient, cs: ConfigService,
                     context, id: int = -1) -> BaseState:
        if id == -1:
            id = cs.config.client_id

        result = await client.claim_client(id)

        if result:
            context['name'] = result['name']
            context['id'] = result['id']

            if result['state'] == 'ACTIVE':
                from apps.cli_client.states.active import ActiveState
                return ActiveState.instance()
            else:
                from apps.cli_client.states.suspended import SuspendedState
                return SuspendedState.instance()
        else:
            print('Failed to claim client')


class ConnectedState(BaseState):
    _instance: 'ConnectedState' = None

    @classmethod
    def instance(cls) -> 'ConnectedState':
        if cls._instance is None:
            cls._instance = ConnectedState()
        return cls._instance

    def __init__(self):
        super().__init__(ConnectedStateCommands(), GeneralClientCommands(),
                         ModuleManagementCommands(), ServerJobCommands())

    def prompt_prefix(self) -> str:
        return 'connected'
