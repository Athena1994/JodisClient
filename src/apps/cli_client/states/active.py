from apps.cli_client.command_pages.general_client_commands \
    import GeneralClientCommands
from apps.cli_client.command_pages.module_management_commands \
    import ModuleManagementCommands
from apps.cli_client.command_pages.server_job_commands\
    import ServerJobCommands
from apps.cli_client.services.job_service import JobService
from core.cli_state_machine.cli_command_decorator import CliCommand
from core.cli_state_machine.cli_command_page import CLICommandPage
from core.socket_api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState
from utils.injector import inject


class ActiveCommandPage(CLICommandPage):

    @CliCommand('deactivate')
    @inject
    def _release_active(self, client: APIClient) -> BaseState:
        new_state = client.release_active_state()
        if new_state:
            print(f"Entered suspended state ({new_state})")
            from apps.cli_client.states.suspended import SuspendedState
            return SuspendedState.instance()
        else:
            print("Failed to release active state")

    @CliCommand('run-job')
    @inject
    async def _start_next(self, js: JobService):
        await js.run_next()


class ActiveState(BaseState):
    _instance: 'ActiveState' = None

    @classmethod
    def instance(cls) -> 'ActiveState':
        if cls._instance is None:
            cls._instance = ActiveState()
        return cls._instance

    def __init__(self):
        super().__init__(ActiveCommandPage(), GeneralClientCommands(),
                         ModuleManagementCommands(), ServerJobCommands())

    def prompt_prefix(self) -> str:
        return (
            f'connected ({self.context["id"]}: {self.context["name"]}, '
            'active)'
        )
