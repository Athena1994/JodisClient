from cli_client.command_pages.general_client_commands \
    import GeneralClientCommands
from cli_client.command_pages.module_management_commands \
    import ModuleManagementCommands
from cli_client.command_pages.server_job_commands\
    import ServerJobCommands
from cli_client.services.job_service import JobService
from jodisutils.cli.cli_state_machine.cli_command_decorator import CliCommand
from jodisutils.cli.cli_state_machine.cli_command_page import CLICommandPage
from jodiscore.client.socket_api.api_client import APIClient
from jodisutils.cli.cli_state_machine.base_state import BaseState
from jodisutils.architecture.injector import inject


class ActiveCommandPage(CLICommandPage):

    @CliCommand('deactivate')
    @inject
    def _release_active(self, client: APIClient) -> BaseState:
        new_state = client.release_active_state()
        if new_state:
            print(f"Entered suspended state ({new_state})")
            from cli_client.states.suspended import SuspendedState
            return SuspendedState.instance()
        else:
            print("Failed to release active state")

    @CliCommand('run-next-job')
    @inject
    async def _start_next(self, js: JobService, state: BaseState):
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
