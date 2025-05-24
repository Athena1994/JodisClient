
from jodisutils.cli.cli_state_machine.base_state import BaseState
from jodisutils.cli.cli_state_machine.cli_command_decorator import CliCommand
from jodisutils.cli.cli_state_machine.cli_command_page import CLICommandPage
from jodiscore.client.socket_api.api_client import APIClient
from jodisutils.architecture.injector import inject


class GeneralClientCommands(CLICommandPage):

    def __init__(self):
        super().__init__()

    @CliCommand('list')
    @inject
    def _get_clients(self, client: APIClient):
        clients = client.get_client_list()
        for client in clients:
            print(f"{client['id']}: {client['name']}")

    @CliCommand('disconnect')
    @inject
    async def _disconnect(self, client: APIClient) -> BaseState:
        from cli_client.states.unconnected import UnconnectedState

        await client.disconnect()
        return UnconnectedState.instance()
