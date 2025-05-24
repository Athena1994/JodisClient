
from cli_client.states.suspended import SuspendedState
from jodiscore.client.socket_api.api_client import APIClient
from jodisutils.cli.cli_state_machine.state_command import StateCommand
from jodisutils.cli.cli_state_machine.base_state import BaseState
from jodisutils.architecture.injector import inject


class PauseActiveJobCommand(StateCommand[SuspendedState]):
    def __init__(self) -> None:
        super().__init__()

    @inject
    def run(self, state: SuspendedState, client: APIClient) -> BaseState:
        client.drop_active_job()
