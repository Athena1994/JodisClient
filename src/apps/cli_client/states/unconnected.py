
from apps.cli_client.states.connected import ConnectedState
from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState, CliCommand, ignore


class UnconnectedState(BaseState):
    def __init__(self):
        super().__init__()

    @inject
    @CliCommand('connect')
    def connect(self, client: APIClient) -> BaseState:
        if self.client.connect():
            return ConnectedState(self.client)
        else:
            return self

    def prompt_prefix(self) -> str:
        return 'unconnected'
