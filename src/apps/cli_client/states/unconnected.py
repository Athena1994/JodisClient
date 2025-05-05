
from apps.cli_client.states.connected import ConnectedState
from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState


class UnconnectedState(BaseState):
    def __init__(self, client: APIClient):
        super().__init__()
        self._client = client
        self.add_handler('connect', self._connect)

    @property
    def client(self) -> APIClient:
        return self._client

    @BaseState.handler
    def connect(self, params: dict, context: dict) -> BaseState:
        if self.client.connect():
            return ConnectedState(self.client)
        else:
            return self

    def prompt_prefix(self, context: dict) -> str:
        return 'unconnected'
