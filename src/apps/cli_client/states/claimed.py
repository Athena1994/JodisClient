

from apps.cli_client.states.connected import ConnectedState
from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState


class ClaimedState(ConnectedState):
    def __init__(self,
                 client: APIClient,
                 client_name: str,
                 client_id: int):
        super().__init__(client)
        self._client_name = client_name
        self._client_id = client_id
        self.add_handler('drop', self._drop_claim)

    @property
    def client_name(self) -> str:
        return self._client_name

    @property
    def client_id(self) -> int:
        return self._client_id

    def _drop_claim(self, _, __) -> BaseState:
        self.client.drop_claim()
        return ConnectedState(self.client)

    def prompt_prefix(self, _) -> str:
        return (
            f'connected ({self._client_id}: {self._client_name})'
        )
