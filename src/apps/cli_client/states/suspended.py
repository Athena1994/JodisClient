

from apps.cli_client.states.active import ActiveState
from apps.cli_client.states.claimed import ClaimedState
from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState


class SuspendedState(ClaimedState):
    def __init__(self,
                 client: APIClient,
                 client_name: str,
                 client_id: int):
        super().__init__(client, client_name, client_id)

        self.add_handler('activate', self._claim_active)

    def _claim_active(self, _, __) -> BaseState:
        new_state = self.client.claim_active_state()
        if new_state:
            print(f"Claimed active state ({new_state})")
            return ActiveState(self.client,
                               self._client_name, self._client_id)
        else:
            print("Failed to claim active state")
            return self

    def prompt_prefix(self, _) -> str:
        return (
            f'connected ({self._client_id}: {self._client_name}, suspended)'
        )
