

from inspect import Parameter
from apps.cli_client.states.claimed import ClaimedState
from apps.cli_client.states.suspended import SuspendedState
from core.api.api_client import APIClient
from core.cli_state_machine.base_state import BaseState


class ActiveState(ClaimedState):
    def __init__(self,
                 client: APIClient,
                 client_name: str,
                 client_id: int):
        super().__init__(client, client_name, client_id)

        self.add_handler('deactivate', self._release_active)
        self.add_handler('start-next', self._start_next)

        self.add_handler('set-phase', self._set_phase,
                         [Parameter('phase', str),
                          Parameter('cnt', int, default_value=-1)])
        self.add_handler('update-phase', self._update_phase,
                         [Parameter('ix', int),
                          Parameter('tpi', float)])
        self.add_handler('set-msg', self._set_msg,
                         [Parameter('msg', str)])

    def _start_next(self, _, __) -> BaseState:
        self.client.claim_next_job()
        return self

    def _release_active(self, _, __) -> BaseState:
        new_state = self.client.release_active_state()
        if new_state:
            print(f"Entered suspended state ({new_state})")
            return SuspendedState(self.client,
                                  self._client_name, self._client_id)
        else:
            print("Failed to release active state")
            return self

    def _set_phase(self, params: dict, _) -> BaseState:
        self.client.set_phase(params['phase'], params['cnt'])
        return self

    def _update_phase(self, params: dict, _) -> BaseState:
        self.client.update_phase(params['ix'], params['tpi'])
        return self

    def _set_msg(self, params: dict, _) -> BaseState:
        self.client.set_message(params['msg'])
        return self

    def prompt_prefix(self, _) -> str:
        return (
            f'connected ({self._client_id}: {self._client_name}, active)'
        )
