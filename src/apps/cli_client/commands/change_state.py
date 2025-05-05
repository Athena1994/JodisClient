
from apps.cli_client.states.active import ActiveState
from apps.cli_client.states.claimed import ClaimedState
from apps.cli_client.states.suspended import SuspendedState
from core.cli_state_machine.base_state import BaseState
from core.cli_state_machine.state_command import StateCommand


class ChangeStateCommand(StateCommand[ClaimedState]):
    def __init__(self, activate: bool) -> None:
        super().__init__()
        self._activate = activate

    def run(self, state: ClaimedState) -> BaseState:
        if self._activate and isinstance(state, SuspendedState):
            return state._claim_active(None, None)
        elif not self._activate and isinstance(state, ActiveState):
            return state._release_active(None, None)
        else:
            return state