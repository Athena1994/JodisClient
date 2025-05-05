

from apps.cli_client.states.suspended import SuspendedState
from core.cli_state_machine.base_state import BaseState
from core.cli_state_machine.state_command import StateCommand


class CancelActiveJobCommand(StateCommand[SuspendedState]):
    def __init__(self) -> None:
        super().__init__()

    def run(self, state: SuspendedState) -> BaseState:
        state.client.cancel_active_job()
