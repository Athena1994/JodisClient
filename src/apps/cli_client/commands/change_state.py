
from typing import Union
from apps.cli_client.states.active import ActiveState
from apps.cli_client.states.suspended import SuspendedState
from core.cli_state_machine.base_state import BaseState
from core.cli_state_machine.state_command import StateCommand
from utils.cli.cli_command import CLICommand


class ChangeStateCommand(StateCommand[Union[ActiveState, SuspendedState]]):
    def __init__(self, activate: bool) -> None:
        super().__init__()
        self._activate = activate

    def run(self, state: ActiveState | SuspendedState) -> BaseState:
        if self._activate and isinstance(state, SuspendedState):
            return state.handle(CLICommand.parse('activate'))
        elif not self._activate and isinstance(state, ActiveState):
            return state.handle(CLICommand.parse('deactivate'))
        else:
            return state
