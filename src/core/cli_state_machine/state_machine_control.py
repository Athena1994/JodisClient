import enum
import logging
import traceback
from typing import List, get_args

from core.cli_state_machine.base_state import BaseState
from core.cli_state_machine.state_command import Dispatcher, StateCommand
from utils.cli.cli_command import CLICommand
from utils.cli.stoppable_input import StoppableInput


class BaseCLICommands(enum.Enum):
    HELP = 'help'
    EXIT = 'exit'


class StateMachineControl(Dispatcher):

    def __init__(self, context: dict = {}):

        self._state = None
        self._context = context

        self._command_queue: List[StateCommand] = []
        self._input = StoppableInput()

    @property
    def state(self) -> BaseState:
        return self._state

    @state.setter
    def state(self, value: BaseState):
        self._state = value
        self._state.context = self._context

    def dispatch(self, cmd: StateCommand):
        self._command_queue.append(cmd)
        self._input.cancel()

    async def _execute_command(self, cmd: StateCommand):
        if not isinstance(self.state,
                          get_args(cmd.__orig_bases__[0])[0]):
            logging.warning('Invalid state type for provided command')
            return

        try:
            self.state = cmd.run(self.state) or self.state
        except Exception as e:
            logging.error(f"Error processing command: {e}\n"
                          f"{traceback.format_exc()}")
            return

    def _wait_for_user_input(self) -> str:
        prompt = self._state.prompt_prefix()
        if len(prompt) != 0:
            print(f'{prompt}: ', end='')

        return self._input.read()

    async def run(self, init_state: BaseState):
        self.state = init_state

        while True:
            # process any pending state commands
            while len(self._command_queue) != 0:
                await self._execute_command(self._command_queue.pop(0))

            # --- await and parse user command ---
            cmd = self._wait_for_user_input()
            if cmd is None:
                continue

            try:
                parser_result = CLICommand.parse(cmd)
            except Exception as e:
                logging.error(f'Error parsing command: {e}')
                continue

            if parser_result is None:
                continue

            # --- base commands ---

            if parser_result.name == BaseCLICommands.EXIT.value:
                break

            if parser_result.name == BaseCLICommands.HELP.value:
                self.state.print_help()
                continue

            # --- state commands ---

            try:
                new_state = await self.state.handle(parser_result)
            except Exception as e:
                logging.error(f'Error processing command: {parser_result} in '
                              f'state {self.state}\n{e}\n'
                              f'{traceback.format_exc()}')
                continue

            if new_state is None:
                print('Invalid command')
            else:
                self.state = new_state
