
import inspect
from typing import Callable, Dict, Tuple

from core.cli_state_machine.cli_command_handler import CLICommandHandler
from utils.cli.cli_command import CLICommand


class CLICommandPage:
    def __init__(self):
        self._handlers: Dict[str, CLICommandHandler] = {}

        # register decorated handlers
        for _, method in inspect.getmembers(self,
                                            predicate=inspect.ismethod):
            if hasattr(method, '_cli_command'):
                command = getattr(method, '_cli_command')
                params = getattr(method, '_cli_params')
                self.add_handler(command, method, params)

    # --- properties ----

    @property
    def commands(self) -> list[Tuple[str, str]]:
        def make_param_str(params: list[CLICommandHandler.Parameter]) -> str:
            return ', '.join(
                [f'{p.name}={p.default}' if p.has_default else p.name
                 for p in params]
            )

        return [(cmd, make_param_str(handler._params))
                for cmd, handler in self._handlers.items()]

    # --- public methods ---

    def add_handler(self,
                    cmd: str,
                    callback: Callable[[dict, dict], object],
                    params: list[CLICommandHandler.Parameter] = []):
        self._handlers[cmd] = CLICommandHandler(callback, params)

    def has_handler(self, cmd: str) -> bool:
        return cmd in self._handlers

    async def handle(self, context: dict, command: CLICommand) -> object:

        handler = self._handlers.get(command.name)
        if handler is None:
            raise KeyError(f"Command '{command.name}' not found in handler "
                           "list.")

        return await handler.handle(context, command)
