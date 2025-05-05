
import inspect
from typing import Callable, Dict

from core.cli_state_machine.cli_command_handler import CLICommandHandler
from utils.cli.cli_command import CLICommand


class BaseState:
    def __init__(self):
        self._handlers: Dict[str, CLICommandHandler] = {}

        self._context: dict = None

        # register decorated handlers
        for name, method in inspect.getmembers(self,
                                               predicate=inspect.ismethod):
            if hasattr(method, '_cli_command'):
                command = getattr(method, '_cli_command')
                params = getattr(method, '_cli_params')
                self.add_handler(command, method, params)

    @property
    def context(self) -> dict:
        return self._context

    @context.setter
    def context(self, value: dict):
        self._context = value

    def add_handler(self,
                    cmd: str,
                    callback: Callable[[dict, dict], 'BaseState'],
                    params: list[CLICommandHandler.Parameter] = []):
        self._handlers[cmd] = CLICommandHandler(callback, params)

    def handle(self, command: CLICommand) -> 'BaseState':

        handler = self._handlers.get(command.name)
        if handler is None:
            return None

        return handler.handle(self.context, command)

    def prompt_prefix(self) -> str:
        return ''

    def print_help(self):
        print('Available commands:')
        for cmd, handler in self._handlers.items():
            params = ', '.join([f'{p.name}={p.default}' if p.has_default()
                                else p.name for p in handler._params])
            print(f'  {cmd}({params})')


class CliCommand:
    """
    Decorator class to register a function as a handler for a command.
    """
    def __init__(self, command: str, inject_context: bool = False):
        self._command = command
        self._inject_context = inject_context

    def __call__(self, func):
        def wrapper(self, parameters: dict, context: dict):
            if wrapper._inject_context:
                parameters['context'] = context
            result = func(self, **parameters)
            return result or self

        wrapper._cli_command = self._command
        wrapper._inject_context = self._inject_context

        # --- create cli parameter list ---
        sig = inspect.signature(func)

        params = filter(lambda p: p.name != 'self', sig.parameters.values())

        # ignore parameters with complex type
        params = filter(lambda p: p.annotation in (inspect.Parameter.empty,
                                                   str, int, float, bool),
                        params)

        if self._inject_context:
            params = filter(lambda p: p.name != 'context', params)

        def default_value(param: inspect.Parameter) -> object:
            if param.default is not inspect.Parameter.empty:
                return param.default
            return None

        wrapper._cli_params = [
            CLICommandHandler.Parameter(name=param.name,
                                        type_=param.annotation,
                                        default_value=default_value(param))
            for param in params
        ]

        return wrapper


def ignore(func):
    func._ignore = True  # Add a marker attribute to the function
    return func
