

import functools
import inspect

from core.cli_state_machine.cli_command_handler import CLICommandHandler


class CliCommand:
    """
    Decorator class to register a function as a handler for a command.
    """
    def __init__(self, command: str,
                 inject_context: bool = False,
                 inheritable: bool = False):
        self._command = command
        self._inject_context = inject_context
        self._inheritable = inheritable

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(self, param_dict, context):

            if wrapper._inject_context:
                param_dict['context'] = context
            result = func(self, **param_dict)
            return result

        wrapper._cli_command = self._command
        wrapper._cli_inheriable = self._inheritable

        # --- create cli parameter list ---
        sig = inspect.signature(func)

        self._inject_context = sig.parameters.get('context') is not None
        wrapper._inject_context = self._inject_context
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
            CLICommandHandler.Parameter(
                name=param.name,
                type_=param.annotation,
                has_default=param.default is not inspect.Parameter.empty,
                default_value=default_value(param))
            for param in params
        ]

        if inspect.iscoroutinefunction(func):
            wrapper = inspect.markcoroutinefunction(wrapper)

        return wrapper
