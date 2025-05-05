from __future__ import annotations

from dataclasses import dataclass

from typing import Callable, TYPE_CHECKING

from utils.cli.cli_command import CLICommand

if TYPE_CHECKING:
    from core.cli_state_machine.base_state import BaseState


class CLICommandHandler:

    def __init__(self, callback: Callable[[dict, dict], BaseState],
                 params: list[Parameter]):
        self._callback = callback
        self._params = params

    def handle(self, context: dict, command: CLICommand) -> BaseState:

        parameter_dict = {}

        for i, param in enumerate(self._params):
            try:
                value = command.get_parameter(i, param.name)
                parameter_dict[param.name] = param.get_value(value)
            except ValueError as e:
                print(f'Error processing parameter {param.name}: {e}')
                return None

        return self._callback(parameter_dict, context)

    @dataclass
    class Parameter:
        name: str
        type_: type
        default_value: object | None = None
        default_expr: str | None = None

        def has_default(self) -> bool:
            return self.default_value is not None \
                or self.default_expr is not None

        @property
        def default(self) -> object:
            if self.default_value is not None:
                return self.default_value
            elif self.default_expr is not None:
                return eval(self.default_expr)
            else:
                raise ValueError("no default value or expression for property "
                                 f"'{self.name}'")

        def get_value(self, value: object) -> object:
            if value is None:
                value = self.default

            if not isinstance(value, self.type_):
                try:
                    value = self.type_(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Parameter '{self.name}' must be of type"
                                     f' {self.type_}')
            return value
