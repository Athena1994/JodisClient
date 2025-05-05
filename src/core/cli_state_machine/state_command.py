
from abc import abstractmethod
from typing import Generic, TypeVar

from core.cli_state_machine.base_state import BaseState


T = TypeVar("T")


class StateCommand(Generic[T]):

    @abstractmethod
    def run(self, state: T) -> BaseState:
        pass

    def get_state_type(self) -> type:
        return T
