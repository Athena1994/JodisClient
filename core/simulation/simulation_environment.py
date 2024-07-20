
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Self
from core.data.data_provider import ChunkType
from core.simulation.sample_provider import Sample


@dataclass
class State:
    samples: Dict[str, Sample]
    context: dict

    def copy(self) -> Self:
        return State(self.samples.copy(),
                     self.context.copy())


class SimulationEnvironment:

    @abstractmethod
    def get_initial_context(self) -> dict:
        pass

    @abstractmethod
    def perform_transition(self,
                           samples: Dict[str, Sample],
                           context: dict,
                           action: int,
                           mode: ChunkType) -> dict:
        pass

    @abstractmethod
    def calculate_reward(self,
                         old_state: State,
                         new_state: State,
                         action: int) -> float:
        pass

    @abstractmethod
    def on_episode_start(self, context: dict) -> dict:
        pass

    @abstractmethod
    def on_action(self,
                  context: dict,
                  action: int,
                  mode: ChunkType) -> dict:
        pass
