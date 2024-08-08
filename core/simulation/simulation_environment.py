
from abc import abstractmethod
from typing import Dict
from core.data.data_provider import ChunkType
from core.simulation.data_classes import State, TransitionResult
import core.simulation.sample_provider as sp


class SimulationEnvironment:

    @abstractmethod
    def get_initial_context(self) -> dict:
        pass

    @abstractmethod
    def on_episode_start(self,
                         context: dict,
                         mode: ChunkType) -> dict:
        pass

    @abstractmethod
    def on_new_samples(self,
                       samples: Dict[str, sp.Sample],
                       context: dict,
                       mode: ChunkType) -> dict:
        pass

    @abstractmethod
    def perform_transition(self,
                           samples: Dict[str, sp.Sample],
                           context: dict,
                           action: int,
                           mode: ChunkType) -> dict:
        pass

    @abstractmethod
    def on_transition(self,
                      transition_result: TransitionResult,
                      mode: ChunkType) -> dict:
        pass

    @abstractmethod
    def calculate_reward(self,
                         old_state: State,
                         new_state: State,
                         action: int,
                         mode: ChunkType) -> float:
        pass
