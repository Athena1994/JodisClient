from dataclasses import dataclass
from typing import Dict

from torch import Tensor
from core.data.data_provider import ChunkType, ContinuousProvider, Sample
from core.data.normalizer import Normalizer

from core.simulation.data_classes import State
from utils.config_utils import assert_fields_in_dict


class StateManager:

    def __init__(self) -> None:
        self._context = None
        self._samples = None

    def reset_state(self, context: dict) -> None:
        self._context = context.copy()
        self._samples = None

    def update_context(self,
                       context: dict) -> None:

        self._context = context.copy()

    def update_samples(self,
                       samples: Dict[str, Sample]) -> None:
        self._samples = samples.copy()

    def get_context(self) -> dict:
        if self._context is None:
            return None
        return self._context.copy()

    def get_samples(self) -> Dict[str, Sample]:
        if self._samples is None:
            return None
        return self._samples.copy()

    def get_state(self) -> State:
        return State(self.get_samples(), self.get_context())


class StateProvider(ContinuousProvider):

    @dataclass
    class Config:
        normalizer: Normalizer.Config
        include: list[str]

        @staticmethod
        def from_dict(conf: dict) -> 'StateProvider.Config':
            assert_fields_in_dict(conf, ['normalizer', 'include'])
            return StateProvider.Config(
                Normalizer.Config.from_dict(conf['normalizer']),
                conf['include'])

    def __init__(self,
                 state_manager: StateManager,
                 normalizer: Normalizer,
                 cfg: Config) -> None:
        super().__init__()
        self._state_manager = state_manager
        self._normalizer = normalizer

        self._normalizer_conf = cfg.normalizer

        self._fields = cfg.include

    def get_iterator(self, chunk_type: ChunkType) -> "StateProvider":
        return self

    def provide_sample(self, context: dict):
        np_dict = {
            k: [context[k]]
            for k in context if k in self._fields
        }

        np_dict = self._normalizer.normalize_data(np_dict,
                                                  self._normalizer_conf)
        return Sample(Tensor([np_dict[k][0] for k in np_dict]), context)

    def __next__(self) -> Sample:
        context = self._state_manager.get_context()
        if context is None:
            return None
        return self.provide_sample(context)
