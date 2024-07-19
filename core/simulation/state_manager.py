
from pandas import DataFrame
from typing import Dict

from torch import Tensor
from core.data.data_provider import ChunkType, ContinuousProvider, Sample
from core.data.normalizer import Normalizer
from core.simulation.simulation_environment import State


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

    def __init__(self, 
                 state_manager: StateManager,
                 normalizer: Normalizer,
                 conf: dict) -> None:
        super().__init__()
        self._state_manager = state_manager
        self._normalizer = normalizer

        if "normalizer" not in conf:
            raise ValueError("Normalizer field not found in config.")
        self._normalizer_conf = conf['normalizer']

        if "include" not in conf:
            raise ValueError("Include field not found in config.")
        self._fields = conf["include"]


    def get_iterator(self, chunk_type: ChunkType) -> "StateProvider":
        return self
    
    def __next__(self) -> Sample:
        context = self._state_manager.get_context()
        if context is None:
            return None

        context_df = DataFrame(context, index=[0])[self._fields]
        n_df = self._normalizer.normalize_df(context_df,
                                             self._normalizer_conf)

        return Sample(Tensor(n_df.values).flatten(), context)

    def update_sample(self, sample: Sample) -> Sample:
        return next(self)