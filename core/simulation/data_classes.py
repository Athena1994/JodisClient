from dataclasses import dataclass
from typing import Dict, Self

from torch import Tensor

from core.data.data_provider import Sample
from core.qlearning.replay_buffer import Experience


@dataclass
class State:
    samples: Dict[str, Sample]
    context: dict

    def copy(self) -> Self:
        return State(self.samples.copy(),
                     self.context.copy())

    def to_tensor(self, cuda: bool) -> Dict[str, Tensor]:
        if cuda:
            return {
                k: self.samples[k].tensor[None, ...].cuda()
                for k in self.samples
            }
        else:
            return {
                k: self.samples[k].tensor[None, ...]
                for k in self.samples
            }


@dataclass
class TransitionResult:
    old_state: State
    action: int
    reward: float
    new_state: State

    def as_experience(self):
        return Experience(
            self.old_state.to_tensor(False),
            self.action,
            self.reward,
            self.new_state.to_tensor(False)
        )
