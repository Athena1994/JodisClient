from abc import abstractmethod
from typing import Callable, Tuple

import numpy as np
from torch import Tensor

import torch.nn as nn


class QFunction:

    @abstractmethod
    def get_q_values(self, state: object) -> np.ndarray:
        pass


class DQNWrapper(nn.Module):

    @staticmethod
    def _def_converter(state: object) -> Tuple[Tensor]:
        return state,

    def __init__(self,
                 nn: nn.Module,
                 state_converter: Callable[[object], Tuple] = _def_converter):
        """
        Wraps a standard-nn so a conversion step can be added before forwarding
        input to nn. Expected forward input is a replay buffer state.
        :param nn:
        :param state_converter: default -> state as tuple
        """
        super().__init__()
        self._converter = state_converter
        self._nn = nn

    def forward(self, x):
        x = self._converter(x)
        return self._nn.forward(*x)


class DeepQFunction(QFunction):

    def __init__(self, nn: nn.Module):
        super().__init__()
        self._nn = nn

    def get_q_values(self, state: object) -> np.ndarray:

        self._nn.requires_grad_(False)
        result = self._nn(state)
        result = result.cpu()
        return np.array(result)


class Arbiter:
    @abstractmethod
    def decide(self, state: object, explore: bool) -> np.array:
        pass


class QSigArbiter(Arbiter):
    def __init__(self, q_fct: QFunction, sig: float = 0.05):
        super().__init__()
        self._q_fct = q_fct
        self._sig = sig

    def decide(self, state: object, explore: bool) -> np.array:
        q_values = self._q_fct.get_q_values(state)
        if len(q_values.shape) == 1:
            q_values = q_values.reshape(1, -1)
        action_ix = np.argmax(q_values, axis=-1)

        if explore:
            random = np.random.uniform(0, 1, size=action_ix.shape) <= self._sig
            action_ix[random] = np.random.choice(range(q_values.shape[-1]),
                                                 np.sum(random))

        return action_ix

