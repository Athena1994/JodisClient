from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from torch import Tensor

import torch
import torch.nn as nn

from utils.config_utils import assert_fields_in_dict


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
        with torch.no_grad():
            result = self._nn(state).cpu()
        return np.array(result)


class QDecisionPolicy:
    def __init__(self):
        pass

    def make_decision(self, q_values: np.ndarray) -> np.array:
        pass


class QGreedyPolicy(QDecisionPolicy):
    def make_decision(self, q_values: np.ndarray) -> np.array:
        return np.argmax(q_values, axis=1)


class QArbiter:
    def __init__(self,
                 q_fct: QFunction):
        self._q_fct = q_fct

    def decide(self, state: object) -> np.array:
        q_values = self._q_fct.get_q_values(state)
        if len(q_values.shape) == 1:
            q_values = q_values.reshape(1, -1)

        return self.make_decision(q_values)

    @abstractmethod
    def make_decision(self, q_values: np.ndarray) -> np.array:
        pass


class ExplorationArbiter(QArbiter):
    @dataclass
    class Config:
        type: str
        params: object

        @staticmethod
        def from_dict(cfg: dict):
            assert_fields_in_dict(cfg, ['type', 'params'])
            return ExplorationArbiter.Config(cfg['type'], cfg['params'])

    class ExploreContext:
        def __init__(self, arbiter: 'ExplorationArbiter', val: bool):
            self._arbiter = arbiter
            self._old_val = arbiter.get_explore()
            self._val = val
            self._epoch = 0

        def __enter__(self):
            self._arbiter.set_explore(self._val)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._arbiter.set_explore(self._old_val)

    def __init__(self,
                 q_fct: QFunction,
                 decision_policy: QDecisionPolicy):
        super().__init__(q_fct)
        self._explore = False
        self._decision_policy = decision_policy

    def set_explore(self, explore: bool):
        self._explore = explore

    def get_explore(self) -> bool:
        return self._explore

    def explore(self, val: bool = True) -> ExploreContext:
        return self.ExploreContext(self, val)

    def update(self, epoch: int):
        self._epoch = epoch

    def make_decision(self, q_values: np.ndarray) -> np.array:
        actions = self._decision_policy.make_decision(q_values)
        if self._explore:
            actions = self.explore_actions(actions, q_values)
        return actions

    @abstractmethod
    def explore_actions(self,
                        policy_actions: np.array,
                        q_values: np.array) -> np.array:
        pass


class EpsilonGreedyArbiter(ExplorationArbiter):
    @abstractmethod
    def from_dict(q_fct: QFunction, params: dict) -> 'EpsilonGreedyArbiter':
        assert_fields_in_dict(params, ['epsilon'])
        return EpsilonGreedyArbiter(q_fct, params['epsilon'])

    def __init__(self, q_fct: QFunction, epsilon: float = 0.05):
        super().__init__(q_fct, QGreedyPolicy())
        self._epsilon = epsilon

    def get_epsilon(self) -> float:
        return self._epsilon

    def explore_actions(self,
                        policy_actions: np.array,
                        q_values: np.array):
        randomize_choices = np.random.uniform(
            0, 1, size=policy_actions.shape) <= self._epsilon

        policy_actions[randomize_choices] \
            = np.random.choice(range(q_values.shape[-1]),
                               np.sum(randomize_choices))

        return policy_actions


class DecayingEpsilonGreedyArbiter(EpsilonGreedyArbiter):
    @abstractmethod
    def from_dict(q_fct: QFunction, params: dict) -> 'EpsilonGreedyArbiter':
        assert_fields_in_dict(params, ['epsilon_start',
                                       'epsilon_end',
                                       'episode_cnt'])
        return DecayingEpsilonGreedyArbiter(q_fct,
                                            params['epsilon_start'],
                                            params['epsilon_end'],
                                            params['episode_cnt'])

    def __init__(self,
                 q_fct: QFunction,
                 epsilon_start: float,
                 epsilon_end: float,
                 episode_cnt: int):
        super().__init__(q_fct, epsilon_start)

        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._episode_cnt = episode_cnt

    def update(self, epoch: int):
        super().update(epoch)

        if epoch > self._episode_cnt:
            self._epsilon = self._epsilon_end
        else:
            self._epsilon = self._epsilon_start + \
                            (self._epsilon_end - self._epsilon_start) * \
                            epoch / self._episode_cnt


class ArbiterFactory:
    @staticmethod
    def create(q_fct: QFunction, type: str, params: dict) -> ExplorationArbiter:
        if type == 'epsilon_greedy':
            return EpsilonGreedyArbiter.from_dict(q_fct, params)
        elif type == 'decaying_epsilon_greedy':
            return DecayingEpsilonGreedyArbiter.from_dict(q_fct, params)
        else:
            raise ValueError(f"Arbiter type {params['type']} not supported.")
