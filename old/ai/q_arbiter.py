import unittest
from abc import abstractmethod
from typing import Callable, Tuple

import numpy as np
import torch.nn
from torch import Tensor

from ai.decision_arbiter import DecisionArbiter
from ai.utils import Actions


class QFunction:

    @abstractmethod
    def get_q_values(self, state: object) -> np.ndarray:
        pass


class DQNWrapper(torch.nn.Module):

    @staticmethod
    def _def_converter(state: object) -> Tuple[Tensor]:
        return state,

    def __init__(self,
                 nn: torch.nn.Module,
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

    def __init__(self, nn: torch.nn.Module):
        super().__init__()
        self._nn = nn

    def get_q_values(self, state: object) -> np.ndarray:

        self._nn.requires_grad_(False)
        result = self._nn(state)
        result = result.cpu()
        return np.array(result)


# class OptimalQFunction(QFunction):
#
#     def __init__(self):
#         super().__init__()
#         self._eval = RewardEvaluator()
#         self._env = None
#
#     def set_env(self, env: EnvironmentSimulation):
#         self._env = env
#
#     def get_q_values(self,
#                      feature_vec: np.array,
#                      state_vec: np.array) -> List[float]:
#         chunk = self._env.get_chunk()
#         wallet = self._env.get_wallet()
#
#         verb = wallet.get_verbosity()
#         wallet.set_verbosity(0)
#
#         ix = chunk.get_ix()
#         wallet_state = wallet.get_state()
#
#         buy_reward = self._eval.get_action_reward(chunk, ix, wallet_state,
#                                                   Actions.BUY)
#         sell_reward = self._eval.get_action_reward(chunk, ix, wallet_state,
#                                                    Actions.SELL)
#         wait_reward = self._eval.get_action_reward(chunk, ix, wallet_state,
#                                                    Actions.WAIT)
#
#         return [wait_reward,
#                 buy_reward,
#                 sell_reward]


# class RandomQFunction(QFunction):
#
#     def get_q_values(self,
#                      feature_vec: np.array,
#                      state_vec: np.array) -> List[float]:
#         return np.random.uniform(0, 1, 3) * np.array([100, 1, 1])


class QArbiter(DecisionArbiter):
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


class TestDeepQFunction(unittest.TestCase):

    def test_normal_input(self):
        nn = DQNWrapper(torch.nn.Identity(), lambda state: (Tensor(state),))
        qf = DeepQFunction(nn)

        test_data = np.random.uniform(-100, 100, size=(64, 8))
        for d in test_data:
            q_vals = qf.get_q_values(d)
            self.assertTrue(np.allclose(d, q_vals),
                            f"exp: {d}, given: {q_vals}")

        self.assertTrue(np.allclose(test_data, qf.get_q_values(test_data)))

        qf = DeepQFunction(nn)
        self.assertTrue(np.allclose(test_data,
                                    qf.get_q_values(Tensor(test_data))))


class TestQArbiter(unittest.TestCase):
    def test(self):
        np.random.seed(42)
        test_data = np.array([[0, 1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5, 3],
                             [2, 3, 4, 5, 0, 1],
                             [0, 1, 6, 3, 4, 5],
                             [0, 1, 2, 3, 8, 5]])
        expected_actions = [5, 4, 3, 2, 4]
        explore_prob = 0.2

        nn = DQNWrapper(torch.nn.Identity(), lambda state: (Tensor(state),))
        qf = DeepQFunction(nn)
        qa = QArbiter(qf, sig=explore_prob)

        self.assertListEqual(expected_actions,
                             list(qa.decide(test_data, explore=False)))

        test_data = np.random.uniform(-100, 100, (10000, 10))
        expected_actions = np.argmax(test_data, axis=1)
        selected_actions = qa.decide(test_data, explore=True)
        misses = np.sum(expected_actions != selected_actions)

        self.assertAlmostEqual(misses/test_data.shape[0],
                               explore_prob*0.9, 2)
