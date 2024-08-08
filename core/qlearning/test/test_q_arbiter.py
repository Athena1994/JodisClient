
import unittest

import numpy as np
import torch

from core.qlearning.q_arbiter import ArbiterFactory, DecayingEpsilonGreedyArbiter, DeepQFunction, EpsilonGreedyArbiter, ExplorationArbiter


class TestDeepQFunction(unittest.TestCase):

    def test_normal_input(self):
        nn = torch.nn.Identity()
        qf = DeepQFunction(nn)

        test_data = np.random.uniform(-100, 100, size=(64, 8))
        for d in test_data:
            q_vals = qf.get_q_values(torch.Tensor(d))
            self.assertTrue(np.allclose(d, q_vals),
                            f"exp: {d}, given: {q_vals}")

        self.assertTrue(np.allclose(test_data,
                                    qf.get_q_values(torch.Tensor(test_data))))

        qf = DeepQFunction(nn)
        self.assertTrue(np.allclose(test_data,
                                    qf.get_q_values(torch.Tensor(test_data))))


class TestQArbiter(unittest.TestCase):
    def test_epsilon_greedy(self):
        np.random.seed(42)
        test_data = np.array([[0, 1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5, 3],
                             [2, 3, 4, 5, 0, 1],
                             [0, 1, 6, 3, 4, 5],
                             [0, 1, 2, 3, 8, 5]])
        expected_actions = [5, 4, 3, 2, 4]
        explore_prob = 0.2

        nn = torch.nn.Identity()
        qf = DeepQFunction(nn)

        qa = ArbiterFactory.create(qf,
                                   'epsilon_greedy',
                                   {'epsilon': explore_prob})
        self.assertTrue(isinstance(qa, EpsilonGreedyArbiter))
        self.assertEqual(qa.get_epsilon(), explore_prob)

        qa = EpsilonGreedyArbiter(qf, epsilon=explore_prob)

        # test without explore
        with qa.explore(False):
            actions = list(qa.decide(torch.Tensor(test_data)))
        self.assertListEqual(expected_actions, actions)

        # test with explore
        test_data = np.random.uniform(-100, 100, (10000, 10))
        expected_actions = np.argmax(test_data, axis=1)
        with qa.explore():
            selected_actions = qa.decide(torch.Tensor(test_data))
        misses = np.sum(expected_actions != selected_actions)
        self.assertAlmostEqual(misses/test_data.shape[0],
                               explore_prob*0.9, 2)

    def test_decaying_epsilon_greedy(self):
        np.random.seed(42)
        test_data = np.array([[0, 1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5, 3],
                             [2, 3, 4, 5, 0, 1],
                             [0, 1, 6, 3, 4, 5],
                             [0, 1, 2, 3, 8, 5]])
        expected_actions = [5, 4, 3, 2, 4]

        cfg = ExplorationArbiter.Config.from_dict({
            'type': 'decaying_epsilon_greedy',
            'params': {
                'epsilon_start': 0.9,
                'epsilon_end': 0.1,
                'episode_cnt': 10
            }})

        nn = torch.nn.Identity()
        qf = DeepQFunction(nn)

        qa = ArbiterFactory.create(qf,
                                   cfg.type,
                                   cfg.params)
        self.assertTrue(isinstance(qa, DecayingEpsilonGreedyArbiter))
        self.assertEqual(qa.get_epsilon(), 0.9)

        # test without explore
        with qa.explore(False):
            actions = list(qa.decide(torch.Tensor(test_data)))
        self.assertListEqual(expected_actions, actions)

        # test with explore
        test_data = np.random.uniform(-100, 100, (50000, 10))
        expected_actions = np.argmax(test_data, axis=1)
        for i in range(20):
            qa.update(i)
            with qa.explore():
                selected_actions = list(qa.decide(torch.Tensor(test_data)))
            misses = np.sum(expected_actions != selected_actions)
            self.assertAlmostEqual(misses/test_data.shape[0],
                                   max(0.1, 0.9 - 0.8 * i/10)*0.9, 2)
