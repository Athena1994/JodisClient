
import unittest

import numpy as np
import torch

from core.qlearning.q_arbiter import DeepQFunction, QSigArbiter


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
    def test(self):
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
        qa = QSigArbiter(qf, sig=explore_prob)

        self.assertListEqual(expected_actions,
                             list(qa.decide(torch.Tensor(test_data), 
                                            explore=False)))

        test_data = np.random.uniform(-100, 100, (10000, 10))
        expected_actions = np.argmax(test_data, axis=1)
        selected_actions = qa.decide(torch.Tensor(test_data), explore=True)
        misses = np.sum(expected_actions != selected_actions)

        self.assertAlmostEqual(misses/test_data.shape[0],
                               explore_prob*0.9, 2)
