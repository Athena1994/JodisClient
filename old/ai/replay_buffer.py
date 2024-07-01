import random
import unittest
from typing import List, NamedTuple, Tuple
from scipy.stats import norm

import numpy as np
from torch import Tensor


class Experience(NamedTuple):
    old_state: object
    action: int
    reward: float
    new_state: object


class ReplayBuffer:

    def __init__(self, size: int):
        self._reward_buffer = np.array([.0] * size)
        self._action_buffer = np.array([-1] * size)
        self._prev_state_buffer = np.array([None]*size)
        self._next_state_buffer = np.array([None]*size)

        self._next_ix = 0
        self._cnt = 0
        self._capacity = size

        self._weights = np.zeros(size)
        self._weight_sum = 0  # keep track of new/removed rewards for probs

    def __len__(self):
        return self._cnt

    def clear(self):
        self._cnt = 0
        self._weight_sum = 0
        self._weights[:] = 0

    def get_experiences(self) -> List[Experience]:
        return list(filter(
            lambda t: t[0] is not None,
            map(lambda t: Experience(*t),
                zip(self._prev_state_buffer,
                    self._action_buffer,
                    self._reward_buffer,
                    self._next_state_buffer))))

    def add_experience(self, experience: Experience, weight: float):
        if not isinstance(experience, Experience):
            raise Exception("parameter must be of type Experience")
        if weight <= 0:
            raise Exception(f'Experience weight must not be <= 0 ({weight})')
        # keep track of cumulated weights for sample probability
        self._weight_sum += weight - self._weights[self._next_ix]
        self._weights[self._next_ix] = weight

        # overwrite buffer entry
        self._prev_state_buffer[self._next_ix] = experience.old_state
        self._action_buffer[self._next_ix] = experience.action
        self._reward_buffer[self._next_ix] = experience.reward
        self._next_state_buffer[self._next_ix] = experience.new_state
        # cycle buffer pointer
        self._next_ix += 1
        if self._next_ix == self._capacity:
            self._next_ix = 0

        if self._cnt != self._capacity:
            self._cnt += 1

    def sample_experiences(self, size: int | Tuple, replace: bool) -> dir:
        # maybe include priority selection for buy/sell experiences?

        cnt = size[-1] if isinstance(size, Tuple) else size

        if self._cnt == 0:
            raise Exception("can not sample on empty replay buffer")
        if self._cnt < cnt and not replace:
            raise Exception("not enough samples in replace buffer")

        choices = np.random.choice(range(self._capacity),
                                   size,
                                   replace=replace,
                                   p=self._weights / self._weight_sum)
        return {
            'prev_state': self._prev_state_buffer[choices],
            'action': self._action_buffer[choices],
            'reward': self._reward_buffer[choices],
            'next_state': self._next_state_buffer[choices],
        }


class TestReplayBuffer(unittest.TestCase):

    def test_multidim_sample(self):
        rb = ReplayBuffer(128)
        for i in range(128):
            rb.add_experience(Experience([i], i, i, [i+1]), 1)
        res = rb.sample_experiences((16, 64), True)['prev_state']
        self.assertEqual((16, 64), res.shape)

    def test_reward_is_properly_processed(self):
        rb = ReplayBuffer(64)
        rb.add_experience(Experience(0, 0, .42, 0), 1)
        self.assertEqual(rb.get_experiences()[0].reward, .42)

    def test_add_experiences(self):
        def sort(ar):
            return sorted(ar, key=lambda e: e[1])

        test_size = 640
        replay_buffer = ReplayBuffer(test_size)

        #
        # add 2*TEST_SIZE entries and assert resulting buffer after each add
        #
        check_list = []
        for i in range(2 * test_size):
            new_exp = Experience({'value': i}, i, 0, {'value': i + 1})

            # update comparison list
            check_list += [new_exp]
            if len(check_list) > test_size:
                del check_list[0]

            replay_buffer.add_experience(new_exp, 1)
            result = replay_buffer.get_experiences()
            self.assertListEqual(sort(check_list), sort(result))
            self.assertEqual(len(check_list), len(replay_buffer))

        #
        # assert that weights are respected while sampling
        #
        sample_num = 10000
        replay_buffer.clear()
        vals = [{'val': 1, 'weight': 5},
                {'val': 2, 'weight': 10},
                {'val': 3, 'weight': 20},
                {'val': 4, 'weight': 7}]
        total_weight = sum(map(lambda e: e['weight'], vals))
        for v in vals:
            new_exp = Experience((0,), v['val'], v['weight'], (1,))
            replay_buffer.add_experience(new_exp, new_exp.reward)
        experiences = replay_buffer.sample_experiences(sample_num, True)
        samples = experiences['action']

        for v in vals:
            num_with_value = samples[samples == v['val']].shape[0]
            frequency = num_with_value / samples.shape[0]
            self.assertAlmostEqual(frequency, v['weight']/total_weight, 1)

        # assert that weights are respected while sampling without replace
        buffer_size = sample_num * 10
        replay_buffer = ReplayBuffer(buffer_size)
        for v in vals:
            for _ in range(buffer_size//len(vals)):
                replay_buffer.add_experience(Experience(0, v['val'], 0, 0),
                                             v['weight'])

        experiences = replay_buffer.sample_experiences(sample_num, False)
        samples = experiences['action']
        for v in vals:
            num_with_value = samples[samples == v['val']].shape[0]
            frequency = num_with_value / samples.shape[0]
            self.assertAlmostEqual(frequency, v['weight']/total_weight, 1)

        experiences = replay_buffer.sample_experiences(buffer_size, False)
        samples = experiences['action']
        for v in vals:
            num_with_value = samples[samples == v['val']].shape[0]
            frequency = num_with_value / samples.shape[0]
            self.assertAlmostEqual(frequency, 1/len(vals), 1)

