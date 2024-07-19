
from typing import List, NamedTuple, Tuple
import numpy as np


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
