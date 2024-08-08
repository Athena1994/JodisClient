
from typing import List, NamedTuple, Tuple
import numpy as np


class Experience(NamedTuple):
    old_state: object
    action: int
    reward: float
    new_state: object


class ReplayBuffer:

    def __init__(self, size: int, input_tensor_dict: bool):
        self._reward_buffer = np.array([.0] * size)
        self._action_buffer = np.array([-1] * size)

        self._use_dict_state = input_tensor_dict

        if not self._use_dict_state:
            self._prev_state_buffer = np.array([None]*size)
            self._next_state_buffer = np.array([None]*size)
        else:
            self._prev_state_buffer = None
            self._next_state_buffer = None

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

        if self._capacity == self._cnt:
            self._next_ix = np.random.randint(0, self._cnt)

        # keep track of cumulated weights for sample probability
        self._weight_sum += weight - self._weights[self._next_ix]
        self._weights[self._next_ix] = weight

        # overwrite buffer entry
        if self._use_dict_state:
            if self._prev_state_buffer is None:
                shapes = {k: experience.old_state[k].shape
                          for k in experience.old_state}
                for k in shapes:
                    if shapes[k][0] == 1:
                        shapes[k] = shapes[k][1:]
                shapes = {k: (self._capacity, *shapes[k])
                          for k in shapes}

                self._prev_state_buffer \
                    = {k: np.zeros(shapes[k]) for k in shapes}
                self._next_state_buffer \
                    = {k: np.zeros(shapes[k]) for k in shapes}

            for k in experience.old_state:
                self._prev_state_buffer[k][self._next_ix, ...] \
                    = experience.old_state[k]
                self._next_state_buffer[k][self._next_ix, ...] \
                    = experience.new_state[k]
        else:
            self._prev_state_buffer[self._next_ix] = experience.old_state
            self._next_state_buffer[self._next_ix] = experience.new_state

        self._action_buffer[self._next_ix] = experience.action
        self._reward_buffer[self._next_ix] = experience.reward

        # cycle buffer pointer
        if self._cnt != self._capacity:
            self._cnt += 1
            self._next_ix += 1

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

        if self._use_dict_state:
            return {
                'prev_state': {k: self._prev_state_buffer[k][choices]
                               for k in self._prev_state_buffer},
                'action': self._action_buffer[choices],
                'reward': self._reward_buffer[choices],
                'next_state': {k: self._next_state_buffer[k][choices]
                               for k in self._next_state_buffer},
            }
        else:
            return {
                'prev_state': self._prev_state_buffer[choices],
                'action': self._action_buffer[choices],
                'reward': self._reward_buffer[choices],
                'next_state': self._next_state_buffer[choices],
            }
