from typing import Dict
from unittest import TestCase

import numpy as np
from torch import Tensor

from aithena.core.data.data_provider import Sample
from aithena.core.qlearning.q_arbiter import Arbiter
from aithena.core.simulation.sample_provider import SampleProvider
from aithena.core.simulation.simulation_environment import SimulationEnvironment


class MockSim(SimulationEnvironment):
    def __init__(self):
        pass

    def get_initial_context(self):
        return {}

    def perform_transition(self, samples, context, action, mode):
        return context

    def on_transition(self, context, action, mode):
        return context

    def on_episode_start(self, context, mode):
        return context

    def on_new_samples(self, samples, context, mode):
        return context

    def calculate_reward(self, old_state, new_state, action, mode):
        return 0


class MockAgent(Arbiter):

    def decide(self, state, training):
        return 0


class MockTrainer:
    def perform_exploration(self, cnt, provide_experience):
        pass

    def perform_training(self, batch_size, batch_cnt, cuda):
        return 0, 0


class MockProvider(SampleProvider):
    def __init__(self):
        self.episode_complete = False
        self.empty = False

    def reset(self, type):
        pass

    def current_episode_complete(self):
        return self.episode_complete

    def advance(self) -> Dict[str, Sample]:
        if self.episode_complete:
            self.episode_complete = False
        if self.empty:
            return None
        return {'a': Sample(Tensor(np.random.rand(3, 4)), {'t': "foo"}),
                'b': Sample(Tensor(np.random.rand(10)), {'t': "bar"})}


class TestTrainingManager(TestCase):
    def test_training_manager(self):
        pass
