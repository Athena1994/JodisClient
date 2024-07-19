import numpy as np
import unittest

from core.data.data_provider import ChunkType, Sample
from core.qlearning.q_arbiter import Arbiter

from core.simulation.experience_provider import ExperienceProvider
from core.simulation.sample_provider import SampleProvider
from core.simulation.simulation_environment import SimulationEnvironment, State
from core.simulation.state_manager import StateManager
from core.simulation.test.mockup import DummyChunkProvider, DummyContProvider


class DummyAgent(Arbiter):
    def __init__(self, testcase: unittest.TestCase) -> None:
        self._next_action = 0
        self._test_case = testcase
        self._expect_explore = False

        self._expect_call = False
        self._was_called = False

    def assert_called(self):
        self._test_case.assertTrue(self._was_called)
        self._was_called = False

    def set_next(self, action: int):
        self._next_action = action

    def decide(self, state: object, explore: bool) -> np.array:
        if not self._expect_call:
            self._test_case.fail("Unexpected call")
        self._test_case.assertEqual(explore, self._expect_explore)

        return np.array([self._next_action])


class DummyEnvironment(SimulationEnvironment):
    def __init__(self,
                 testcase: unittest.TestCase,
                 init_context) -> None:
        self._test_case = testcase

        self.expected_action = None
        self.expected_mode = None

        self.expected_old_state = None
        self.expected_new_state = None

        self.transition_action = None
        self.episode_start_action = None
        self.action_action = None

        self.next_reward = None

        self._on_action_result = None

        self._init_context = init_context

    def get_initial_context(self):
        return self._init_context

    def perform_transition(self, samples, context, action, mode):
        self._test_case.assertDictEqual(self.expected_old_state.context,
                                        context)
        self._test_case.assertDictEqual(self.expected_old_state.samples,
                                        samples)
        self._test_case.assertEqual(self.expected_action, action)
        self._test_case.assertEqual(self.expected_mode, mode)
        c = self.transition_action(context)
        return c

    def calculate_reward(self, old_state, new_state, action):
        self._test_case.assertDictEqual(self.expected_old_state.context,
                                        old_state.context)
        self._test_case.assertDictEqual(self.expected_old_state.samples,
                                        old_state.samples)
        self._test_case.assertDictEqual(self.expected_new_state.context,
                                        new_state.context)
        self._test_case.assertDictEqual(self.expected_new_state.samples,
                                        new_state.samples)

        self._test_case.assertEqual(self.expected_action, action)

        return self.next_reward

    def on_action(self, context, action, mode):
        self._test_case.assertDictEqual(self.expected_new_state.context,
                                        context)
        self._test_case.assertEqual(self.expected_action, action)
        self._test_case.assertEqual(self.expected_mode, mode)

        self._on_action_result = self.action_action(context)
        return self._on_action_result

    def on_episode_start(self, context):
        return self.episode_start_action(context)


class TestExperienceProvider(unittest.TestCase):

    def test_experience_provider(self):

        states = []
        overwrites = {}

        def mock_update_context(c):
            overwrites['uc'](c)
            states.append(sm.get_state())

        def mock_update_samples(s):
            overwrites['us'](s)
            states.append(sm.get_state())

        def mock_reset_state(c):
            overwrites['rs'](c)
            states.append(sm.get_state())

        init_cont = {'a': 1}

        sm = StateManager()
        overwrites['rs'] = sm.reset_state
        overwrites['uc'] = sm.update_context
        overwrites['us'] = sm.update_samples
        sm.update_context = mock_update_context
        sm.update_samples = mock_update_samples
        sm.reset_state = mock_reset_state

        sim = DummyEnvironment(self, init_cont)
        agent = DummyAgent(self)
        cont = DummyContProvider(self)
        chunk = DummyChunkProvider(self)
        sp = SampleProvider({
            'cont': cont,
            'chunk': chunk
        })

        ep = ExperienceProvider(sm, sp, agent, sim)
        self.assertFalse(ep._running)

        try:
            ep.provide_experience()
            self.fail("Expected Exception")
        except Exception:
            pass

        last_context = init_cont.copy()

        def episode_start(c):
            self.assertEqual(c, last_context)
            c['b'] = 2
            return c

        sim.episode_start_action = episode_start
        cont.next_sample = Sample(None, {'s': 1})
        chunk.next_sample = Sample(None, {'s': 2})

        ep.start(ChunkType.VALIDATION)
        self.assertTrue(ep._running)
        self.assertEqual(cont.last_iter_type, ChunkType.VALIDATION)
        self.assertEqual(chunk.last_iter_type, ChunkType.VALIDATION)

        self.assertDictEqual(sm.get_state().context, {'a': 1, 'b': 2})
        self.assertEqual(len(states), 3)
        self.assertDictEqual(states[0].context, {'a': 1})
        self.assertIsNone(states[0].samples)
        self.assertDictEqual(states[1].context, {'a': 1, 'b': 2})
        self.assertIsNone(states[1].samples)
        self.assertDictEqual(states[2].context, {'a': 1, 'b': 2})
        self.assertDictEqual(states[2].samples, {'cont': cont.next_sample,
                                                 'chunk': chunk.next_sample})

        last_context = states[2].context.copy()

        def transition(c):
            c = c.copy()
            c['b'] = 5
            return c

        def after_action(c):
            c['b'] = 3
            return c

        cont.expected_update_sample = Sample(None, {'s': 5})
        cont_samples = [Sample(None, {'s': 5}), Sample(None, {'s': 6})]
        chunk_sample = Sample(None, {'s': 6})
        cont.next_sample = cont_samples
        chunk.next_sample = chunk_sample
        agent.set_next(1)
        agent._expect_call = True
        agent._expect_explore = False
        sim.expected_action = 1
        sim.expected_mode = ChunkType.VALIDATION
        sim.next_reward = 0.5
        sim.action_action = after_action
        sim.transition_action = transition
        sim.expected_old_state = states[-1]
        sim.expected_new_state = State({'cont': cont_samples[0],
                                        'chunk': chunk_sample},
                                       {'a': 1, 'b': 5})
        old_state = states[-1].copy()
        for s in states:
            print(s)
        states.clear()
        exp, w = ep.provide_experience()

        self.assertEqual(w, 1)
        self.assertDictEqual(exp.old_state.context, old_state.context)
        self.assertDictEqual(exp.old_state.samples, old_state.samples)
        self.assertEqual(exp.action, 1)
        self.assertEqual(exp.reward, 0.5)
        self.assertDictEqual(exp.new_state.context, {'a': 1, 'b': 5})
        self.assertDictEqual(exp.new_state.samples, {'cont': cont_samples[0],
                                                     'chunk': chunk_sample})

        self.assertEqual(len(states), 4)
        print()
        for s in states:
            print(s)

        self.assertDictEqual(states[0].context, {'a': 1, 'b': 5})
        self.assertIsNone(states[0].samples)
        self.assertDictEqual(states[1].context, {'a': 1, 'b': 5})
        self.assertDictEqual(states[1].samples, {'cont': cont_samples[0],
                                                 'chunk': chunk_sample})
        self.assertDictEqual(states[2].context, {'a': 1, 'b': 3})
        self.assertDictEqual(states[2].samples, {'cont': cont_samples[0],
                                                 'chunk': chunk_sample})
        self.assertDictEqual(states[3].context, {'a': 1, 'b': 3})
        self.assertDictEqual(states[3].samples, {'cont': cont_samples[1],
                                                 'chunk': chunk_sample})

        states.clear()

        # Test episode end
        chunk.reader_exhausted = True
        chunk.last_chunk_reached = False
        chunk_samples = [Sample(None, {'s': 7}), Sample(None, {'s': 8})]
        cont_samples = [Sample(None, {'s': 7}), Sample(None, {'s': 8}),
                        Sample(None, {'s': 9})]
        cont.next_sample = cont_samples
        chunk.next_sample = chunk_samples

        def esa(c):
            c['c'] = 4
            return c
        sim.episode_start_action = esa

        sim.expected_old_state = State({'cont': cont_samples[0],
                                        'chunk': chunk_samples[0]},
                                       {'a': 1, 'b': 3, 'c': 4})
        sim.expected_new_state = State({'cont': cont_samples[1],
                                        'chunk': chunk_samples[1]},
                                       {'a': 1, 'b': 5, 'c': 4})
        cont.expected_update_sample = cont_samples[1]

        exp, _ = ep.provide_experience()
        self.assertDictEqual(exp.old_state.context, {'a': 1, 'b': 3, 'c': 4})
        self.assertDictEqual(exp.old_state.samples, {'cont': cont_samples[0],
                                                     'chunk': chunk_samples[0]})
        self.assertDictEqual(exp.new_state.context, {'a': 1, 'b': 5, 'c': 4})
        self.assertDictEqual(exp.new_state.samples, {'cont': cont_samples[1],
                                                     'chunk': chunk_samples[1]})

        print()
        for s in states:
            print(s)
        self.assertEqual(len(states), 6)
        self.assertDictEqual(states[0].context, {'a': 1, 'b': 3, 'c': 4})
        self.assertIsNone(states[0].samples)
        self.assertDictEqual(states[1].context, {'a': 1, 'b': 3, 'c': 4})
        self.assertDictEqual(states[1].samples, {'cont': cont_samples[0],
                                                 'chunk': chunk_samples[0]})
        self.assertDictEqual(states[2].context, {'a': 1, 'b': 5, 'c': 4})
        self.assertIsNone(states[2].samples)
        self.assertDictEqual(states[3].context, {'a': 1, 'b': 5, 'c': 4})
        self.assertDictEqual(states[3].samples, {'cont': cont_samples[1],
                                                 'chunk': chunk_samples[1]})
        self.assertDictEqual(states[4].context, {'a': 1, 'b': 3, 'c': 4})
        self.assertDictEqual(states[4].samples, {'cont': cont_samples[1],
                                                 'chunk': chunk_samples[1]})
        self.assertDictEqual(states[5].context, {'a': 1, 'b': 3, 'c': 4})
        self.assertDictEqual(states[5].samples, {'cont': cont_samples[2],
                                                 'chunk': chunk_samples[1]})

        # test end of data
        chunk.reader_exhausted = True
        chunk.last_chunk_reached = True
        res = ep.provide_experience()
        self.assertIsNone(res)
        self.assertFalse(ep._running)
