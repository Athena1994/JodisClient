import numpy as np
import unittest

from torch import Tensor

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
        self.new_samples_action = None

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

    def calculate_reward(self, old_state, new_state, action, mode):
        self._test_case.assertDictEqual(self.expected_old_state.context,
                                        old_state.context)
        self._test_case.assertDictEqual(self.expected_old_state.samples,
                                        old_state.samples)
        self._test_case.assertDictEqual(self.expected_new_state.context,
                                        new_state.context)
        self._test_case.assertDictEqual(self.expected_new_state.samples,
                                        new_state.samples)
        self._test_case.assertEqual(self.expected_mode, mode)

        self._test_case.assertEqual(self.expected_action, action)

        return self.next_reward

    def on_action(self, context, action, mode):
        self._test_case.assertDictEqual(self.expected_new_state.context,
                                        context)
        self._test_case.assertEqual(self.expected_action, action)
        self._test_case.assertEqual(self.expected_mode, mode)

        self._on_action_result = self.action_action(context)
        return self._on_action_result

    def on_episode_start(self, context, mode):
        self._test_case.assertEqual(self.expected_mode, mode)
        return self.episode_start_action(context)

    def on_new_samples(self, samples, context, mode):
        self._test_case.assertEqual(self.expected_mode, mode)
        return self.new_samples_action(context)


class TestExperienceProvider(unittest.TestCase):

    def test_experience_provider(self):

        # --- prepare state manager
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

        sm = StateManager()
        overwrites['rs'] = sm.reset_state
        overwrites['uc'] = sm.update_context
        overwrites['us'] = sm.update_samples
        sm.update_context = mock_update_context
        sm.update_samples = mock_update_samples
        sm.reset_state = mock_reset_state

        # --- prepare simulation environment

        class AVals:
            EPISODE_START = 0
            NEW_SAMPLE = 1
            ON_ACTION = 2
            INIT_STATE = 3
            TRANSITION = 4

        def new_samples(c):
            c['a'] = AVals.NEW_SAMPLE
            c['fetch_cnt'] += 1
            return c

        def episode_start(c):
            c['a'] = AVals.EPISODE_START
            return c

        def transition(c):
            c = c.copy()
            c['a'] = AVals.TRANSITION
            return c

        def after_action(c):
            c['a'] = AVals.ON_ACTION
            return c

        init_cont = {'a': AVals.INIT_STATE, 'fetch_cnt': 0}

        sim = DummyEnvironment(self, init_cont)
        sim.episode_start_action = episode_start
        sim.new_samples_action = new_samples
        sim.transition_action = transition
        sim.action_action = after_action
        sim.expected_mode = ChunkType.VALIDATION
        sim.expected_action = 1
        sim.next_reward = 0.5

        # --- prepare agent and data providers

        agent = DummyAgent(self)
        agent.set_next(1)

        cont = DummyContProvider(self)
        cont_samples = [Sample(None, {'co': i}) for i in range(10)]
        cont.next_sample = cont_samples

        chunk = DummyChunkProvider(self)
        chunk_samples = [Sample(None, {'ch': i}) for i in range(10)]
        chunk.next_sample = chunk_samples

        sp = SampleProvider({
            'cont': cont,
            'chunk': chunk
        })

        # --- start tests --------------------------------------

        ep = ExperienceProvider(sm, sp, agent, sim)

        # assert session must be started first
        self.assertFalse(ep._running)
        try:
            ep.provide_experience()
            self.fail("Expected Exception")
        except Exception:
            pass

        # test start
        expected_states = [
            State(None,
                  {'a': AVals.INIT_STATE, 'fetch_cnt': 0}),

            State(None,
                  {'a': AVals.EPISODE_START, 'fetch_cnt': 0}),

            State({'cont': cont_samples[0], 'chunk': chunk_samples[0]},
                  {'a': AVals.EPISODE_START, 'fetch_cnt': 0}),

            State({'cont': cont_samples[0], 'chunk': chunk_samples[0]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 1}),

            State({'cont': cont_samples[1], 'chunk': chunk_samples[0]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 1}),
        ]
        ep.start(ChunkType.VALIDATION)
        self.assertTrue(ep._running)
        self.assertEqual(cont.last_iter_type, ChunkType.VALIDATION)
        self.assertEqual(chunk.last_iter_type, ChunkType.VALIDATION)
        self.assertListEqual(states, expected_states)

        # test transition

        expected_states = [
            State(None,
                  {'a': AVals.TRANSITION, 'fetch_cnt': 1}),

            State({'cont': cont_samples[2], 'chunk': chunk_samples[1]},
                  {'a': AVals.TRANSITION, 'fetch_cnt': 1}),

            State({'cont': cont_samples[2], 'chunk': chunk_samples[1]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 2}),

            State({'cont': cont_samples[3], 'chunk': chunk_samples[1]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 2}),

            State({'cont': cont_samples[3], 'chunk': chunk_samples[1]},
                  {'a': AVals.ON_ACTION, 'fetch_cnt': 2}),

            State({'cont': cont_samples[4], 'chunk': chunk_samples[1]},
                  {'a': AVals.ON_ACTION, 'fetch_cnt': 2}),
        ]

        agent._expect_call = True
        agent._expect_explore = False
        sim.expected_old_state = states[-1]
        sim.expected_new_state = expected_states[3]
        states.clear()

        exp, w = ep.provide_experience()

        self.assertEqual(w, 1)
        self.assertEqual(exp.action, 1)
        self.assertEqual(exp.reward, 0.5)
        self.assertEqual(exp.old_state, sim.expected_old_state)
        self.assertEqual(exp.new_state, sim.expected_new_state)
        self.assertListEqual(states, expected_states)

        # test episode end

        chunk.reader_exhausted = True
        chunk.last_chunk_reached = False
        expected_states = [
            # 0
            State(None,
                  {'a': AVals.EPISODE_START, 'fetch_cnt': 2}),
            # 1
            State({'cont': cont_samples[5], 'chunk': chunk_samples[2]},
                  {'a': AVals.EPISODE_START, 'fetch_cnt': 2}),
            # 2
            State({'cont': cont_samples[5], 'chunk': chunk_samples[2]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 3}),
            # 3
            State({'cont': cont_samples[6], 'chunk': chunk_samples[2]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 3}),
            # 4
            State(None,
                  {'a': AVals.TRANSITION, 'fetch_cnt': 3}),
            # 5
            State({'cont': cont_samples[7], 'chunk': chunk_samples[3]},
                  {'a': AVals.TRANSITION, 'fetch_cnt': 3}),
            # 6
            State({'cont': cont_samples[7], 'chunk': chunk_samples[3]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 4}),
            # 7
            State({'cont': cont_samples[8], 'chunk': chunk_samples[3]},
                  {'a': AVals.NEW_SAMPLE, 'fetch_cnt': 4}),
            # 8
            State({'cont': cont_samples[8], 'chunk': chunk_samples[3]},
                  {'a': AVals.ON_ACTION, 'fetch_cnt': 4}),
            # 9
            State({'cont': cont_samples[9], 'chunk': chunk_samples[3]},
                  {'a': AVals.ON_ACTION, 'fetch_cnt': 4}),
        ]
        sim.expected_old_state = expected_states[3]
        sim.expected_new_state = expected_states[7]
        states.clear()

        ep.provide_experience()

        self.assertListEqual(states, expected_states)

        # test end of data

        chunk.reader_exhausted = True
        chunk.last_chunk_reached = True
        res = ep.provide_experience()
        self.assertIsNone(res)
        self.assertFalse(ep._running)
