

import copy
from dataclasses import dataclass
import datetime
import json
import unittest

import numpy as np
from torch import Tensor
import pandas as pd
import torch

from aithena.core.data.assets.asset_manager import AssetManager
from aithena.core.data.data_provider import ChunkType, Sample
from aithena.core.data.loader.ohcl_loader import OHCLLoader
from aithena.core.data.normalizer import Normalizer
from aithena.core.nn.dynamic_nn import DynamicNN
from aithena.core.qlearning.dqn_trainer import DQNTrainer
from aithena.core.qlearning.q_arbiter import DeepQFunction, EpsilonGreedyArbiter
from aithena.core.simulation.experience_provider import ExperienceProvider
from aithena.core.simulation.sample_provider import SampleProvider
from aithena.core.simulation.simulation_environment import State
from aithena.core.simulation.state_manager import StateManager, StateProvider
from aithena.core.simulation.data_classes import TransitionResult
from aithena.core.simulation.utils import perform_transition
from aithena.mock.mockup import MockAgent, MockChunkProvider
from core.exchange_manager \
    import ExchangeDirection, Exchanger, StateSourcedExchanger
from core.trading_environment import ActionType, TradingEnvironment
from aithena.utils import benchmark


class DummyExchanger(Exchanger):

    def __init__(self, test: unittest.TestCase):
        self.test = test

        self.receipt_currency_balance = None
        self.receipt_asset_balance = None

        self.expected_currency = "EUR"
        self.expected_asset = "BTC"
        self.expected_action = None
        self.expected_amount = None

    def prepare_exchange(self,
                         currency: str, asset: str,
                         amount: float, action: ExchangeDirection):
        self.test.assertEqual(currency, self.expected_currency)
        self.test.assertEqual(asset, self.expected_asset)
        self.test.assertEqual(amount, self.expected_amount)
        if self.expected_action is not None:
            self.test.assertEqual(action, self.expected_action)
        return Exchanger.Receipt(currency, asset,
                                 self.receipt_currency_balance,
                                 self.receipt_asset_balance,
                                 1, 0)


class TestSimulationEnvironment(unittest.TestCase):

    class CONSTANTS:
        initial_balance = 1000
        buy_reward = 1
        hold_penalty = 2
        wait_penalty = 3
        invalid_action_penalty = 4

    ACTION_TYPES = [ActionType.BUY, ActionType.HOLD, ActionType.SELL]
    CHUNK_TYPES = [ChunkType.TRAINING, ChunkType.VALIDATION, ChunkType.TEST]

    constants = {
        'initial_balance': CONSTANTS.initial_balance,
        'buy_reward': CONSTANTS.buy_reward,
        'hold_penalty': CONSTANTS.hold_penalty,
        'wait_penalty': CONSTANTS.wait_penalty,
        'invalid_action_penalty': CONSTANTS.invalid_action_penalty
    }

    def test_init(self):
        sm = DummyExchanger(self)

        with self.assertRaises(Exception):
            te = TradingEnvironment({}, sm)

        with self.assertRaises(Exception):
            te = TradingEnvironment({'constants': {}}, sm)

        te = TradingEnvironment(TradingEnvironment.Config.from_dict(
            {'constants': self.constants}), sm)

        self.assertDictEqual(te.get_initial_context(), {
            'asset': {'BTC': 0, 'EUR': 1000},
            'position_open': False,
            'buy_in_price': None,
            'balance': 1
        })

        self.assertEqual(te._get_constant('initial_balance'), 1000)
        with self.assertRaises(Exception):
            te._get_constant('unknown')

    def test_on_action(self):
        sm = DummyExchanger(self)
        te = TradingEnvironment(TradingEnvironment.Config.from_dict(
            {'constants': self.constants}), sm)

        init_context = te.get_initial_context()

        ia_context = init_context.copy()
        ia_context.update({
            'invalid_action': True
        })

        ia_b_context = init_context.copy()
        ia_b_context.update({
            'invalid_action': True,
            'asset': {'BTC': 0, 'EUR': 500}
        })
        b_context = init_context.copy()
        b_context.update({
            'asset': {'BTC': 0, 'EUR': 500}
        })

        # --- test on inital context -> no changes expected

        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, init_context.copy())),
							 ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, init_context.copy())),
							 ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, init_context.copy())),
							 ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, init_context.copy())),
							 ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, init_context.copy())),
							 ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, init_context.copy())),
							 ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, init_context.copy())),
							 ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, init_context.copy())),
							 ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, init_context.copy())),
							 ChunkType.TRAINING),
            init_context)

        # --- test on invalid action

        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, ia_context.copy())),
							 ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, ia_context.copy())),
							 ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, ia_context.copy())),
							 ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, ia_context.copy())),
							 ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, ia_context.copy())),
							 ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, ia_context.copy())),
							 ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, ia_context.copy())),
							 ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, ia_context.copy())),
							 ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, ia_context.copy())),
							 ChunkType.TRAINING),
            init_context)

        # --- test on invalid action and changed balance
        te.on_transition(TransitionResult(State()))
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, ia_b_context.copy())),
							 ChunkType.TRAINING),
            init_context)

        # --- test on changed balance

        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
                                              State(None, b_context.copy())),
                             ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
                                              State(None, b_context.copy())),
							 ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
                                              State(None, b_context.copy())),
							 ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
                                              State(None, b_context.copy())),
							 ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, b_context.copy())),
							 ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, b_context.copy())),
							 ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.BUY, 0,
											  State(None, b_context.copy())),
							 ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.HOLD, 0,
											  State(None, b_context.copy())),
							 ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_transition(TransitionResult(ActionType.SELL, 0,
											  State(None, b_context.copy())),
							 ChunkType.TRAINING),
            init_context)

    def test_calculate_reward(self):

        sm = DummyExchanger(self)
        te = TradingEnvironment(TradingEnvironment.Config.from_dict(
            {'constants': self.constants}), sm)

        init_context = te.get_initial_context()

        invalid_action_state = State(None, init_context.copy())
        invalid_action_state.context.update({
            'invalid_action': True
        })

        closed_state = State(None, init_context.copy())
        closed_state.context.update({
            'position_open': False,
            'balance': 1.
        })

        open_state = State(None, init_context.copy())
        open_state.context.update({
            'position_open': True,
            'balance': 1.1
        })

        for chunk_type in self.CHUNK_TYPES:

            # --- test on invalid action
            for action in self.ACTION_TYPES:
                self.assertEqual(
                    te.calculate_reward(closed_state, invalid_action_state,
                                        action, chunk_type),
                    self.CONSTANTS.invalid_action_penalty)
                self.assertEqual(
                    te.calculate_reward(open_state, invalid_action_state,
                                        action, chunk_type),
                    self.CONSTANTS.invalid_action_penalty)

            # --- HOLD
            self.assertEqual(
                te.calculate_reward(closed_state, closed_state,
                                    ActionType.HOLD, chunk_type),
                self.CONSTANTS.wait_penalty)
            self.assertEqual(
                te.calculate_reward(open_state, open_state,
                                    ActionType.HOLD, chunk_type),
                self.CONSTANTS.hold_penalty)

            # --- BUY
            self.assertEqual(
                te.calculate_reward(closed_state, open_state,
                                    ActionType.BUY, chunk_type),
                self.CONSTANTS.buy_reward)

            # --- SELL
            self.assertEqual(
                te.calculate_reward(open_state, closed_state,
                                    ActionType.SELL, chunk_type),
                np.log(1.1) / np.log(1.05))

    def test_on_episode_start(self):
        sm = DummyExchanger(self)
        te = TradingEnvironment(TradingEnvironment.Config.from_dict(
            {'constants': self.constants}), sm)

        init_context = te.get_initial_context()

        closed_position = {
            'position_open': False,
            'asset': {'BTC': 0, 'EUR': 500},
            'buy_in_price': None,
            'balance': 1
        }
        open_position = {
            'position_open': True,
            'asset': {'BTC': 1, 'EUR': 0},
            'buy_in_price': 500,
            'balance': 0.5
        }

        for ct in [ChunkType.TRAINING, ChunkType.VALIDATION]:
            self.assertDictEqual(
                te.on_episode_start({'a': 0}, ct),
                init_context)
            self.assertDictEqual(
                te.on_episode_start(open_position, ct),
                init_context)
            self.assertDictEqual(
                te.on_episode_start(closed_position, ct),
                init_context)

        self.assertDictEqual(
            te.on_episode_start(open_position, ChunkType.TEST),
            closed_position)
        self.assertDictEqual(
            te.on_episode_start(closed_position, ChunkType.TEST),
            open_position)

    def test_on_new_samples(self):
        sm = DummyExchanger(self)
        te = TradingEnvironment(TradingEnvironment.Config.from_dict(
            {'constants': self.constants}), sm)

        sm.expected_action = ExchangeDirection.SELL
        sm.expected_currency = 'EUR'
        sm.expected_asset = 'BTC'
        sm.expected_amount = 2
        sm.receipt_currency_balance = 100

        closed_position = {
            'position_open': False,
            'asset': {'BTC': 0, 'EUR': 500},
            'buy_in_price': None,
            'balance': 1
        }
        open_old_sample = {
            'position_open': True,
            'asset': {'BTC': 2, 'EUR': 0},
            'buy_in_price': 500,
            'balance': 0.5
        }
        open_new_sample = {
            'position_open': True,
            'asset': {'BTC': 2, 'EUR': 0},
            'buy_in_price': 500,
            'balance': 1/5
        }

        for ct in self.CHUNK_TYPES:
            self.assertDictEqual(
                te.on_new_samples(None, closed_position, ct),
                closed_position)
            self.assertDictEqual(
                te.on_new_samples(None, open_old_sample, ct),
                open_new_sample)
            self.assertDictEqual(
                te.on_new_samples(None, open_new_sample, ct),
                open_new_sample)

    def test_transition(self):
        ex = DummyExchanger(self)
        te = TradingEnvironment(TradingEnvironment.Config.from_dict(
            {'constants': self.constants}), ex)

        ex.expected_currency = 'EUR'
        ex.expected_asset = 'BTC'

        closed_position = {
            'position_open': False,
            'asset': {'BTC': 0, 'EUR': 500},
            'buy_in_price': None,
            'balance': 1
        }
        open_after_closed = {
            'position_open': True,
            'asset': {'BTC': 1, 'EUR': 0},
            'buy_in_price': 500,
            'balance': 1
        }
        open_position = {
            'position_open': True,
            'asset': {'BTC': 1, 'EUR': 0},
            'buy_in_price': 500,
            'balance': 0.5
        }

        closed_inv = copy.deepcopy(closed_position)
        closed_inv.update({
            'invalid_action': True
        })
        open_inv = copy.deepcopy(open_position)
        open_inv.update({
            'invalid_action': True
        })

        for ct in self.CHUNK_TYPES:
            # closed: sell -> invalid
            self.assertDictEqual(
                te.perform_transition(None, copy.deepcopy(closed_position),
                                      ActionType.SELL, ct),
                closed_inv)
            # closed: buy -> open
            ex.expected_amount = 500
            ex.receipt_asset_balance = 1
            self.assertDictEqual(
                te.perform_transition(None, copy.deepcopy(closed_position),
                                      ActionType.BUY, ct),
                open_after_closed)
            # closed: hold -> closed
            self.assertDictEqual(
                te.perform_transition(None, copy.deepcopy(closed_position),
                                      ActionType.HOLD, ct),
                closed_position)
            # open: sell -> closed
            ex.expected_amount = 1
            ex.receipt_currency_balance = 500
            self.assertDictEqual(
                te.perform_transition(None, copy.deepcopy(open_position),
                                      ActionType.SELL, ct),
                closed_position)
            # open: buy -> invalid
            self.assertDictEqual(
                te.perform_transition(None, copy.deepcopy(open_position),
                                      ActionType.BUY, ct),
                open_inv)
            # open: hold -> open
            self.assertDictEqual(
                te.perform_transition(None, copy.deepcopy(open_position),
                                      ActionType.HOLD, ct),
                open_position)

    def test_configured_env(self):
        FEE = 0.1
        REM = 1-FEE

        with open('config/simulation.json', 'r') as f:
            sim_config = json.load(f)['environment']
            sim_config = TradingEnvironment.Config.from_dict(sim_config)
        with open('config/agent.json', 'r') as f:
            agent_config = json.load(f)
            input_config = DynamicNN.Config.Input.from_dict(
                agent_config['input']).data[1].config

        norm = Normalizer()
        sm = StateManager()

        asset_provider = MockChunkProvider(self)
        asset_provider.next_sample = Sample(None,
                                            {'close': 100, 'open': 100})

        state_provider = StateProvider(sm, norm, input_config)

        sp = SampleProvider({'time_series': asset_provider,
                            'meta': state_provider})

        agent = MockAgent(self)

        ex = StateSourcedExchanger(StateSourcedExchanger.Config.from_dict({
            'pairs': [{
                'asset': 'BTC',
                'currency': 'EUR',
                'fee': {
                    'relative': FEE,
                    'fixed': 0
                },
                'candle_src': 'time_series'
            }]
        }))

        sim = TradingEnvironment(sim_config, ex)

        @dataclass
        class Context:
            position_open: bool
            balance: float

        def set_followup_price(price):
            if isinstance(price, list):
                asset_provider.next_sample = [
                    Sample(Tensor([0]), {'close': p, 'open': p})
                    for p in price
                ]
            else:
                asset_provider.next_sample =\
                    Sample(Tensor([0]), {'close': price, 'open': price})

        def assert_exp(action: ActionType,
                       reward: int,
                       old_context: Context,
                       new_context: Context):
            agent.set_next(action.value)
            exp = ep.provide_experience().experience

            self.assertEqual(exp.reward, reward)
            self.assertEqual(exp.action, action.value)

            self.assertEqual(exp.old_state['meta'][0][1],
                             np.log(old_context.balance) / np.log(1.10))
            self.assertEqual(exp.old_state['meta'][0][0],
                             1 if old_context.position_open else 0)
            self.assertEqual(exp.new_state['meta'][0][1],
                             np.log(new_context.balance) / np.log(1.10))
            self.assertEqual(exp.new_state['meta'][0][0],
                             1 if new_context.position_open else 0)

            return new_context

        def log(x: float):
            return np.log(x) / np.log(1.10)

        ep = ExperienceProvider(sm, sp, agent, sim)

        set_followup_price(100)

        ep.start(ChunkType.TRAINING)
        # price: 100

        #old_context = Context(position_open=False, balance=1)

        for i in range(2):
            # HOLD on closed;
            #   position_open: False
            #   balance: 1
            # reward: wait_penalty
            assert_exp(action=ActionType.HOLD,
                       reward=-1,
                       old_context=Context(
                            position_open=False,
                            balance=1
                        ),
                       new_context=Context(
                            position_open=False,
                            balance=1
                        ))
            # SELL on closed -> no change;
            #   position_open: False
            #   balance: 1
            # reward: invalid_action_penalty

            assert_exp(action=ActionType.SELL,
                       reward=-100,
                       old_context=Context(
                            position_open=False,
                            balance=1
                        ),
                       new_context=Context(
                            position_open=False,
                            balance=1
                        ))

            set_followup_price(50)
            # BUY on closed
            #   position_open: True
            #   balance: (1-F)^2 * P1 / P0
            # reward: 0
            # new price: 50
            assert_exp(action=ActionType.BUY,
                       reward=0,
                       old_context=Context(
                            position_open=False,
                            balance=1
                        ),
                       new_context=Context(
                            position_open=True,
                            balance=1
                        ))

            # HOLD on open
            # reward: hold_penalty
            # new price: 200
            set_followup_price(200)
            assert_exp(
                action=ActionType.HOLD,
                reward=0,
                old_context=Context(
                    position_open=True,
                    balance=REM**2 * 50 / 100
                ),
                new_context=Context(
                    position_open=True,
                    balance=REM**2 * 50 / 100
                ))

            # BUY on open
            # reward: invalid_action_penalty
            # new price: 500
            set_followup_price(500)
            assert_exp(
                action=ActionType.BUY,
                reward=-100,
                old_context=Context(
                    position_open=True,
                    balance=REM**2 * 200 / 100
                ),
                new_context=Context(
                    position_open=True,
                    balance=REM**2 * 200 / 100
                ))
            # SELL on open
            #   position_open: False
            #   balance: 1
            # reward: log((1-F)**2 * P_old / P0)
            # new price: 100
            set_followup_price(100)
            assert_exp(
                action=ActionType.SELL,
                reward=log(REM**2 * 500 / 100),
                old_context=Context(
                    position_open=True,
                    balance=REM**2 * 500 / 100
                ),
                new_context=Context(
                    position_open=False,
                    balance=1
                ))

    def test_learing_inalid_actions(self):

        class MockLoader(OHCLLoader):
            def get(self,
                    pair: str, interval: str,
                    earliest: datetime = None,
                    last: datetime = None) -> pd.DataFrame:
                # sin-curve with 10000 samples and wavelength of 1000 samples
                x = 2 * np.pi * np.arange(N) / T
                base_line = np.sin(x)
                data = np.concatenate(
                    [base_line + np.random.normal(0, 0.1, (N, 4)),
                     np.random.random((N, 1))], axis=1)

                return pd.DataFrame({
                    'open': data[0],
                    'high': data[1],
                    'low': data[2],
                    'close': data[3],
                    'volume': data[4],
                    'time': pd.timedelta_range('2024-08-04 12:00', periods=N)
                })

        FEE = 0
        N = 10000
        T = 100

        # --- prepare config

        with open('config/simulation.json', 'r') as f:
            sim_config = json.load(f)['environment']
            sim_config = TradingEnvironment.Config.from_dict(sim_config)
        with open('config/data.json', 'r') as f:
            data_config = json.load(f)
            data_config = AssetManager.Config.from_dict(data_config)
        with open('config/agent.json', 'r') as f:
            agent_config = json.load(f)
            input_config \
                = DynamicNN.Config.Input.from_dict(agent_config['input'])
            nn_config = DynamicNN.Config.from_dict(agent_config['nn'])

        with open('config/training.json', 'r') as f:
            trainer_cfg = json.load(f)
            trainer_cfg = DQNTrainer.Config.from_dict(trainer_cfg)

        with open('config/data.json', 'r') as f:
            data_config = json.load(f)
            data_config = AssetManager.Config.from_dict(data_config)

        data_config.chunks.chunk_size = 510
        data_config.chunks.test_chunk_cnt = 1

        # --- init sample provider
        am = AssetManager(data_config, False)
        am._cache_asset = lambda _: None
        am._create_source = lambda _: MockLoader()

        norm = Normalizer()
        sm = StateManager()

        sp = SampleProvider.from_config(norm, am, sm, input_config)

        # --- init environment

        ex = StateSourcedExchanger(StateSourcedExchanger.Config.from_dict({
            'pairs': [{
                'asset': 'BTC',
                'currency': 'EUR',
                'fee': {
                    'relative': FEE,
                    'fixed': 0
                },
                'candle_src': 'time_series'
            }]
        }))
        sim = TradingEnvironment(sim_config, ex)

        # --- prepare trainer

        nn = DynamicNN(nn_config, input_config).cuda()
        trainer = DQNTrainer.from_config(nn, trainer_cfg)

        agent = EpsilonGreedyArbiter(DeepQFunction(nn))

        exp = ExperienceProvider(sm, sp, agent, sim)

        class Provider:
            def __init__(self) -> None:
                self.samples = []
                self.ix = -1
                self.size = 1000

            def provide(self):
                e = exp.provide_experience()

                if len(self.samples) != self.size:
                    self.samples.append(e)

                self.ix += 1
                if self.ix == self.size:
                    self.ix = 0
                return e

            def validate(self):
                values = {'buy': 0, 'sell': 0, 'hold': 0, 'wait': 0, 'inv': 0,
                          'reward': 0}

                with torch.no_grad():

                    watch = benchmark.Watch()
                    watch.start()
                    for exp, w, r in self.samples:
                        res = perform_transition(
                            agent, sim, sp,
                            r.old_state, ChunkType.VALIDATION,
                            cuda=True)

                        values['reward'] += res.reward

                        if 'invalid_action' in res.new_state.context:
                            values['inv'] += 1
                        elif res.action == ActionType.BUY.value:
                            values['buy'] += 1
                        elif res.action == ActionType.SELL.value:
                            values['sell'] += 1
                        elif res.action == ActionType.HOLD.value:
                            if res.old_state.context['position_open']:
                                values['hold'] += 1
                            else:
                                values['wait'] += 1

                    print(f'Exploration time: {watch.elapsed()} s')
                    print(f'Buy: {values["buy"]}, Sell: {values["sell"]}, '
                        f'Invalid: {values["inv"]}, '
                        f'Hold: {values["hold"]}, Wait: {values["wait"]}')
                    print(f'Total reward: {values["reward"]}')

        prov = Provider()

        # --- begin training
        for j in range(10):
            exp.start(ChunkType.TRAINING)
            i = 0
            while exp.has_next():
                trainer.perform_exploration(10, prov.provide)
                trainer.perform_training(32, 1, True)
                i += 1
                if i % 100 == 0:
                    prov.validate()