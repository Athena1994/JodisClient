

import copy
import unittest

import numpy as np

from core.data.data_provider import ChunkType
from core.simulation.simulation_environment import State
from program.exchange_manager import ExchangeDirection, Exchanger
from program.trading_environment import ActionType, TradingEnvironment


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
            te.on_action(init_context.copy(),
                         ActionType.BUY, ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.HOLD, ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.SELL, ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.BUY, ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.HOLD, ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.SELL, ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.BUY, ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.HOLD, ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_action(init_context.copy(),
                         ActionType.SELL, ChunkType.TRAINING),
            init_context)

        # --- test on invalid action

        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.BUY, ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.HOLD, ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.SELL, ChunkType.TEST),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.BUY, ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.HOLD, ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.SELL, ChunkType.VALIDATION),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.BUY, ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.HOLD, ChunkType.TRAINING),
            init_context)
        self.assertDictEqual(
            te.on_action(ia_context.copy(),
                         ActionType.SELL, ChunkType.TRAINING),
            init_context)

        # --- test on invalid action and changed balance

        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.BUY, ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.HOLD, ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.SELL, ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.BUY, ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.HOLD, ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.SELL, ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.BUY, ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.HOLD, ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_action(ia_b_context.copy(),
                         ActionType.SELL, ChunkType.TRAINING),
            init_context)

        # --- test on changed balance

        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.BUY, ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.HOLD, ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.SELL, ChunkType.TEST),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.BUY, ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.HOLD, ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.SELL, ChunkType.VALIDATION),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.BUY, ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.HOLD, ChunkType.TRAINING),
            b_context)
        self.assertDictEqual(
            te.on_action(b_context.copy(),
                         ActionType.SELL, ChunkType.TRAINING),
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
