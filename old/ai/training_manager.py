import copy
from datetime import datetime
import time
import unittest
from abc import abstractmethod
from typing import NamedTuple, List

import numpy as np
import torch.optim

from old.agent import SimpleIndicatorAgent
from core.qlearning.q_arbiter import QSigArbiter, DQNWrapper
from ai.q_nn import QNN
from core.qlearning.replay_buffer import ReplayBuffer
from core.qlearning.trainer import DQNTrainer
from old.chunk_type import ChunkType
from old.chunks import DataChunk
from data.utils import split_df
from technical_indicators.indicators import IndicatorDescription, Indicator


class InvalidAmountException(Exception):
    def __init__(self, present, requested):
        super().__init__(f"Invalid amount requested! (requested: {requested}"
                         f", present: {present})")


class InvalidParameterException(Exception):
    def __init__(self, desc):
        super().__init__(desc)


class FeesNotCoveredException(Exception):
    def __init__(self, fixed_fee):
        super().__init__("Amount given does not suffice to cover fixed fees! "
                       f"(fixed_fee: {fixed_fee})")


class FundsInsufficientException(Exception):
    def __init__(self, asset, amount, requested):
        super().__init__(f"Insufficient amounts of asset '{asset}' "
                         f"(present: {amount}, requested: {requested})")


class EmptyTradeException(Exception):
    def __init__(self):
        super().__init__("Trading amount must be a positive number.")


class TradingWallet:
    def __init__(self, init_state: dir):
        self._assets = init_state

    def get_asset(self, asset: str):
        return self._assets.get(asset, 0)

    def change_asset(self, asset: str, amount: float):
        if asset not in self._assets:
            self._assets[asset] = 0
        new_amount = self._assets[asset] + amount
        if new_amount < 0:
            raise InvalidAmountException(self._assets[asset], amount)
        self._assets[asset] = new_amount

    def get_state(self):
        return copy.deepcopy(self._assets)


class TradeInfo(NamedTuple):
    currency_asset: str
    currency_amount: float
    stock_asset: str
    stock_amount: str
    base_rate: float
    effective_rate: float
    fixed_fee: float
    relative_fee: float
    total_fee: float
    trade_type: str


def calculate_buy_trade(asset: str, rate: float,
                        currency: str, currency_amount: float,
                        fixed_fee: float, relative_fee: float):

    remaining = currency_amount - fixed_fee
    rel_fee = relative_fee * remaining
    total_fee = fixed_fee + rel_fee
    remaining = currency_amount - total_fee
    if remaining <= 0:
        raise FeesNotCoveredException(fixed_fee)
    amount = remaining / rate
    effective_rate = currency_amount / amount

    return TradeInfo(currency, currency_amount,
                     stock_asset=asset, stock_amount=amount,
                     base_rate=rate, effective_rate=effective_rate,
                     relative_fee=rel_fee, fixed_fee=fixed_fee,
                     total_fee=total_fee,
                     trade_type='buy')


def calculate_sell_trade(asset: str, asset_amount: float, rate: float,
                         currency: str, fixed_fee: float, relative_fee: float):
    currency_amount = asset_amount * rate
    rel_fee = (currency_amount - fixed_fee) * relative_fee
    total_fee = fixed_fee + rel_fee
    remaining = currency_amount - total_fee
    effective_rate = remaining/asset_amount

    return TradeInfo(currency_asset=currency, currency_amount=remaining,
                     stock_asset=asset, stock_amount=asset_amount,
                     base_rate=rate, effective_rate=effective_rate,
                     relative_fee=rel_fee, fixed_fee=fixed_fee,
                     total_fee=total_fee,
                     trade_type='sell')


class TradingBroker:
    def __init__(self, fixed_fee: float, rel_fee_rate: float):
        self._fixed_fee = fixed_fee
        self._rel_fee_rate = rel_fee_rate

    @abstractmethod
    def get_rate(self, asset, currency):
        pass

    def make_trade(self, wallet: TradingWallet, trade_type: str,
                   asset: str, currency: str, amount: float) -> TradeInfo:

        if amount == 0:
            raise EmptyTradeException()
        if amount < 0:
            raise InvalidParameterException("Amount may not be negative!")

        if trade_type == 'buy':
            trade = calculate_buy_trade(asset, self.get_rate(asset, currency),
                                        currency, amount,
                                        self._fixed_fee, self._rel_fee_rate)
            minus_asset, minus_amount, plus_asset, plus_amount \
                = (trade.currency_asset, trade.currency_amount,
                   trade.stock_asset, trade.stock_amount)
        elif trade_type == 'sell':
            trade = calculate_sell_trade(asset, amount,
                                         self.get_rate(asset, currency),
                                         currency,
                                         self._fixed_fee, self._rel_fee_rate)
            minus_asset, minus_amount, plus_asset, plus_amount\
                = (trade.stock_asset, trade.stock_amount,
                   trade.currency_asset, trade.currency_amount)
        else:
            raise NotImplementedError(f'"{trade_type}" not implemented!')

        if plus_amount < 0:
            raise FeesNotCoveredException(self._fixed_fee)

        if minus_amount > wallet.get_asset(minus_asset):
            raise FundsInsufficientException(minus_asset,
                                             minus_amount,
                                             wallet.get_asset(minus_asset))

        wallet.change_asset(minus_asset, -minus_amount)
        wallet.change_asset(plus_asset, plus_amount)

        return trade

class OHCLProvider:
    def __init__(self):
        pass
#    def get_range(self, ):


class DataRequest(NamedTuple):
    asset: str
    currency: str
    interval: str
    indicators: List[Indicator]


class TradingDataProvider:
    """
    A TradingDataProvider can be configured with a list
    """
    def __init__(self, data):


        pass

    def get_data(self, date: datetime, cnt: int) -> np.ndarray:
        pass

class TradingSession:
    def __init__(self, wallet: TradingWallet,
                 data_provider: TradingDataProvider):
        self._wallet
        self._assets = {}

#    def add_asset

class TradingSimulation:
    def provide_experience(self):
        pass





'''
    Splits provided data into training, validation and test set.
    
'''
class TrainingManager:

    def __init__(self, candle_df):

        precursor_len = 60*24
        post_len = 60*24
        min_len = 60*24

        split_list = split_df(candle_df, precursor_len, post_len, min_len, 1 / 7)
        print(split_list)
        tr_splits = [split for split in split_list if split[2] == ChunkType.TRAINING]
        val_splits = [split for split in split_list if split[2] == ChunkType.VALIDATION]
        test_splits = [split for split in split_list if split[2] == ChunkType.TEST]

        tr_dfs = [candle_df.iloc[split[0]:split[1], :] for split in tr_splits]
        val_dfs = [candle_df.iloc[split[0]:split[1], :] for split in val_splits]
        test_dfs = [candle_df.iloc[split[0]:split[1], :] for split in test_splits]

        self._tr_chunks = [DataChunk(df, precursor_len, post_len) for df in tr_dfs]
        self._val_chunks = [DataChunk(df, precursor_len, post_len) for df in val_dfs]
        self._test_chunks = [DataChunk(df, precursor_len, 0) for df in test_dfs]

#        self._tr_sim = [EnvironmentSimulation(chunk)
#        self._eval_sim = [EnvironmentSimulation(chunk)
#                          for chunk in self._val_chunks]
#        self._test_sim = [EnvironmentSimulation(chunk)
#                          for chunk in self._test_chunks]
#
#                        for chunk in self._tr_chunks]
    def _train_cycle(self, replay_buffer: ReplayBuffer):
        pass




    def run(self):

        t = time.time_ns()

        replay_buffer_size = 512*16
        target_network_update_cnt = 200
        episode_cnt = 1000
        explorations_per_episode = 512
        training_batch_size = 512
        mini_batch_cnt = 1

        random_exploration_chance = 0.1
        learning_rate = 0.01
        discount_factor = 0.999

        nn = QNN()

        q_function = DeepQFunction(DQNWrapper(nn))
        arbiter = QSigArbiter(q_function, random_exploration_chance)
        agent = SimpleIndicatorAgent(arbiter)
        replay_buffer = ReplayBuffer(replay_buffer_size)
        optimizer = torch.optim.Adam(nn.parameters(), learning_rate)
        trainer = DQNTrainer(nn, replay_buffer, optimizer,
                             target_network_update_cnt, discount_factor)
        evaluator = ActionEvaluator()

        sim = Simulation(self._tr_sim, agent)
        def provide_experience():
            old_state = sim.next_state()
            action = agent.select_action(old_state)
            reward = evaluator.evaluate(old_state)


#        for episode in range(episode_cnt):
#            trainer.perform_exploration(explorations_per_episode,
#                                        sim.)

 #       env = EnvironmentSimulation()




        trainer.perform_exploration(explorations_per_episode, )


        # q_function.set_env(env)

        print('starting training session')
        # start training session
        for i, chunk in enumerate(self._tr_chunks):
            print(f'training chunk {i+1} / {len(self._tr_chunks)}')
            env.set_chunk(chunk)
            env.reset({'EUR': 1000})
            chunk_iter = env.get_chunk()

            for j, _ in enumerate(chunk_iter):
                env.notify_update()
                if j % 100 == 0:
                    trainer.perform_training()
                    print(f'sample {j + 1} / {len(chunk_iter)} '
                          f'Value: {round(env.get_wallet_worth(), 2)} €\n')

        print('starting eval session')

        print('cycle finished')

        print(f'total time: {((time.time_ns()-t)/10**9)} s')


class TestTradingWallet(unittest.TestCase):

    def test(self):
        wallet = TradingWallet({'d': 32.1})

        self.assertEqual(0, wallet.get_asset('a'))
        wallet.change_asset('a', 1)
        self.assertEqual(1, wallet.get_asset('a'))
        wallet.change_asset('b', 1)
        self.assertEqual(1, wallet.get_asset('b'))
        wallet.change_asset('a', -1)
        self.assertEqual(0, wallet.get_asset('a'))
        self.assertRaises(InvalidAmountException,
                          wallet.change_asset, 'b', -2)
        self.assertEqual(1, wallet.get_asset('b'))
        self.assertRaises(InvalidAmountException,
                          wallet.change_asset, 'c', -2)
        self.assertEqual(32.1, wallet.get_asset('d'))

        wallet2 = TradingWallet(wallet.get_state())
        self.assertEqual(32.1, wallet2.get_asset('d'))

        wallet2.change_asset('d', 9.9)
        self.assertEqual(32.1, wallet.get_asset('d'))
        self.assertEqual(42, wallet2.get_asset('d'))


class TestTradingBroker(unittest.TestCase):
    class DummyTradingBroker(TradingBroker):
        def __init__(self, f, r, rate):
            super().__init__(f, r)
            self._r = rate

        def get_rate(self, asset, currency) -> float:
            if asset in self._r:
                return self._r[asset][currency]
            else:
                return 5

    def test(self):

        broker = TestTradingBroker.DummyTradingBroker(5, 0.1, {'a': {'c': 5,
                                                                     'b': 5,
                                                                     'd': 5},
                                                               'e': {'c': 10}})
        wallet = TradingWallet({'c': 100, 'd': 4, 'e': 20})

        self.assertRaises(NotImplementedError, broker.make_trade,
                          wallet, 'bhu', 'a', 'b', 5)
        self.assertRaises(InvalidParameterException, broker.make_trade,
                          wallet, 'buy', 'a', 'b', -5)
        self.assertRaises(FundsInsufficientException, broker.make_trade,
                          wallet, 'buy', 'a', 'b', 10)
        self.assertRaises(FundsInsufficientException, broker.make_trade,
                          wallet, 'buy', 'a', 'd', 6)
        self.assertRaises(FeesNotCoveredException, broker.make_trade,
                          wallet, 'buy', 'a', 'b', 5)
        self.assertRaises(EmptyTradeException, broker.make_trade,
                          wallet, 'sell', 'a', 'b', 0)

        trade = broker.make_trade(wallet, 'buy',
                                  asset='a', currency='c', amount=10)

        self.assertEqual('buy', trade.trade_type)
        self.assertEqual(4.5/5, trade.stock_amount)
        self.assertEqual('a', trade.stock_asset)
        self.assertEqual(10, trade.currency_amount)
        self.assertEqual('c', trade.currency_asset)
        self.assertEqual(5, trade.fixed_fee)
        self.assertEqual(0.5, trade.relative_fee)
        self.assertEqual(5.5, trade.total_fee)
        self.assertEqual(5, trade.base_rate)
        self.assertEqual(50/4.5, trade.effective_rate)

        self.assertEqual(4.5/5, wallet.get_asset('a'))
        self.assertEqual(90, wallet.get_asset('c'))

        trade = broker.make_trade(wallet, 'sell',
                                  asset='e', currency='c', amount=10)

        self.assertEqual('sell', trade.trade_type)
        self.assertEqual(10, trade.stock_amount)
        self.assertEqual('e', trade.stock_asset)
        self.assertEqual(85.5, trade.currency_amount)
        self.assertEqual('c', trade.currency_asset)
        self.assertEqual(5, trade.fixed_fee)
        self.assertEqual(9.5, trade.relative_fee)
        self.assertEqual(14.5, trade.total_fee)
        self.assertEqual(10, trade.base_rate)
        self.assertEqual(85.5/10, trade.effective_rate)

        self.assertEqual(10, wallet.get_asset('e'))
        self.assertEqual(175.5, wallet.get_asset('c'))





