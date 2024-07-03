import typing

import numpy as np

from old.chunks import DataChunk, ChunkDataSource, DataChunkIt
from old.data_source import DataSource

from trading.interface import BaseTradingInterface, TradingInterface, TradingEnvironment
from trading.wallet import Wallet

import ai
from ai.reward_calculation import RewardEvaluator
from ai.replay_buffer import ReplayBuffer
from ai.utils import Actions


class EnvironmentSimulation(TradingEnvironment):

    def __init__(self):

        super().__init__()

        self._trading_interface = BaseTradingInterface(self)
        self._trading_interface.set_update_callback(self._tick)
        self._reward_eval = RewardEvaluator()

        self._wallet: Wallet = None

        self._chunk: DataChunk = None
        self._chunk_iter: DataChunkIt = None
        self._datasource: DataSource = None

        self._money_norm: typing.Tuple[float, float] = None

        self._fee_fix: float = 0
        self._fee_rel: float = 0.0026
        self._reward_eval.set_fees(self._fee_fix, self._fee_rel)
        self._ix = 0

    # ### TradingEnvironment interface ########################

    def get_datasource(self) -> DataSource:
        return self._datasource

    def get_wallet(self) -> Wallet:
        return self._wallet

    def get_trading_state(self, offset: int = 0) \
            -> typing.Tuple[np.ndarray, np.ndarray]:
        return self._get_feature_vector(offset)[0], self._get_state_vec()

    def buy(self, asset, cur, rate=None):
        if rate is None:
            rate = self.get_datasource().get_current_buy_price(asset, cur)

        money = self.get_wallet().get_asset(cur)

        fee = (money - self._fee_fix) * self._fee_rel
        net = money - fee

        return self.get_wallet().exec_transaction(
            source_asset=cur,
            source_amount=money,
            target_asset=asset,
            target_amount=net / rate,
            exchange_rate=rate,
            fee=fee,
            timestamp=self._chunk_iter.get_timestamp()
        )

    def sell(self, asset, cur, rate=None):
        if rate is None:
            rate = self.get_datasource().get_current_sell_price(asset, cur)

        amount = self.get_wallet().get_asset(asset)

        gross = amount * rate
        net = (gross - self._fee_fix) * (1 - self._fee_rel)
        fee = gross - net

        return self.get_wallet().exec_transaction(
            source_asset=asset,
            source_amount=amount,
            target_asset=cur,
            target_amount=net,
            exchange_rate=rate,
            fee=fee,
            timestamp=self._chunk_iter.get_timestamp()
        )

    # ### control interface ########################

    def set_chunk(self, chunk: DataChunk):
        self._chunk = chunk
        self._chunk_iter = iter(chunk)
        self._datasource = ChunkDataSource(self._chunk_iter)
        self._datasource.set_fees(self._fee_fix, self._fee_rel)

        rich_df, self._money_norm \
            = self._agent.add_indicators_to_df(self._chunk.get_df())
        self._chunk_iter.set_rich_data(np.array(rich_df))

    def get_chunk(self) -> DataChunkIt:
        return self._chunk_iter

    def get_wallet_worth(self):
        return self._wallet.get_asset('EUR') \
            + self._wallet.get_asset('BTC') * self._datasource\
                                                  .get_current_sell_price("BTC",
                                                                          "EUR")

    def reset(self, wallet_dict: dict):
        self._wallet = Wallet(wallet_dict)
        self._chunk_iter.reset()

    def __iter__(self):
        self._ix = 0
        return self

    def notify_update(self):
        self._trading_interface.notify_update()

    def _get_feature_vector(self, ix: int = 0) -> np.ndarray:
        if ix != -1:
            ix -= self._agent.get_lookback()
        return self._chunk_iter.get(ix,
                                    cnt=self._agent.get_lookback(),
                                    relative_zero=True,
                                    use_rich_data=True,
                                    restrict=False),

    def _get_state_vec(self) -> np.ndarray:
        date, value = self._wallet.get_open_position('BTC')
        if date is None:
            has_open = False
        else:
            has_open = True
            value = (value - self._money_norm[0]) / self._money_norm[1]
        return np.array([float(has_open), value])

    def _tick(self, ti: TradingInterface):
        state = self.get_trading_state()
        wallet_state = self._wallet.get_state()
        action = self._agent.select_action(ti, state)

        reward = self._reward_eval.get_action_reward(
            chunk=self._chunk_iter,
            state_ix=self._chunk_iter.get_ix(),
            wallet_state=wallet_state,
            action=action
        )

        if action == Actions.BUY:
            ti.buy('BTC', 'EUR')
        elif action == Actions.SELL:
            ti.sell('BTC', 'EUR')

        new_state = self.get_trading_state(1)
        new_wallet_state = self._wallet.get_state()

        self._replay_buffer.add_experience(
            state_dict={'feature_vec': state,
                        'wallet': wallet_state},
            new_state_dict={'feature_vec': new_state,
                            'wallet': new_wallet_state},
            action=action,
            reward=reward
        )

