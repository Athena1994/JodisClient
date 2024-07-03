import typing
from abc import abstractmethod
from collections import namedtuple

import numpy as np

from old.chunks import DataChunk, DataChunkIt
from old.data_source import CandleData
from data.utils import get_open_close_av, correct_rate_for_fee
from trading.interface import Wallet
from ai.utils import Actions
from trading.wallet import ReceiptItems


def calc_best_buy_gain(best_future_rates: dir) -> float:
    return best_future_rates['sell'] - best_future_rates['buy']


def calc_best_sell_gain(best_future_rates: dir, buy_rate: float) -> float:
    return best_future_rates['sell'] - buy_rate


def calc_buy_reward(buy_rate: float,
                    max_sell_rate: float) -> float:
    # reward: best possible profit (- distance to the best future buy offer)
    best_profit = max_sell_rate - buy_rate

    return best_profit  # - distance_to_best_buy


def calc_sell_reward(buy_rate: float,
                     sell_rate: float) -> float:
    # reward: gain #- distance to the best future sell offer
    gain = sell_rate - buy_rate

    return gain  # - distance_to_best_sell


TransactionInfo = namedtuple('TransactionInfo', ['prev_action', 'rate'])

class RewardEvaluator:

    def __init__(self):
        self._invalid_action_penalty = -10000
        self._evaluation_lookahead = 60  # lookahead in min
        self._default_cur = 'EUR'
        self._fix_fee = 0
        self._rel_fee = 0

    def set_fees(self, fix, rel):
        self._fix_fee = fix
        self._rel_fee = rel

    def _get_best_rates(self,
                        chunk: DataChunkIt,
                        state_ix: int):

        # find optimal price rates in [lookahead] timespan
        future = chunk.get(ix_from=state_ix,
                           cnt=self._evaluation_lookahead,
                           restrict=False,
                           relative_zero=False)
        av_buy_prices = get_open_close_av(future[:, 1], future[:, 3])
        av_sell_prices = get_open_close_av(future[:, 1], future[:, 3])
        # TODO: fee correction incorrect -> should be applied to actual value not rates
        return {'buy': correct_rate_for_fee(np.min(av_buy_prices), fix=self._fix_fee, rel=self._rel_fee, is_buy=True),
                'sell': correct_rate_for_fee(np.max(av_sell_prices), fix=self._fix_fee, rel=self._rel_fee, is_buy=False)}

    def _get_current_rates(self,
                           chunk: DataChunkIt,
                           state_ix: int):
        current_candle = chunk.get_current(False)
        av_rate = get_open_close_av(current_candle[CandleData.OPEN],
                                    current_candle[CandleData.CLOSE])
        buy_price = correct_rate_for_fee(av_rate,
                                         fix=self._fix_fee,
                                         rel=self._rel_fee,
                                         is_buy=True)
        sell_price = correct_rate_for_fee(av_rate,
                                          fix=self._fix_fee,
                                          rel=self._rel_fee,
                                          is_buy=False)
        return {'buy': buy_price,
                'sell': sell_price}

    def _get_transaction_info(self, wallet_state: dict) -> TransactionInfo:
        wallet = Wallet.load_from_state(wallet_state)

        if len(wallet.history) == 0:
            return TransactionInfo('none', 0)

        if wallet.history[-1][ReceiptItems.SOURCE_ASSET] == self._default_cur:
            prev_action = 'buy'
        else:
            prev_action = 'sell'

        rate = wallet.history[-1][ReceiptItems.EFFECTIVE_RATE]
        return TransactionInfo(prev_action, rate)

    # chunk: base chunk
    # state_ix: state index rel to chunk
    # wallet_state: wallet after action was performed
    # action: selected action in state_ix
    def get_action_reward(self,
                          chunk: DataChunkIt,
                          state_ix: int,
                          wallet_state: dict,
                          action: int):

        current_rates = self._get_current_rates(chunk, state_ix)
        best_rates = self._get_best_rates(chunk, state_ix)
        info = self._get_transaction_info(wallet_state)

        if (info.prev_action == 'buy' and action == Actions.BUY)\
        or (info.prev_action != 'buy' and action == Actions.SELL):
            return self._invalid_action_penalty

        # NOTE: does not yet account for fees, buy/sell volume, different assets and/or currencies
        if action == Actions.BUY:
            return calc_buy_reward(buy_rate=current_rates['buy'],
                                   max_sell_rate=best_rates['sell'])
        elif action == Actions.SELL:
            return calc_sell_reward(buy_rate=info.rate,
                                    sell_rate=current_rates['sell'])
        elif action == Actions.WAIT:
            if info.prev_action == 'buy':
                return calc_sell_reward(buy_rate=best_rates['buy'],
                                        sell_rate=best_rates['sell'])
            else:
                return calc_buy_reward(buy_rate=best_rates['buy'],
                                       max_sell_rate=best_rates['sell'])
        else:
            raise NotImplementedError('This should never be reached!')


