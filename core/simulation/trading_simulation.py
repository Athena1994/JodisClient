


from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import torch
from core.data.data_provider import DataProvider
from core.qlearning.replay_buffer import Experience

class TradingSimulation:

    class Modes:
        RESET_AFTER_SELL = "reset_after_sell"
        RESET_EACH_EPISODE = "reset_each_episode"
        NO_RESET = "no_reset"

    def __init__(self, 
                 data_provider: DataProvider, 
                 sim_config: dict) -> None:
        self._data_provider = data_provider
        self._config = sim_config

        self._chunk_iterator = None  # iterates data chunks 
        self._sample_iterator = None  # iterates samples within a chunk

        self._episode = 0
        self._episode_state = None
        self._mode = None

    def _reset_state(self):
        self._episode_state = {
            'money': self._config['general']['initial_balance'],
            'asset': 0
        }

    def _next_episode(self) -> bool:
        
        if self._mode == self.Modes.RESET_EACH_EPISODE \
        or self._mode == self.Modes.RESET_AFTER_SELL:
            self._reset_state()

        self._episode += 1
        self._sample_iterator = next(self._chunk_iterator, None)

        return self._sample_iterator is not None

    def start_session(self, type: str) -> None:

        self._episode = -1

        if type == 'tr':
            self._mode = self.Modes.RESET_AFTER_SELL
        elif type == 'val':
            self._mode = self.Modes.RESET_EACH_EPISODE
        else:
            self._mode = self.Modes.NO_RESET

        self._chunk_iterator = self._data_provider.get_iterator(type)
 
        self._reset_state()
        self._next_episode()

    @staticmethod
    def get_price(current_candle: pd.Series) -> float:
        return current_candle['close']

    def buy(self, current_candle: pd.Series) -> bool:
        money = self._episode_state['money']
        if money == 0:
            return None

        price = self.get_price(current_candle)
        amount = money / price

        self._episode_state['money'] = 0
        self._episode_state['asset'] = amount

        return amount
        
    def sell(self, current_candle: pd.Series) -> None:
        asset = self._episode_state['asset']
        if asset == 0:
            return None
        
        price = self.get_price(current_candle)
        money = asset * price
        
        if self._mode == self.Modes.RESET_AFTER_SELL:
            self._reset_state()
        else:
            self._episode_state['money'] = money
            self._episode_state['asset'] = 0

        return money



    def get_next_state(self) -> Tuple[torch.Tensor, pd.Series, dict, int]:
     
        if self._sample_iterator is None:
            raise RuntimeError("Session not started.")

        try:
            return (*next(self._sample_iterator), 
                    self._episode_state, 
                    self._episode)
        except StopIteration:
            if not self._next_episode():
                return None
            return self.get_next_state()

        

