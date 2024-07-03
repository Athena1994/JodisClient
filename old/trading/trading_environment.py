from abc import abstractmethod

import numpy as np

from old.data_source import DataSource
from trading.wallet import Wallet


class TradingEnvironment:
    @abstractmethod
    def get_wallet(self) -> Wallet:
        yield

    @abstractmethod
    def get_datasource(self) -> DataSource:
        yield

    @abstractmethod
    def get_trading_state(self) -> np.ndarray:
        yield

    @abstractmethod
    def buy(self, asset, cur):
        yield

    @abstractmethod
    def sell(self, asset, cur):
        yield

