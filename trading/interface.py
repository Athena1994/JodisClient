import datetime
import typing
from abc import abstractmethod

import numpy as np

from data.data_source import DataSource
from trading.trading_environment import TradingEnvironment
from trading.wallet import Wallet


class TradingInterface:

    @abstractmethod
    def buy(self, asset: str, cur: str):
        raise NotImplementedError()

    @abstractmethod
    def sell(self, asset: str, cur: str):
        raise NotImplementedError()

    @abstractmethod
    def get_buy_price(self, asset: str, cur: str) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_sell_price(self, asset: str, cur: str) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_date(self) -> datetime.datetime:
        raise NotImplementedError()

    @abstractmethod
    def get_candles(self,
                    asset: str, cur: str,
                    timespan: str, since: datetime.datetime) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_state(self):
        yield


class BaseTradingInterface(TradingInterface):
    def __init__(self, environment: TradingEnvironment):
        super().__init__()
        self._update_callback: typing.Callable[[TradingInterface], None] = None
        self._environment: TradingEnvironment = environment

#    def set_environment(self, env: TradingEnvironment):
#        self._environment = env

    def get_environment(self) -> TradingEnvironment:
        return self._environment

    def get_wallet(self) -> Wallet:
        return self._environment.get_wallet()

    def get_datasource(self) -> DataSource:
        return self._environment.get_datasource()

    def set_update_callback(self, callback: typing.Callable[[TradingInterface], None]):
        self._update_callback = callback

    def notify_update(self):
        if self._update_callback is not None:
            self._update_callback(self)

    def buy(self, asset: str, cur: str):
        return self.get_environment().buy(asset, cur)

    def sell(self, asset: str, cur: str):
        return self.get_environment().sell(asset, cur)

    def get_buy_price(self, asset: str, cur: str) -> float:
        return self.get_datasource().get_current_buy_price(asset, cur)

    def get_sell_price(self, asset: str, cur: str) -> float:
        return self.get_datasource().get_current_sell_price(asset, cur)

    def get_date(self) -> datetime.datetime:
        return self.get_datasource().get_timestamp()

    def get_candles(self, asset: str, cur: str,
                    timespan: str,
                    since: datetime.datetime) -> np.ndarray:
        return self.get_datasource().get_candles(asset, cur, timespan=timespan, since=since)

    def get_state(self) -> np.ndarray:
        return self.get_environment().get_trading_state()

