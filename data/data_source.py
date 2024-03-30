from abc import abstractmethod
from enum import Enum


class CandleData:
    DATE: int = 0,
    OPEN: int = 1,
    CLOSE: int = 2,
    HIGH: int = 3,
    LOW: int = 4


class DataSource:

    def __init__(self):
        pass

    @abstractmethod
    def get_candles(self, asset, cur, timespan, since=None):
        pass

    @abstractmethod
    def get_current_candle(self, asset, cur, timespan):
        pass

    @abstractmethod
    def get_current_buy_price(self, asset, cur) -> float:
        pass

    @abstractmethod
    def get_current_sell_price(self, asset, cur) -> float:
        pass

