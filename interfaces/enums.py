from enum import Enum


class CryptoAssets(Enum):
    BTC = 'BTC'
    ETH = 'ETH'


class CurrencyAsset(Enum):
    EUR = 'EUR'
    USD = 'USD'


class OHCLTimespan(Enum):
    ONE_MINUTE = 'ONE_MINUTE'
    TEN_MINUTES = 'TEN_MINUTES'
    HOURLY = 'HOURLY'
    DAILY = 'DAILY'
    WEEKLY = 'WEEKLY'


class TransactionResult(Enum):
    SUCCESS = 0
    INSUFFICIENT_ASSETS = 1
    UNKNOWN_FAILURE = 2
