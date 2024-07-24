
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

from core.simulation.state_manager import StateManager
from utils.config_utils import assert_fields_in_dict


class ExchangeDirection(Enum):
    BUY = 0
    SELL = 1


class Exchanger:
    @dataclass
    class FeeDetails:
        relative: float
        fixed: float

    @dataclass
    class PairDetails:
        buy_rate: float
        sell_rate: float
        fee: 'Exchanger.FeeDetails'

    @dataclass
    class Receipt:
        currency: str
        asset: str
        currency_balance: float
        asset_balance: float
        rate: float
        fee: float

    @abstractmethod
    def get_exchange_details(self,
                             currency: str,
                             asset: str) -> PairDetails:
        pass

    @abstractmethod
    def prepare_exchange(self,
                         currency: str,
                         asset: str,
                         amount: float,
                         action: ExchangeDirection) -> Receipt:
        pass


class StateSourcedExchanger(Exchanger):

    @dataclass
    class Config:
        @dataclass
        class Pair:
            asset: str
            currency: str
            fee: Exchanger.FeeDetails
            candle_src: str

            @staticmethod
            def from_dict(conf: dict) -> 'StateSourcedExchanger.Config.Pair':
                assert_fields_in_dict(conf, ['asset', 'currency', 'fee',
                                             'candle_src'])
                assert_fields_in_dict(conf['fee'], ['relative', 'fixed'])
                return StateSourcedExchanger.Config.Pair(
                    conf['asset'],
                    conf['currency'],
                    Exchanger.FeeDetails(
                        conf['fee']['relative'],
                        conf['fee']['fixed']),
                    conf['candle_src'])

        pairs: list[Pair]

        @staticmethod
        def from_dict(conf: dict) -> 'StateSourcedExchanger.Config':
            assert_fields_in_dict(conf, ['pairs'])
            return StateSourcedExchanger.Config(
                [StateSourcedExchanger.Config.Pair.from_dict(d)
                 for d in conf['pairs']])

    def __init__(self,
                 state_manager: StateManager,
                 cfg: Config) -> None:
        super().__init__()

        self._state_manager = state_manager

        self._exchange_config = {}

        for pair in cfg.pairs:
            pair_name = f'{pair.asset}{pair.currency}'
            self._exchange_config[pair_name] = {
                'fee': pair.fee,
                'candle_src': pair.candle_src
            }

    @staticmethod
    def _calculate_rate(candle: dict) -> float:
        return (candle['close'] + candle['open']) / 2

    def get_exchange_details(self,
                             currency: str,
                             asset: str) -> Exchanger.PairDetails:
        pair = f'{asset}{currency}'
        if pair not in self._exchange_config:
            raise Exception(f"Exchanger has no pair {pair}")

        conf = self._exchange_config[pair]

        samples = self._state_manager.get_samples()

        if conf['candle_src'] not in samples:
            raise Exception(f"Missing candle source '{conf['candle_src']}'")

        candle = samples[conf['candle_src']].context
        rate = self._calculate_rate(candle)

        return Exchanger.PairDetails(rate, rate, conf['fee'])

    def prepare_exchange(self,
                         currency: str,
                         asset: str,
                         amount: float,
                         action: ExchangeDirection) -> Exchanger.Receipt:

        details = self.get_exchange_details(currency, asset)

        if action == ExchangeDirection.BUY:
            rate = details.buy_rate
            total_fee = details.fee.fixed \
                + details.fee.relative * (amount - details.fee.fixed)
            effective_amount = amount - total_fee
            asset_amount = effective_amount / rate
            currency_balance = -amount
            asset_balance = asset_amount
        else:
            rate = details.sell_rate
            turnover = amount * rate
            total_fee = details.fee.fixed \
                + details.fee.relative * (turnover - details.fee.fixed)
            currency_balance = turnover - total_fee
            asset_balance = -amount

        return Exchanger.Receipt(currency, asset,
                                 currency_balance, asset_balance,
                                 rate, total_fee)
