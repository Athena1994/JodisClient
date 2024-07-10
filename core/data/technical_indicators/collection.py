from typing import List

from core.data.technical_indicators.indicators import Indicator

from .momentum import *


class IndicatorCollection:

    AwesomeOscillator = AwesomeOscillatorIndicator()
    KAMA = KAMAIndicator()
    PPO = PercentagePriceOscillatorIndicator()
    PVO = PercentageVolumeOscillatorIndicator()
    ROC = ROCIndicator()
    RSI = RSIIndicator()
    TSI = TSIIndicator()
    StochasticRSI = StochRSIIndicator()
    StochasticOscillator = StochasticOscillatorIndicator()
    UltimateOscillator = UltimateOscillatorIndicator()
    WilliamsR = WilliamsRIndicator()

    @staticmethod
    def get(name: str) -> IndicatorDescription:
        prototype = IndicatorCollection.__dict__.get(name)

        if prototype is None:
            raise Exception(f"Indicator {name} not found!")

        return prototype.get_descriptor()

    @staticmethod
    def get_all() -> List[IndicatorDescription]:
        return [IndicatorCollection.__dict__[v].get_descriptor()
                for v, m in vars(IndicatorCollection).items()
                if not (v.startswith('_') or callable(m))]

    @staticmethod
    def get_from_cfg(cfg: dict) -> Indicator:
        if 'name' not in cfg or 'params' not in cfg:
            raise ValueError("Indicator config must contain 'name' and 'params' fields!")
        return IndicatorCollection.get(cfg['name']).create_indicator(cfg['params'])