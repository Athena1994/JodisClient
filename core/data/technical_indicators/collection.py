from typing import List

from technical_indicators.momentum import *


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

    def get(self, name: str) -> IndicatorDescription:
        return IndicatorCollection.__dict__[name].get_descriptor()

    @staticmethod
    def get_all() -> List[IndicatorDescription]:
        return [IndicatorCollection.__dict__[v].get_descriptor()
                for v, m in vars(IndicatorCollection).items()
                if not (v.startswith('_') or callable(m))]

