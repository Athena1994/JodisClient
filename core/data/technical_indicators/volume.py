import ta

from .indicators import IndicatorParameterDescription, IndicatorPrototype, IndicatorDescription


class AccDistIndexIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("AccDistIndexIndicator")

    def calculate(self, params, df):
        return ta.volume.acc_dist_index(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'])


class ChaikinMoneyFlowIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("ChaikinMoneyFlowIndicator", [
                IndicatorParameterDescription('window', 5, 30, 20, 'int')
            ], 
            "window")

    def calculate(self, params, df):
        return ta.volume.chaikin_money_flow(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'])


class EaseOfMovementIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("EaseOfMovementIndicator", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int')
            ], "window")

    def calculate(self, params, df):
        return ta.volume.ease_of_movement(high=df['high'],
                                    low=df['low'],
                                    volume=df['volume'],
                                    window=params['window'])


class ForceIndexIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("ForceIndexIndicator", [
                IndicatorParameterDescription('window', 5, 20, 13, 'int')
            ], "window")

    def calculate(self, params, df):
        return ta.volume.force_index(volume=df['volume'],
                                    close=df['close'],
                                    window =params['window'])


class MFIIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("MFIIndicator", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int')
            ], "window")

    def calculate(self, params, df):
        return ta.volume.money_flow_index(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'],)


class NegativeVolumeIndexIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("NegativeVolumeIndexIndicator", [], None)

    def calculate(self, params, df):
        return ta.volume.negative_volume_index(close=df['close'],
                                    volume=df['volume'],)


class OnBalanceVolumeIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("OnBalanceVolumeIndicator", [], None)

    def calculate(self, params, df):
        return ta.volume.on_balance_volume(
                                    close=df['close'],
                                    volume=df['volume'])


class SMAEaseOfMovementIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("SMAEaseOfMovementIndicator", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int')
            ], "window")

    def calculate(self, params, df):
        return ta.volume.on_balance_volume(
                                    close=df['close'],
                                    volume=df['volume'],
                                    high=df['high'],
                                    low=df['low'],
                                    window=df['window'])


class VolumePriceTrendIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("VolumePriceTrendIndicator", [], "window")

    def calculate(self, params, df):
        return ta.volume.volume_price_trend(
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'],)


class VolumeWeightedAveragePriceIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("VolumeWeightedAveragePrice", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int')
            ], "window")

    def calculate(self, params, df):
        return ta.volume.volume_weighted_average_price(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'])
