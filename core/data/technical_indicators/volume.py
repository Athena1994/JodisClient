import ta

from .indicators import IndicatorPrototype, IndicatorDescription


class AccDistIndexIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("AccDistIndexIndicator",
                                    [],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.acc_dist_index(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'])


class ChaikinMoneyFlowIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("ChaikinMoneyFlowIndicator",
                                    [('window', 5, 30, 20, 'int')],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.chaikin_money_flow(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'])


class EaseOfMovementIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("EaseOfMovementIndicator",
                                    [('window', 5, 20, 14, 'int')],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.ease_of_movement(high=df['high'],
                                    low=df['low'],
                                    volume=df['volume'],
                                    window=params['window'])


class ForceIndexIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("ForceIndexIndicator",
                                    [('window', 5, 20, 13, 'int')],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.force_index(volume=df['volume'],
                                    close=df['close'],
                                    window =params['window'])


class MFIIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("MFIIndicator",
                                    [('window', 5, 20, 14, 'int')],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.money_flow_index(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'],)


class NegativeVolumeIndexIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("NegativeVolumeIndexIndicator",
                                    [],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.negative_volume_index(close=df['close'],
                                    volume=df['volume'],)


class OnBalanceVolumeIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("OnBalanceVolumeIndicator",
                                    [],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.on_balance_volume(
                                    close=df['close'],
                                    volume=df['volume'])


class SMAEaseOfMovementIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("SMAEaseOfMovementIndicator",
                                    [('window', 5, 20, 14, 'int')],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.on_balance_volume(
                                    close=df['close'],
                                    volume=df['volume'],
                                    high=df['high'],
                                    low=df['low'],
                                    window=df['window'])


class VolumePriceTrendIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("VolumePriceTrendIndicator",
                                    [],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.volume_price_trend(
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['window'],)


class VolumeWeightedAveragePriceIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("VolumeWeightedAveragePrice",
                                    [('window', 5, 20, 14, 'int')],
                                    self.calculate)

    def calculate(self, params, df):
        return ta.volume.volume_weighted_average_price(high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    volume=df['volume'],
                                    window=params['volume'])
