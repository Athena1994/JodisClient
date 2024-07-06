import pandas as pd
import ta.momentum


from .indicators import IndicatorParameterDescription, IndicatorPrototype, IndicatorDescription


class AwesomeOscillatorIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("AwsomeOscillator", [
                IndicatorParameterDescription('short_period', 1, 10, 5, 'int'),
                IndicatorParameterDescription('long_period', 10, 50, 34, 'int')
            ],
            norm_factor=100,
            skip_field='long_period')

    def calculate(self, params, df):
        return ta.momentum.awesome_oscillator(high=df['high'],
                                              low=df['low'],
                                              window1=params['short_period'],
                                              window2=params['long_period'])


class KAMAIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("KAMA", [
                IndicatorParameterDescription('window', 5, 20, 10, 'int'),
                IndicatorParameterDescription('pow1', 1, 10, 2, 'int'),
                IndicatorParameterDescription('pow2', 11, 40, 30, 'int')
            ],
            norm_factor=1,
            skip_field='window')

    def calculate(self, params, df):
        return ta.momentum.kama(df['close'],
                                window=params['window'],
                                pow1=params['pow1'],
                                pow2=params['pow2'])


class PercentagePriceOscillatorIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("PercentagePriceOscillator", [
                IndicatorParameterDescription('window_slow', 15, 40, 26, 'int'),
                IndicatorParameterDescription('window_fast', 5, 15, 12, 'int'),
                IndicatorParameterDescription('window_sign', 2, 20, 9, 'int')
            ],
            norm_factor=(1, 1, 1),
            skip_field='window_slow')
        
    def calculate(self, params, df):
        ppo = ta.momentum.ppo(close=df['close'],
                              window_slow=params['window_slow'],
                              window_fast=params['window_fast'],
                              window_sign=params['window_sign'], )
        ppo_hist = ta.momentum.ppo_hist(
                                close=df['close'],
                                window_slow=params['window_slow'],
                                window_fast=params['window_fast'],
                                window_sign=params['window_sign'], )
        ppo_sig = ta.momentum.ppo_signal(
                                close=df['close'],
                                window_slow=params['window_slow'],
                                window_fast=params['window_fast'],
                                window_sign=params['window_sign'],)
        
        return ppo, ppo_hist, ppo_sig


class PercentageVolumeOscillatorIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("PercentagePriceOscillator", [
                IndicatorParameterDescription('window_slow', 15, 40, 26, 'int'),
                IndicatorParameterDescription('window_fast', 5, 15, 12, 'int'),
                IndicatorParameterDescription('window_sign', 2, 20, 9, 'int')
            ],
            norm_factor=(100, 100, 100),
            skip_field='window_slow')

    def calculate(self, params, df):
        ppo = ta.momentum.pvo(volume=df['volume'],
                              window_slow=params['window_slow'],
                              window_fast=params['window_fast'],
                              window_sign=params['window_sign'], )
        ppo_hist = ta.momentum.pvo_hist(
                                volume=df['volume'],
                                window_slow=params['window_slow'],
                                window_fast=params['window_fast'],
                                window_sign=params['window_sign'], )
        ppo_sig = ta.momentum.pvo_signal(
                                volume=df['volume'],
                                window_slow=params['window_slow'],
                                window_fast=params['window_fast'],
                                window_sign=params['window_sign'],)
        return ppo, ppo_hist, ppo_sig


class ROCIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("ROC", [
                IndicatorParameterDescription('window', 5, 20, 12, 'int')
            ],
            norm_factor=1,
            skip_field='window')

    def calculate(self, params, df):
        return ta.momentum.roc(close=df['close'],
                               window=params['window'])


class RSIIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("RSI", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int')
            ], 
            norm_factor=100,
            skip_field='window')

    def calculate(self, params, df):
        return ta.momentum.rsi(df['close'],
                               window=params['window'])


class StochasticOscillatorIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("StochasticOscillator", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int'),
                IndicatorParameterDescription('smooth_window', 1, 10, 3, 'int')
            ],
        norm_factor=(100, 100),
        skip_field='window')

    def calculate(self, params, df):
        stoch = ta.momentum.stoch(high=df['high'],
                                  low=df['low'],
                                  close=df['close'],
                                  window=params['window'],
                                  smooth_window=params['smooth_window'])
        stoch_sig = ta.momentum.stoch_signal(
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 window=params['window'],
                                 smooth_window=params['smooth_window'])

        return stoch, stoch_sig


class StochRSIIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("StochRSI", [
                IndicatorParameterDescription('window', 5, 20, 14, 'int'),
                IndicatorParameterDescription('smooth1', 1, 10, 3, 'int'),
                IndicatorParameterDescription('smooth2', 1, 10, 3, 'int')
            ],
            norm_factor=(1, 1, 1),
            skip_field='window')

    def calculate(self, params, df):
        srsi = ta.momentum.stochrsi(
                                  df['close'],
                                  window=params['window'],
                                  smooth1=params['smooth1'],
                                  smooth2=params['smooth2'])
        srsid = ta.momentum.stochrsi_d(
                                  df['close'],
                                  window=params['window'],
                                  smooth1=params['smooth1'],
                                  smooth2=params['smooth2'])
        srsik = ta.momentum.stochrsi_k(
                                  df['close'],
                                  window=params['window'],
                                  smooth1=params['smooth1'],
                                  smooth2=params['smooth2'])
        return srsi, srsid, srsik


class TSIIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("TSI", [
                IndicatorParameterDescription('window_slow', 20, 30, 25, 'int'),
                IndicatorParameterDescription('window_fast', 10, 20, 13, 'int')
            ],
            norm_factor=100,
            skip_field='window_slow')

    def calculate(self, params, df):
        return ta.momentum.tsi(close=df['close'],
                                    window_slow=params['window_slow'],
                                    window_fast=params['window_fast'])


class UltimateOscillatorIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("UltimateOscillator", [
                IndicatorParameterDescription('window1', 5, 10, 7, 'int'),
                IndicatorParameterDescription('window2', 11, 20, 14, 'int'),
                IndicatorParameterDescription('window3', 21, 35, 28, 'int'),
                IndicatorParameterDescription('weight1', 2, 5, 4.0, 'float'),
                IndicatorParameterDescription('weight2', 1, 3, 2.0, 'float'),
                IndicatorParameterDescription('weight3', 0.5, 2, 1.0, 'float')
            ],
            norm_factor=100,
            skip_field='window3')

    def calculate(self, params, df):
        return ta.momentum.ultimate_oscillator(high=df['high'],
                                               low=df['low'],
                                               close=df['close'],
                                               window1=params['window1'],
                                               window2=params['window2'],
                                               window3=params['window3'],
                                               weight1=params['weight1'],
                                               weight2=params['weight2'],
                                               weight3=params['weight3'])


class WilliamsRIndicator(IndicatorPrototype):

    def __init__(self):
        super().__init__("WilliamsRIndicator", [
                IndicatorParameterDescription('lbp', 5, 20, 14, 'int')
            ],
            norm_factor=100,
            skip_field='lbp')

    def calculate(self, params, df):
        return ta.momentum.williams_r(high=df['high'],
                                      low=df['low'],
                                      close=df['close'],
                                      lbp=params['lbp'])

