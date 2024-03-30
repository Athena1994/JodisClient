import pandas as pd
import ta.momentum


from technical_indicators.indicators import IndicatorPrototype, IndicatorDescription


class AwesomeOscillatorIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("AwsomeOscillator",
                                    [('short_period', 1, 10, 5, 'int'),
                              ('long_period', 10, 50, 34, 'int')],
                                    self.calculate,
                                    norm_factor=100,
                                    skip_field='long_period')

    def calculate(self, params, df):
        return ta.momentum.awesome_oscillator(high=df['high'],
                                              low=df['low'],
                                              window1=params['short_period'],
                                              window2=params['long_period'])


class KAMAIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("KAMA",
                                    [('window', 5, 20, 10, 'int'),
                              ('pow1', 1, 10, 2, 'int'),
                              ('pow2', 11, 40, 30, 'int')],
                                    self.calculate,
                                    norm_factor=-1,
                                    skip_field='window')

    def calculate(self, params, df):
        return ta.momentum.kama(df['close'],
                                window=params['window'],
                                pow1=params['pow1'],
                                pow2=params['pow2'])


class PercentagePriceOscillatorIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("PercentagePriceOscillator",
                                    [('window_slow', 15, 40, 26, 'int'),
                              ('window_fast', 5, 15, 12, 'int'),
                              ('window_sign', 2, 20, 9, 'int')],
                                    self.calculate,
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
#        return pd.merge(pd.merge(ppo, ppo_hist, on='id'), ppo_sig, on='id')


class PercentageVolumeOscillatorIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("PercentagePriceOscillator",
                                    [('window_slow', 15, 40, 26, 'int'),
                              ('window_fast', 5, 15, 12, 'int'),
                              ('window_sign', 2, 20, 9, 'int')],
                                    self.calculate,
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
#        return pd.merge(pd.merge(ppo, ppo_hist, left_index=True, right_index=True),
#                        ppo_sig, left_index=True, right_index=True)


class ROCIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("ROC",
                                    [('window', 5, 20, 12, 'int')],
                                    self.calculate,
                                    norm_factor=1,
                                    skip_field='window')

    def calculate(self, params, df):
        return ta.momentum.roc(close=df['close'],
                               window=params['window'])


class RSIIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("RSI",
                                    [('window', 5, 20, 14, 'int')],
                                    self.calculate,
                                    norm_factor=100,
                                    skip_field='window')

    def calculate(self, params, df):
        return ta.momentum.rsi(df['close'],
                               window=params['window'])


class StochasticOscillatorIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("StochasticOscillator",
                                    [('window', 5, 20, 14, 'int'),
                              ('smooth_window', 1, 10, 3, 'int')],
                                    self.calculate,
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
#        return pd.merge(stoch, stoch_sig, left_index=True, right_index=True)


class StochRSIIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("StochRSI",
                                    [('window', 5, 20, 14, 'int'),
                              ('smooth1', 1, 10, 3, 'int'),
                              ('smooth2', 1, 10, 3, 'int')],
                                    self.calculate,
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

    def get_descriptor(self):
        return IndicatorDescription("TSI",
                                    [('window_slow', 20, 30, 25, 'int'),
                              ('window_fast', 10, 20, 13, 'int')],
                                    self.calculate,
                                    norm_factor=100,
                                    skip_field='window_slow')

    def calculate(self, params, df):
        return ta.momentum.tsi(close=df['close'],
                                    window_slow=params['window_slow'],
                                    window_fast=params['window_fast'])


class UltimateOscillatorIndicator(IndicatorPrototype):

    def get_descriptor(self):
        return IndicatorDescription("UltimateOscillator",
                                    [('window1', 5, 10, 7, 'int'),
                              ('window2', 11, 20, 14, 'int'),
                              ('window3', 21, 35, 28, 'int'),
                              ('weight1', 2, 5, 4.0, 'float'),
                              ('weight2', 1, 3, 2.0, 'float'),
                              ('weight3', 0.5, 2, 1.0, 'float')],
                                    self.calculate,
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

    def get_descriptor(self):
        return IndicatorDescription("WilliamsRIndicator",
                                    [('lbp', 5, 20, 14, 'int')],
                                    self.calculate,
                                    norm_factor=100,
                                    skip_field='lbp')

    def calculate(self, params, df):
        return ta.momentum.williams_r(high=df['high'],
                                      low=df['low'],
                                      close=df['close'],
                                      lbp=params['lbp'])

