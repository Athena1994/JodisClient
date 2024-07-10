
import unittest

import numpy as np
import pandas as pd

from core.data.technical_indicators.indicators import IndicatorDescription, IndicatorParameterDescription, IndicatorPrototype
from core.data.technical_indicators.momentum import AwesomeOscillatorIndicator, PercentagePriceOscillatorIndicator
from core.data.utils import apply_indicator, find_indicator_update_regions, assign_chunk_ids, split_time_chunks


class TestUtils(unittest.TestCase):
    
    def test_apply_multi_indicator(self):
        df = pd.DataFrame({'time': pd.date_range(start='1/1/2024', periods=30, freq='min')})
        df['close'] = np.square(np.arange(0, 30))

        params = {
            'window_slow': 5, 
            'window_fast': 3,
            'window_sign': 2
        }
        ind_desc = {
            'name': 'PPO', 
            'params': params
        }
        ind = PercentagePriceOscillatorIndicator().get_descriptor().create_indicator(params)
        col = ind.get_unique_id()

        apply_indicator(df, ind_desc)

        na_map = ~pd.isna(df[col])

        for i in range(0, 4):
            self.assertFalse(na_map[i])
        for i in range(4, 30):
            self.assertTrue(na_map[i])


    def test_apply_indicator(self):
        df = pd.concat([
            pd.DataFrame({'time': pd.date_range(start='1/1/2024', periods=1, freq='min')}),
            pd.DataFrame({'time': pd.date_range(start='1/10/2024', periods=10, freq='min')}),
            pd.DataFrame({'time': pd.date_range(start='3/5/2024', periods=4, freq='min')})])
        df = df.reset_index(drop=True)
        df['high'] = np.square(np.arange(15, 30))
        df['low'] = np.square(np.arange(15))
        
        params = {
            'short_period': 1, 
            'long_period': 3
        }
        ind_desc = {
            'name': 'AwesomeOscillator', 
            'params': params
        }
        ind = AwesomeOscillatorIndicator().get_descriptor().create_indicator(params)
        col = ind.get_unique_id()

        apply_indicator(df, ind_desc)

        na_map = ~pd.isna(df[col])

        self.assertFalse(na_map[0])
        self.assertFalse(na_map[1])
        self.assertFalse(na_map[2])
        self.assertTrue(na_map[3])
        self.assertTrue(na_map[4])
        self.assertTrue(na_map[5])
        self.assertTrue(na_map[6])
        self.assertTrue(na_map[7])
        self.assertTrue(na_map[8])
        self.assertTrue(na_map[9])
        self.assertTrue(na_map[10])
        self.assertFalse(na_map[11])
        self.assertFalse(na_map[12])
        self.assertTrue(na_map[13])
        self.assertTrue(na_map[14])

        df.loc[0: 15, col] = 1
        apply_indicator(df, ind_desc)
        for i in range(15):
            self.assertEqual(df.loc[i, col], 1)

        df.loc[15] = (pd.to_datetime('2024-03-05 00:04:00'), 900, 200, None)
        apply_indicator(df, ind_desc)

        for i in range(15):
            self.assertEqual(df.loc[i, col], 1)
        self.assertTrue(pd.notna(df.loc[15, col]))



    def test_find_indicator_update_regions(self):
        df = pd.concat([
            pd.DataFrame({'time': pd.date_range(start='1/1/2024', periods=1, freq='min')}),
            pd.DataFrame({'time': pd.date_range(start='1/10/2024', periods=10, freq='min')}),
            pd.DataFrame({'time': pd.date_range(start='3/5/2024', periods=4, freq='min')})])
        df = df.reset_index(drop=True)
        indicator_hash = 'indicator'

        # test with indicator that has not yet been added to the dataframe
        update_list = find_indicator_update_regions(df, indicator_hash, 3)
        self.assertEqual(update_list[0], (1, 10))
        self.assertEqual(update_list[1], (11, 4))

        # test with indicator that has already been added to the dataframe with all values null
        df[indicator_hash] = None
        update_list = find_indicator_update_regions(df, indicator_hash, 3)
        self.assertEqual(len(update_list), 2)
        self.assertEqual(update_list[0], (1, 10))
        self.assertEqual(update_list[1], (11, 4))

        # test with some values not null
        df.at[2, indicator_hash] = 1
        df.at[3, indicator_hash] = 1
        df.at[4, indicator_hash] = 1
        df.at[5, indicator_hash] = 1
        df.at[6, indicator_hash] = 1
        df.at[12, indicator_hash] = 1
        df.at[13, indicator_hash] = 1
        update_list = find_indicator_update_regions(df, indicator_hash, 2)
        self.assertEqual(len(update_list), 2)
        self.assertEqual(update_list[0], (6, 5))
        self.assertEqual(update_list[1], (13, 2))

        # test with all values not null in one time frame
        df.at[14, indicator_hash] = 1
        update_list = find_indicator_update_regions(df, indicator_hash, 2)
        self.assertEqual(len(update_list), 1)
        self.assertEqual(update_list[0], (6, 5))


    def test_split_time_chunks(self):

        # test dataframe with datetime entries
        df = pd.concat([pd.DataFrame({'time': pd.date_range(start='1/1/2024', periods=1, freq='min')}),
        pd.DataFrame({'time': pd.date_range(start='1/10/2024', periods=10, freq='min')}),
        pd.DataFrame({'time': pd.date_range(start='3/5/2024', periods=4, freq='min')})])
        df = df.reset_index(drop=True)

        split_list = split_time_chunks(df)

        self.assertEqual(len(split_list), 3)
        self.assertEqual(split_list[0], (0, 1))
        self.assertEqual(split_list[1], (1, 10))
        self.assertEqual(split_list[2], (11, 4))

    def test_assign_chunk_ids(self):
        df = pd.concat([pd.DataFrame({'time': pd.date_range(start='1/1/2024', periods=1, freq='min')}),
        pd.DataFrame({'time': pd.date_range(start='1/10/2024', periods=10, freq='min')}),
        pd.DataFrame({'time': pd.date_range(start='3/5/2024', periods=4, freq='min')})])
        df = df.reset_index(drop=True)

        split_list = split_time_chunks(df)
        df = assign_chunk_ids(df, split_list, 3, 0)

        self.assertEqual(df['chunk'][0],-1)
        self.assertEqual(df['chunk'][1], 0)
        self.assertEqual(df['chunk'][2], 0)
        self.assertEqual(df['chunk'][3], 0)
        self.assertEqual(df['chunk'][4], 1)
        self.assertEqual(df['chunk'][5], 1)
        self.assertEqual(df['chunk'][6], 1)
        self.assertEqual(df['chunk'][7], 2)
        self.assertEqual(df['chunk'][8], 2)
        self.assertEqual(df['chunk'][9], 2)
        self.assertEqual(df['chunk'][10], -1)
        self.assertEqual(df['chunk'][11], 3)
        self.assertEqual(df['chunk'][12], 3)
        self.assertEqual(df['chunk'][13], 3)
        self.assertEqual(df['chunk'][14], -1)

        split_list = split_time_chunks(df)
        df = assign_chunk_ids(df, split_list, 3, 0, 5)

        self.assertEqual(df['chunk'][0],-1)
        self.assertEqual(df['chunk'][1], 5)
        self.assertEqual(df['chunk'][2], 5)
        self.assertEqual(df['chunk'][3], 5)
        self.assertEqual(df['chunk'][4], 6)
        self.assertEqual(df['chunk'][5], 6)
        self.assertEqual(df['chunk'][6], 6)
        self.assertEqual(df['chunk'][7], 7)
        self.assertEqual(df['chunk'][8], 7)
        self.assertEqual(df['chunk'][9], 7)
        self.assertEqual(df['chunk'][10], -1)
        self.assertEqual(df['chunk'][11], 8)
        self.assertEqual(df['chunk'][12], 8)
        self.assertEqual(df['chunk'][13], 8)
        self.assertEqual(df['chunk'][14], -1)


        split_list = split_time_chunks(df)
        df = assign_chunk_ids(df, split_list, 3, 2)
        print(split_list, df)
        self.assertEqual(df['chunk'][0], -1)
        self.assertEqual(df['chunk'][1], -1)
        self.assertEqual(df['chunk'][2], -1)
        self.assertEqual(df['chunk'][3], 0)
        self.assertEqual(df['chunk'][4], 0)
        self.assertEqual(df['chunk'][5], 0)
        self.assertEqual(df['chunk'][6], 1)
        self.assertEqual(df['chunk'][7], 1)
        self.assertEqual(df['chunk'][8], 1)
        self.assertEqual(df['chunk'][9], -1)
        self.assertEqual(df['chunk'][10], -1)
        self.assertEqual(df['chunk'][11], -1)
        self.assertEqual(df['chunk'][12], -1)
        self.assertEqual(df['chunk'][13], -1)
        self.assertEqual(df['chunk'][14], -1)


