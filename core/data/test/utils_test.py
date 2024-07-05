
import unittest

import pandas as pd

from core.data.technical_indicators.indicators import IndicatorDescription, IndicatorParameterDescription, IndicatorPrototype
from data.utils import find_indicator_update_regions, mark_chunks, split_time_chunks


class DummyIndicator(IndicatorPrototype):

    def __init__(self, skip_f):
        super().__init__()

        self.params = [
                IndicatorParameterDescription("a", 0, 10, 4, 'int'),
                IndicatorParameterDescription("b", -1, 1, 0.2, 'float'),
            ]
        self._skip_f = skip_f

    def apply_to_df(self, params: dict, df: pd.DataFrame):
        df.at[params[self._skip_f]:, hash(self)] = 0

    def get_descriptor(self) -> IndicatorDescription:
        return IndicatorDescription(
            "dummy",
            params=self.params,
            fct=self.apply_to_df,
            norm_factor=3,
            skip_field=self._skip_f
        )


class TestUtils(unittest.TestCase):
    
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
        df.at[3, indicator_hash] = 1
        df.at[4, indicator_hash] = 1
        df.at[5, indicator_hash] = 1
        df.at[6, indicator_hash] = 1
        df.at[13, indicator_hash] = 1
        update_list = find_indicator_update_regions(df, indicator_hash, 2)
        self.assertEqual(len(update_list), 2)
        self.assertEqual(update_list[0], (5, 6))
        self.assertEqual(update_list[1], (12, 3))

        # test with all values not null in one time frame
        df.at[14, indicator_hash] = 1
        update_list = find_indicator_update_regions(df, indicator_hash, 2)
        self.assertEqual(len(update_list), 1)
        self.assertEqual(update_list[0], (5, 6))


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

    def test_mark_chunks(self):
        df = pd.concat([pd.DataFrame({'time': pd.date_range(start='1/1/2024', periods=1, freq='min')}),
        pd.DataFrame({'time': pd.date_range(start='1/10/2024', periods=10, freq='min')}),
        pd.DataFrame({'time': pd.date_range(start='3/5/2024', periods=4, freq='min')})])
        df = df.reset_index(drop=True)

        split_list = split_time_chunks(df)
        df = mark_chunks(df, split_list, 3, 0)

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
        df = mark_chunks(df, split_list, 3, 2)

        self.assertEqual(df['chunk'][0],-1)
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

