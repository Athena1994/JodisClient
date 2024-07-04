
import unittest

import pandas as pd

from data.utils import mark_chunks, split_time_chunks


class TestUtils(unittest.TestCase):
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

