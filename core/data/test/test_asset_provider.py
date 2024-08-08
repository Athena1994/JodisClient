import unittest

import pandas as pd
from torch import Tensor
import torch
from core.data.assets.asset_provider import AssetProvider
from core.data.assets.asset_source import AssetSource
from core.data.normalizer import Normalizer
from core.data.data_provider import ChunkType


class MockNormalizer:
    conf = Normalizer.Config.from_dict({"a": 2})

    def __init__(self, test: unittest.TestCase, off=False) -> None:
        self.test = test
        self.expected_cols = []
        self.normalized = False
        self.expect_normalize = False
        self.off = off

    def prepare(self, df: pd.DataFrame, conf: Normalizer.Config) -> None:
        if self.off:
            return
        self.test.assertEqual(conf, self.conf)

    def set_expected_cols(self, cols: list[str]) -> None:
        self.expected_cols = cols

    def normalize_data(self, df: pd.DataFrame, conf: Normalizer.Config)\
            -> pd.DataFrame:
        if self.off:
            return df

        if not self.expect_normalize:
            self.test.fail("Unexpected call to normalize_df")
        self.normalized = True
        self.test.assertEqual(conf, self.conf)
        self.test.assertListEqual(list(df.columns), self.expected_cols)
        return df


class TestAssetSource(unittest.TestCase):
    def test(self):
        df = pd.DataFrame({
            'date': pd.date_range(start='1/1/2021', periods=100),
            'up': range(100),
            'down': range(100, 0, -1),
            'chunk': [i // 10 for i in range(100)],
            'chunk_type': [0]*80 + [1]*10 + [2]*10,
        })

        normalizer = MockNormalizer(self)
        src = AssetSource(df, normalizer,
                          Normalizer.Config.from_dict({"a": 2}))

        normalizer.expect_normalize = False
        res = src.get_data([
            AssetSource.DataFrameRequirement(
                key='a',
                columns=['up', 'down'],
                normalize=False
            ),
            AssetSource.DataFrameRequirement(
                key='b',
                columns=['up', 'date'],
                normalize=False
            )
        ])
        self.assertTrue('a' in res)
        self.assertTrue('b' in res)
        self.assertListEqual(list(res['a'].columns), ['up', 'down'])
        self.assertListEqual(list(res['b'].columns), ['up', 'date'])

        try:
            src.get_data([
                AssetSource.DataFrameRequirement(
                    key='a',
                    columns=['up', 'foo'],
                    normalize=False
                )])
            self.fail("Should raise exception")
        except ValueError:
            pass

        normalizer.expect_normalize = True
        normalizer.normalized = False
        normalizer.set_expected_cols(['up', 'down'])
        res = src.get_data([
            AssetSource.DataFrameRequirement(
                key='a',
                columns=['up', 'down'],
                normalize=True
            )])
        self.assertTrue(normalizer.normalized)


class TestAssetProvider(unittest.TestCase):
    def test_get_data(self):

        df = pd.DataFrame(
            columns=['date', 'value', 'chunk', 'chunk_type', 'dummy'],
            data=[
                ['2021-01-01', 1, -1, 0, 21],
                ['2021-01-02', 2, -1, 0, 20],

                ['2021-01-03', 3, 0, 1, 19],
                ['2021-01-04', 4, 0, 1, 18],

                ['2021-01-05', 5, 1, 0, 17],
                ['2021-01-06', 6, 1, 0, 16],
                ['2021-01-07', 7, 1, 0, 15],

                ['2021-01-08', 8, 2, 0, 14],
                ['2021-01-09', 9, 2, 0, 13],

                ['2021-01-10', 10, -1, 0, 12],

                ['2021-01-11', 11, 3, 2, 11],

                ['2021-01-12', 12, 4, 1, 10],
                ['2021-01-13', 13, 4, 1, 9],
                ['2021-01-14', 14, 4, 1, 8],
                ['2021-01-15', 15, 4, 1, 7],

                ['2021-01-16', 16, 5, 0, 6],

                ['2021-01-17', 17, -1, 0, 5],

                ['2021-01-18', 18, 6, 2, 4],
                ['2021-01-19', 19, 6, 2, 3],
                ['2021-01-20', 20, 6, 2, 2],
                ['2021-01-21', 21, 6, 2, 1],])

        src = AssetSource(df, MockNormalizer(self, off=True),
                          Normalizer.Config.from_dict({}))

        try:
            AssetProvider(src, ['value', 'foo'], [], 1)
            self.fail("Should raise exception")
        except ValueError:
            pass

        data_provider = AssetProvider(src,
                                      ['value', 'dummy', 'date'],
                                      ['value', 'dummy'], 1)

        self.assertEqual(data_provider.get_chunk_cnt(ChunkType.TRAINING), 3)
        self.assertEqual(data_provider.get_chunk_cnt(ChunkType.VALIDATION), 2)
        self.assertEqual(data_provider.get_chunk_cnt(ChunkType.TEST), 2)

        tr_it = data_provider.get_iterator(ChunkType.TRAINING)
        val_it = data_provider.get_iterator(ChunkType.VALIDATION)
        test_it = data_provider.get_iterator(ChunkType.TEST)

        self.assertIsNotNone(tr_it)
        self.assertIsNotNone(val_it)
        self.assertIsNotNone(test_it)

        self.assertEqual(len(tr_it), 3)
        self.assertEqual(len(val_it), 2)
        self.assertEqual(len(test_it), 2)

        tr_list = [(chunk_reader._tensor,
                    chunk_reader._context) for chunk_reader in tr_it]
        val_list = [(chunk_reader._tensor,
                    chunk_reader._context) for chunk_reader in val_it]
        test_list = [(chunk_reader._tensor,
                     chunk_reader._context) for chunk_reader in test_it]

        self.assertEqual(len(tr_list), 3)
        self.assertEqual(len(val_list), 2)
        self.assertEqual(len(test_list), 2)

        self.assertListEqual(tr_list[0][1].columns.tolist(),
                             ['value', 'dummy', 'date'])

        self.assertTrue(torch.equal(tr_list[0][0],
                                    Tensor([[5, 17], [6, 16], [7, 15]])))
        self.assertTrue(torch.equal(tr_list[1][0],
                                    Tensor([[8, 14], [9, 13]])))
        self.assertTrue(torch.equal(tr_list[2][0],
                                    Tensor([[16, 6]])))

        self.assertTrue(torch.equal(val_list[0][0],
                                    Tensor([[3, 19], [4, 18]])))
        self.assertTrue(torch.equal(val_list[1][0],
                                    Tensor([[12, 10], [13, 9], [14, 8],
                                            [15, 7], ])))

        self.assertTrue(torch.equal(test_list[0][0],
                                    Tensor([[11, 11]])))
        self.assertTrue(torch.equal(test_list[1][0],
                                    Tensor([[18, 4], [19, 3], [20, 2],
                                            [21, 1]])))

    def test_chunk_reader(self):

        df = pd.DataFrame(
            {
                "date": pd.date_range(start='1/1/2021', periods=100),
                "up": range(100),
                "down": range(100, 0, -1),
                "chunk": [i // 10 for i in range(100)],
                "chunk_type": [0]*80 + [1]*10 + [2]*10,
            }
        )

        src = AssetSource(df, MockNormalizer(self, off=True), {})

        data_provider = AssetProvider(src, ['up', 'down'], ['up', 'down'], 3)
        self.assertEqual(data_provider.get_chunk_cnt(ChunkType.TRAINING), 8)
        self.assertEqual(data_provider.get_chunk_cnt(ChunkType.VALIDATION), 1)
        self.assertEqual(data_provider.get_chunk_cnt(ChunkType.TEST), 1)

        chunk_it = data_provider.get_iterator(ChunkType.TRAINING)

        chunk_reader = next(chunk_it)

        tensor = chunk_reader._tensor
        df = chunk_reader._context

        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), 10)
        self.assertEqual(len(tensor), 10)

        for i, s in enumerate(chunk_reader):
            t = s.tensor
            d = s.context
            self.assertTrue(isinstance(t, torch.Tensor))
            self.assertTrue(isinstance(d, pd.Series))
            self.assertEqual(len(t), 3)
            self.assertEqual(d['up'], i+2)
            self.assertEqual(d['down'], 100-i-2)

        i = 12
        for r in chunk_it:
            self.assertEqual(len(r), 8)
            for s in r:
                t, d = s.tensor, s.context

                self.assertEqual(len(t), 3)
                self.assertEqual(d['up'], i)
                self.assertEqual(d['down'], 100-i)
                i += 1
            i += 2
