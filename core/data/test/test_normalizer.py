
import unittest

import numpy as np
import pandas as pd

from core.data.normalizer import Normalizer


class TestNormalizer(unittest.TestCase):

    def test_stats(self):
        data = [1, 2, 3, 4, 5]
        stats = Normalizer.Stats.from_array(data)
        self.assertEqual(stats.mean, 3)
        d = np.array(data)
        self.assertAlmostEqual(stats.std,
                               np.sqrt(np.sum(np.square(d-d.mean()))/len(d)))
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 5)

        data = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        stats = Normalizer.Stats.from_array(data['a'])
        self.assertEqual(stats.mean, 3)
        d = np.array(data['a'])
        self.assertAlmostEqual(stats.std,
                               np.sqrt(np.sum(np.square(d-d.mean()))/len(d)))
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 5)

    def test_normalizer_base(self):
        df_b = pd.DataFrame({
            'c': range(1, 6),
            'd': range(6, 11)
        })

        stats_b = {
            'c': Normalizer.Stats.from_array(df_b['c']),
            'd': Normalizer.Stats.from_array(df_b['d'])
        }

        normalizer = Normalizer()

        cfg = Normalizer.Config.from_dict({})
        normalizer.normalize_df(df_b, cfg)
        cfg = Normalizer.Config.from_dict({'default_strategy': None})
        normalizer.normalize_df(df_b, cfg)

        cfg = Normalizer.Config.from_dict({'default_strategy': {
                                                "type": 'zscore'}})
        try:
            normalizer.normalize_df(df_b, cfg)
            self.fail("Should have raised an exception")
        except ValueError:
            pass

        cfg = Normalizer.Config.from_dict({'default_strategy': {
                                                'type': 'zscore'},
                                           'df_key': 'foo'})
        try:
            normalizer.normalize_df(df_b, cfg)
            self.fail("Should have raised an exception")
        except ValueError:
            pass

        normalizer.prepare(df_b, cfg)

        self.assertTrue('foo' in normalizer._stats)
        self.assertTrue('c' in normalizer._stats['foo'])
        self.assertTrue('d' in normalizer._stats['foo'])
        self.assertEqual(normalizer._stats['foo']['c'].mean,
                         stats_b['c'].mean)
        self.assertEqual(normalizer._stats['foo']['c'].std,
                         stats_b['c'].std)
        self.assertEqual(normalizer._stats['foo']['d'].mean,
                         stats_b['d'].mean)
        self.assertEqual(normalizer._stats['foo']['d'].std,
                         stats_b['d'].std)

        df_norm = normalizer.normalize_df(df_b, cfg)

        ca = np.array(df_b['c'])
        ca = (ca - ca.mean())/ca.std()
        da = np.array(df_b['d'])
        da = (da - da.mean())/da.std()

        for i in range(5):
            self.assertEqual(df_norm['c'][i], ca[i])
            self.assertEqual(df_norm['d'][i], da[i])

        self.assertEqual(df_norm['c'].mean(), 0)
        self.assertAlmostEqual(np.array(df_norm['c']).std(), 1)

        self.assertEqual(df_norm['d'].mean(), 0)
        self.assertAlmostEqual(df_norm['d'].std(ddof=0), 1)

    def test_normalizer_groups(self):

        df_a = pd.DataFrame({
            'a': range(1, 6),
            'b': range(6, 11),
            'c': range(11, 16)
        })

        normalizer = Normalizer()
        cfg = Normalizer.Config.from_dict({
            'df_key': 'foo',
            'groups': [['a', 'b']]
        })
        normalizer.prepare(df_a, cfg)

        ab = np.arange(1, 11)

        self.assertTrue('foo' in normalizer._stats)
        self.assertTrue('a' in normalizer._stats['foo'])
        self.assertTrue('b' in normalizer._stats['foo'])
        self.assertTrue('c' in normalizer._stats['foo'])
        self.assertEqual(normalizer._stats['foo']['a'].mean,
                         ab.mean())
        self.assertEqual(normalizer._stats['foo']['a'].std,
                         ab.std())
        self.assertEqual(normalizer._stats['foo']['b'].mean,
                         ab.mean(),
                         ab.std())
        self.assertEqual(normalizer._stats['foo']['c'].mean,
                         df_a['c'].mean())
        self.assertEqual(normalizer._stats['foo']['c'].std,
                         df_a['c'].std(ddof=0))

    def test_normalizer_extra(self):

        df_a = pd.DataFrame({
            'a': range(1, 6),
            'b': range(6, 11),
            'c': range(11, 16)
        })
        df_b = pd.DataFrame({
            'c': range(1, 6),
            'd': range(6, 11)
        })

        normalizer = Normalizer()

        cfg = Normalizer.Config.from_dict({
            'df_key': 'foo',
            'groups': [['a', 'b']]
        })
        normalizer.prepare(df_a, cfg)
        self.assertEqual(normalizer._stats['foo']['c'].mean,
                         np.arange(11, 16).mean())
        self.assertEqual(normalizer._stats['foo']['c'].std,
                         np.arange(11, 16).std())

        mock_conf = {
            'extra': [
                {
                    'column': 'c',
                    "strategy": None,
                }
            ],
        }
        cfg = Normalizer.Config.from_dict(mock_conf)
        df_norm = normalizer.normalize_df(df_b, cfg)
        for col in df_b.columns:
            for i in range(5):
                self.assertEqual(df_b[col][i], df_norm[col][i])

        mock_conf['extra'][0]['strategy'] = {'type': 'zscore'}
        cfg = Normalizer.Config.from_dict(mock_conf)
        try:
            df_norm = normalizer.normalize_df(df_b, cfg)
            self.fail("Should have raised an exception")
        except ValueError:
            pass

        mock_conf['df_key'] = 'bar'
        cfg = Normalizer.Config.from_dict(mock_conf)
        try:
            df_norm = normalizer.normalize_df(df_b, cfg)
            self.fail("Should have raised an exception")
        except Exception:
            pass

        mock_conf['df_key'] = 'foo'
        cfg = Normalizer.Config.from_dict(mock_conf)
        df_norm = normalizer.normalize_df(df_b, cfg)

        del mock_conf['df_key']
        mock_conf['extra'][0]['strategy'] = {'type': 'zscore'}
        mock_conf['extra'][0]['stats_df'] = 'foo'

        cfg = Normalizer.Config.from_dict(mock_conf)
        df_norm = normalizer.normalize_df(df_b, cfg)
        self.assertTrue('c' in df_norm.columns)
        self.assertTrue('d' in df_norm.columns)
        self.assertEqual(len(df_norm.columns), 2)

        c_norm = np.arange(1, 6)
        t = np.arange(11, 16)
        c_norm = (c_norm - t.mean())/t.std()
        for i in range(5):
            self.assertEqual(df_norm['c'][i], c_norm[i])
            self.assertEqual(df_norm['d'][i], df_b['d'][i])

        mock_conf['extra'][0]['stats_col'] = 'a'
        cfg = Normalizer.Config.from_dict(mock_conf)
        df_norm = normalizer.normalize_df(df_b, cfg)

        c_norm = np.arange(1, 6)
        t = np.arange(1, 11)
        c_norm = (c_norm - t.mean())/t.std()
        for i in range(5):
            self.assertEqual(df_norm['c'][i], c_norm[i])
            self.assertEqual(df_norm['d'][i], df_b['d'][i])

        mock_conf['df_key'] = 'bar'
        mock_conf['default_strategy'] = {"type": 'zscore'}

        cfg = Normalizer.Config.from_dict(mock_conf)
        normalizer.prepare(df_b, cfg)
        df_norm = normalizer.normalize_df(df_b, cfg)
        c_norm = np.arange(1, 6)
        t = np.arange(1, 11)
        c_norm = (c_norm - t.mean())/t.std()
        for i in range(5):
            self.assertEqual(df_norm['c'][i], c_norm[i])

        d_norm = np.arange(6, 11)
        d_norm = (d_norm - d_norm.mean())/d_norm.std()

        for i in range(5):
            self.assertEqual(df_norm['d'][i], d_norm[i])

    def test_formula_strategy(self):
        df = pd.DataFrame({
            'a': [1/4, 1/2, 2/3, 3/4, 9/10, 9.5/10, 9.9/10, 1.01, 1.05, 1.10,
                  1.15, 1.2, 1.35, 1.5, 1.75, 2],
        })

        mock_config = {
            'extra': [
                {
                    'column': 'a',
                    'strategy': {
                        'type': 'formula',
                        'params': {
                            'expression': 'np.log(x)/np.log(1.10)',
                        }
                    },
                }
            ]
        }

        norm = Normalizer()

        cfg = Normalizer.Config.from_dict(mock_config)
        ndf = norm.normalize_df(df, cfg)

        print(df)
        print(ndf)
