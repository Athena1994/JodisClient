

import datetime
import os
import tempfile
import unittest

import pandas as pd

from core.data.assets.asset_manager import Asset, AssetManager

from core.data.technical_indicators.collection import IndicatorCollection


def mock_sql_get(pair: str,
                 interval: str,
                 earliest: datetime = None,
                 last: datetime = None) -> pd.DataFrame:

    if earliest is None:
        earliest = datetime.datetime(2021, 1, 1)

    return pd.DataFrame({'time': pd.date_range(start=earliest, periods=10),
                         'open': range(10),
                         'close': range(10),
                         'high': range(10),
                         'low': range(10)})


class TestDataManager(unittest.TestCase):

    def test_fetch_and_cache(self):

        tmp_dir = tempfile.mkdtemp()

        mock_conf = AssetManager.Config.from_dict({
            "chunks": {
                "tr_val_split": 0.8,
                "test_chunk_cnt": 1,
                "chunk_size": 2,
                "reserve": 0
            },
            "sources": [
                {
                    "name": "foo",
                    "type": "sql",
                    "config": {
                        "host": "host",
                        "db": "db",
                        "table": "table",
                        "user": "user",
                        "pw": "pw"
                    }
                }
            ],
            "assets": [
                {
                    "name": "bar",
                    "interval": "interval",
                    "file": f"{tmp_dir}/mock_data.csv",
                    "source": "foo"
                }
            ]
        })

        value_conf = AssetManager.Config.Provider.from_dict({
            "asset": {
                "name": "bar",
                "source": "foo",
                "interval": "interval"
            },
            "indicators": [
                {
                    "name": "AwesomeOscillator",
                    "params": {
                        "short_period": 2,
                        "long_period": 4
                    }
                }
            ],
            "include": ["volume", "open", "high", "close", "low"],
            "normalizer": {
                "df_key": "default",
                "default_strategy": {"type": "zscore"},
                "groups": [["open", "high", "close", "low"]]
            }
        })
        # assert that DataManager is initialized according to conf
        dm = AssetManager(mock_conf, False)
        self.assertEqual(len(dm._assets), 1)
        self.assertEqual(len(dm._sources), 1)
        asset = Asset('bar', 'interval', 'foo')
        self.assertTrue(asset in dm._assets)
        self.assertIsNone(dm._assets[asset])

        self.assertTrue('foo' in dm._sources)

        # assert that fetch_assets fetches data from source and caches it
        dm._sources['foo'].get = mock_sql_get
        dm.fetch_assets()
        self.assertEqual(len(dm._assets[asset]), 10)

        mock_file = f"{tmp_dir}/mock_data.csv"
        self.assertTrue(os.path.exists(mock_file))
        self.assertTrue(len(pd.read_csv(mock_file)), 10)

        # assert a fresh DataManager initializes from cache
        dm = AssetManager(mock_conf, False)
        dm._sources['foo'].get = mock_sql_get
        self.assertTrue(asset in dm._assets)
        self.assertEqual(len(dm._assets[asset]), 10)

        # assert that fetch_assets appends new data and assigns chunks
        dm.fetch_assets()
        df = dm._assets[asset]
        self.assertEqual(len(df), 19)

        for i in range(len(df)-1):
            self.assertEqual(df.iloc[i]['chunk'], i//2)
        self.assertEqual(df.iloc[18]['chunk'], -1)

        # assert training split is updated correctly
        dm.update_training_split(False)
        self.assertTrue('chunk_type' in df)
        tr_cnt = len(df[df['chunk_type'] == 0]['chunk'].unique())
        val_cnt = len(df[df['chunk_type'] == 1]['chunk'].unique())
        test_cnt = len(df[df['chunk_type'] == 2]['chunk'].unique())
        total_cnt = len([c for c in df['chunk'].unique() if c != -1])
        self.assertEqual(tr_cnt + val_cnt + test_cnt, total_cnt)
        self.assertEqual(test_cnt, 1)
        self.assertEqual(round((total_cnt - test_cnt) * 0.8), tr_cnt)

        mock_conf.chunks.tr_val_split = 0.3
        mock_conf.chunks.test_chunk_cnt = 2
        dm.update_training_split(False)
        self.assertTrue('chunk_type' in df)
        tr_cnt = len(df[df['chunk_type'] == 0]['chunk'].unique())
        val_cnt = len(df[df['chunk_type'] == 1]['chunk'].unique())
        test_cnt = len(df[df['chunk_type'] == 2]['chunk'].unique())
        total_cnt = len([c for c in df['chunk'].unique() if c != -1])
        self.assertEqual(tr_cnt + val_cnt + test_cnt, total_cnt)
        self.assertEqual(test_cnt, 1)
        self.assertEqual(round((total_cnt - test_cnt) * 0.8), tr_cnt)

        dm.update_training_split(True)
        self.assertTrue('chunk_type' in df)
        tr_cnt = len(df[df['chunk_type'] == 0]['chunk'].unique())
        val_cnt = len(df[df['chunk_type'] == 1]['chunk'].unique())
        test_cnt = len(df[df['chunk_type'] == 2]['chunk'].unique())
        total_cnt = len([c for c in df['chunk'].unique() if c != -1])
        self.assertEqual(tr_cnt + val_cnt + test_cnt, total_cnt)
        self.assertEqual(test_cnt, 2)
        self.assertEqual(round((total_cnt - test_cnt) * 0.3), tr_cnt)

        dm.update_indicators(value_conf)

        ind = IndicatorCollection.get('AwesomeOscillator')\
            .create_indicator({'short_period': 2, 'long_period': 4})
        ind_name = ind.get_unique_id()
        self.assertTrue(ind_name in dm._assets[asset])
        self.assertTrue(dm._assets[asset].loc[:2, ind_name].isna().all())
        self.assertTrue((~dm._assets[asset].loc[3:, ind_name].isna()).all())

        dm.fetch_assets()
        self.assertEqual(len(dm._assets[asset]), 28)
        self.assertTrue((~dm._assets[asset].loc[3:18, ind_name].isna()).all())
        self.assertTrue(dm._assets[asset].loc[19:, ind_name].isna().all())

        dm.update_indicators(value_conf)
        self.assertTrue(dm._assets[asset].loc[:2, ind_name].isna().all())
        self.assertTrue((~dm._assets[asset].loc[3:, ind_name].isna().all()))

        print(dm._assets[asset])

        # tmp dir cleanup
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)

    def test_get_asset_df(self):

        tmp_dir = tempfile.mkdtemp()

        mock_conf = AssetManager.Config.from_dict({
            "chunks": {
                "tr_val_split": 0.8,
                "test_chunk_cnt": 1,
                "chunk_size": 2,
                "reserve": 0
            },
            "sources": [
                {
                    "name": "foo",
                    "type": "sql",
                    "config": {
                        "host": "host",
                        "db": "db",
                        "table": "table",
                        "user": "user",
                        "pw": "pw"
                    }
                }
            ],
            "assets": [
                {
                    "name": "bar",
                    "interval": "interval",
                    "file": f"{tmp_dir}/mock_data.csv",
                    "source": "foo"
                }
            ]
        })

        dm = AssetManager(mock_conf, False)
        dm._sources['foo'].get = mock_sql_get

        provider_cfg = AssetManager.Config.Provider.from_dict({
            "asset": {
                "name": "bar",
                "source": "foo",
                "interval": "interval"
            },
            "indicators": [
                {
                    "name": "AwesomeOscillator",
                    "params": {
                        "short_period": 2,
                        "long_period": 4
                    }
                }
            ],
            "include": ["volume", "open", "high", "close", "low"],
            "normalizer": {
                "df_key": "default",
                "default_strategy": {"type": "zscore"},
                "groups": [["open", "high", "close", "low"]]
            }

        })

        df = dm.get_asset_df(provider_cfg.asset)
        self.assertIsNotNone(df)

        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)
