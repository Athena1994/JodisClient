from dataclasses import dataclass
from typing import List
import pandas as pd
from core.data.loader.ohcl_loader import OHCLLoader
from core.data.loader.sql_loader import SQLOHCLLoader
from core.data.normalizer import Normalizer
from core.data.utils import (apply_indicator, assign_chunk_ids,
                             split_time_chunks, training_split)
from utils.config_utils import assert_fields_in_dict


@dataclass
class Asset:
    name: str
    interval: str
    source: str

    def __hash__(self) -> int:
        return hash(self.name) + hash(self.interval) + hash(self.source)

    @staticmethod
    def from_config(conf: dict) -> 'Asset':
        assert_fields_in_dict(conf, ['name', 'interval', 'source'])
        return Asset(conf['name'], conf['interval'], conf['source'])


# todo: support sources with differing timeframes
class AssetManager:

    @dataclass
    class Config:

        @dataclass
        class Provider:
            asset: Asset
            include: List[str]
            indicators: List[dict]
            normalizer: Normalizer.Config

            @staticmethod
            def from_dict(conf: dict) \
                    -> 'AssetManager.Config.Provider':
                assert_fields_in_dict(conf, ['asset', 'include', 'indicators',
                                             'normalizer'])
                return AssetManager.Config.Provider(
                    Asset.from_config(conf['asset']),
                    conf['include'],
                    conf['indicators'],
                    Normalizer.Config.from_dict(conf['normalizer'])
                )

        @dataclass
        class Chunks:
            tr_val_split: float
            test_chunk_cnt: int
            chunk_size: int
            reserve: int

            @staticmethod
            def from_dict(conf: dict) -> 'AssetManager.Config.Chunks':
                assert_fields_in_dict(conf, ['tr_val_split',
                                             'test_chunk_cnt',
                                             'chunk_size'])
                return AssetManager.Config.Chunks(
                    conf['tr_val_split'],
                    conf['test_chunk_cnt'],
                    conf['chunk_size'],
                    conf.get('reserve', 0))

        @dataclass
        class Source:
            name: str
            type: str
            config: dict

            @staticmethod
            def from_dict(conf: dict) -> 'AssetManager.Config.Source':
                assert_fields_in_dict(conf, ['name', 'type', 'config'])
                return AssetManager.Config.Source(conf['name'],
                                                  conf['type'],
                                                  conf['config'])

        @dataclass
        class Asset:
            name: str
            interval: str
            source: str
            file: str

            @staticmethod
            def from_dict(conf: dict) -> 'AssetManager.Config.Asset':
                assert_fields_in_dict(conf, ['name', 'interval', 'source',
                                             'file'])
                return AssetManager.Config.Asset(conf['name'],
                                                 conf['interval'],
                                                 conf['source'],
                                                 conf['file'])

        chunks: Chunks
        sources: List[Source]
        assets: List[Asset]

        @staticmethod
        def from_dict(conf: dict) -> 'AssetManager.Config':
            assert_fields_in_dict(conf, ['chunks', 'sources', 'assets'])
            return AssetManager.Config(
                AssetManager.Config.Chunks.from_dict(conf['chunks']),
                [AssetManager.Config.Source.from_dict(s)
                 for s in conf['sources']],
                [AssetManager.Config.Asset.from_dict(a)
                 for a in conf['assets']]
            )

    def __init__(self, cfg: Config, clear_cache: bool) -> None:

        self._chunk_cfg = cfg.chunks

        self._sources = {}
        for src_cnf in cfg.sources:
            if src_cnf.name in self._sources:
                raise ValueError(f'Source with name {src_cnf.name} already '
                                 'exists!')
            self._sources[src_cnf.name] = self._create_source(src_cnf)

        self._assets = {}
        self._asset_cache = {}
        for asset_cfg in cfg.assets:
            asset = Asset(asset_cfg.name,
                          asset_cfg.interval,
                          asset_cfg.source)
            if asset in self._assets:
                raise ValueError(f'Asset {asset} already exists!')
            if asset.source not in self._sources:
                raise ValueError(f'Unknown source {asset.source} for asset '
                                 f'{asset.name}!')

            self._asset_cache[asset] = asset_cfg.file
            if clear_cache:
                self._assets[asset] = None
            else:
                self._assets[asset] = self._load_cached_asset(asset_cfg.file)

    def fetch_assets(self) -> None:
        for asset in self._assets:
            df = self._assets[asset]
            if df is None:
                df = self._sources[asset.source].get(asset.name, asset.interval)
                df['chunk'] = -1
                df['chunk_type'] = -1
            else:
                df['time'] = pd.to_datetime(df['time'])
                new_data = self._sources[asset.source].get(asset.name,
                                                           asset.interval,
                                                           df['time'].max())
                if len(new_data) == 0:
                    continue

                df = pd.concat([df, new_data]) \
                       .drop_duplicates(subset=['time']) \
                       .reset_index(drop=True)

                df.loc[df['chunk_type'].isna(), 'chunk_type'] = -1

            df['chunk_type'] = df['chunk_type'].astype(int)
            self._assets[asset] = df

            self._update_chunks(asset, False)

            self._cache_asset(asset)

    def update_training_split(self, reset: bool) -> None:
        for asset in self._assets:
            df = self._assets[asset]
            chunk_cnt = df['chunk'].max() + 1

            if reset or 'chunk_type' not in df:
                df['chunk_type'] = -1

            ct = df['chunk_type']
            ch = df['chunk']
            tr_cnt = len(ch[ct == 0].unique())
            val_cnt = len(ch[ct == 1].unique())
            test_cnt = len(ch[ct == 2].unique())

            first_chunk = df[(ct == -1)
                             & (df['chunk'] > -1)]['chunk'].min()

            split_list = training_split(chunk_cnt,
                                        self._chunk_cfg.tr_val_split,
                                        self._chunk_cfg.test_chunk_cnt,
                                        (tr_cnt, val_cnt, test_cnt))
            for i, chunk_type in enumerate(split_list):
                self._assets[asset].loc[df['chunk'] == i+first_chunk,
                                        'chunk_type'] = chunk_type

            self._cache_asset(asset)

    def reset_chunks(self) -> None:
        for asset in self._assets:
            self._update_chunks(asset, True)
            self._cache_asset(asset)

    def update_indicators(self,
                          provider_config: Config.Provider) -> None:
        asset = Asset(provider_config.asset.name,
                      provider_config.asset.interval,
                      provider_config.asset.source)
        if asset not in self._assets:
            raise ValueError(f'Asset {asset} not found!')

        for indicator in provider_config.indicators:
            apply_indicator(self._assets[asset], indicator)

        self._cache_asset(asset)

    def get_asset_df(self, asset: Asset) -> pd.DataFrame:
        if asset not in self._assets:
            raise ValueError(f'Asset {asset} not found!')

        if self._assets[asset] is None:
            self.fetch_assets()

        return self._assets[asset]

        # if len(self._assets) != 1:
        #     raise ValueError('Only single asset training is supported!')

        # include_cols = provider_cfg.include.copy()

        # indicator_cols = []
        # for indicator in provider_cfg.indicators:
        #     ind = IndicatorCollection.get_from_cfg(indicator)
        #     indicator_cols.append(ind.get_unique_id())

        # return AssetProvider(self._assets[provider_cfg.asset],
        #                      include_cols,
        #                      include_cols + indicator_cols,
        #                      window_size)'

    @staticmethod
    def _create_source(cfg: Config.Source) -> OHCLLoader:

        if cfg.type == 'sql':
            return SQLOHCLLoader.from_config(cfg.config)
        else:
            raise ValueError(f'Unknown source type: {cfg.type}')

    @staticmethod
    def _load_cached_asset(file: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(file)
        except FileNotFoundError:
            return None
        return data

    def _update_chunks(self, asset: Asset, reset: bool) -> None:
        df = self._assets[asset]

        first_ix = 0
        last_chunk_id = -1

        if 'chunk' in df and not reset:
            last_chunk_id = df['chunk'].max()
            if last_chunk_id != -1:
                first_ix = df[df['chunk'] == last_chunk_id].index.max() + 1
                if first_ix >= len(df):
                    return

        sl = split_time_chunks(df)
        sl = [(ix + first_ix, l - first_ix)
              for (ix, l) in sl if ix + l > first_ix]
        self._assets[asset] = assign_chunk_ids(
            df, sl,
            self._chunk_cfg.chunk_size,
            self._chunk_cfg.reserve,
            last_chunk_id + 1
        )

        self._assets[asset]['chunk'] = self._assets[asset]['chunk'].astype(int)

    def _cache_asset(self, asset: Asset) -> None:
        self._assets[asset].to_csv(self._asset_cache[asset], index=False)
