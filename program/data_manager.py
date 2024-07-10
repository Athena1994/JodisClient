

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from core.data.data_provider import ChunkType, DataProvider
from core.data.loader.ohcl_loader import OHCLLoader
from core.data.loader.sql_loader import SQLOHCLLoader
from core.data.technical_indicators.collection import IndicatorCollection
from core.data.technical_indicators.indicators import Indicator, IndicatorDescription
from core.data.utils import apply_indicator, assign_chunk_ids, split_time_chunks, training_split

@dataclass
class Asset:
    name: str
    interval: str
    source: str

    def __hash__(self) -> int:
        return hash(self.name) + hash(self.interval) + hash(self.source)

    @staticmethod
    def from_config(conf: dict) -> 'Asset':
        return Asset(conf['name'], conf['interval'], conf['source'])

# todo: support sources with differing timeframes
class DataManager:
    @staticmethod
    def _create_source(source_conf: dict) -> OHCLLoader:

        source_type = source_conf['type']
        if source_type == 'sql':
            return SQLOHCLLoader.from_config(source_conf['config'])
        else:
            raise ValueError(f'Unknown source type: {source_type}')

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
                first_ix = df[df['chunk'] == last_chunk_id].index.max() +1
                if first_ix >= len(df):
                    return

        sl = split_time_chunks(df)
        sl = [(ix + first_ix, l - first_ix) 
              for (ix, l) in sl if ix + l > first_ix]
        self._assets[asset] = assign_chunk_ids(df, sl, 
                         self._chunk_cfg['chunk_size'], 
                         self._chunk_cfg['reserve'],
                         last_chunk_id +1)

        self._assets[asset]['chunk'] = self._assets[asset]['chunk'].astype(int)

    def _cache_asset(self, asset: Asset) -> None:
        self._assets[asset].to_csv(self._asset_cache[asset], index=False)


    def __init__(self, conf: dict, clear_cache: bool) -> None:

        self._chunk_cfg = conf['chunk_config']

        self._sources = {}
        for source_conf in conf['sources']:
            name = source_conf['name']
            if name in self._sources:
                raise ValueError(f'Source with name {name} already exists!')
            self._sources[name] = self._create_source(source_conf)

        self._assets = {}
        self._asset_cache = {}
        for asset_conf in conf['assets']:
            asset = Asset(asset_conf['name'], 
                          asset_conf['interval'], 
                          asset_conf['source'])
            if asset in self._assets:
                raise ValueError(f'Asset {asset} already exists!')
            if asset.source not in self._sources:
                raise ValueError(f'Unknown source {asset.source} for asset {asset.name}!')

            self._asset_cache[asset] = asset_conf['file']
            if clear_cache:
                self._assets[asset] = None
            else:
                self._assets[asset] = self._load_cached_asset(asset_conf['file'])

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
            

            tr_cnt = len(df[df['chunk_type'] == ChunkType.TRAINING]['chunk'].unique())
            val_cnt = len(df[df['chunk_type'] == ChunkType.VALIDATION]['chunk'].unique())
            test_cnt = len(df[df['chunk_type'] == ChunkType.TEST]['chunk'].unique())

            first_chunk = df[(df['chunk_type'] == -1) 
                             & (df['chunk'] > -1)]['chunk'].min()

            split_list = training_split(chunk_cnt, 
                                        self._chunk_cfg['tr_val_split'], 
                                        self._chunk_cfg['test_chunk_cnt'],
                                        (tr_cnt, val_cnt, test_cnt))
            for i, chunk_type in enumerate(split_list):
                self._assets[asset].loc[df['chunk'] == i+first_chunk, 'chunk_type'] = chunk_type

            self._cache_asset(asset)

    def reset_chunks(self) -> None:
        for asset in self._assets:
            self._update_chunks(asset, True)
            self._cache_asset(asset)

    def update_indicators(self, indicator_conf: List[dict]) -> None:
        for asset_desc in indicator_conf:
            asset = Asset(asset_desc['asset']['name'], 
                          asset_desc['asset']['interval'], 
                          asset_desc['asset']['source'])
            if asset not in self._assets:
                raise ValueError(f'Asset {asset} not found!')
            
            for indicator in asset_desc['indicators']:
                apply_indicator(self._assets[asset], indicator)
            
            self._cache_asset(asset)


    def get_provider(self, agent_cfg: dict) -> DataProvider:
        if len(self._assets) != 1:
            raise ValueError('Only single asset training is supported!')
        asset = Asset.from_config(agent_cfg['data'][0]['asset'])
        return DataProvider(self._assets[asset],
                            get_required_cols(agent_cfg)[0][1],
                            'zscore')
        

def get_required_cols(agent_cfg: dict) -> List[Tuple[Asset, List[int]]]:
    asset_lists = []

    for data_desc in agent_cfg['data']:
        cols = []

        if 'volume' in data_desc and data_desc['volume']:
            cols.append('volume')

        if 'ohcl' in data_desc and data_desc['ohcl']:
            cols.append('open')
            cols.append('close')
            cols.append('high')
            cols.append('low')

        for indicator in data_desc['indicators']:
            ind = IndicatorCollection.get_from_cfg(indicator)
            cols.append(ind.get_unique_id())
        asset_lists.append((Asset.from_config(data_desc['asset']), cols))

    return asset_lists