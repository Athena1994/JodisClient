

import numpy as np
import pandas as pd
import torch

class ChunkType:
    TRAINING = 0
    VALIDATION = 1
    TEST = 2

class DataProvider:

    @staticmethod
    def load_csv(file: str) -> pd.DataFrame:
        return pd.read_csv(file)

    @staticmethod
    def _normalize_col(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
        if strategy == 'none':
            return df
        elif strategy == 'warn':
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()

            if abs(mean) > 0.1 and abs(mean - 0.5) > 0.1:
                print(f"Warning: mean of column {col} is {mean}")
            if abs(std -1) > 0.5:
                print(f"Warning: std of column {col} is {std}")
            if abs(min_val) > 1.5 or abs(max_val) > 1.5:
                print(f"Warning: min/max of column {col} is {min_val}/{max_val}")
            return df

        elif strategy == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            return df
        elif strategy == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            return df
        else:
            raise ValueError(f"Normalization strategy {strategy} not supported!")

    def __init__(self, 
                 df: pd.DataFrame, 
                 required_columns: list[str],
                 normalization_strategy: str):

        self._df = df

        missing_cols = [col for col in required_columns if col not in self._df]
        if len(missing_cols) > 0:
            raise ValueError(f"Columns {missing_cols} not found in data.")
        self._required_columns = required_columns

        for col in self._required_columns:
            self._df = DataProvider._normalize_col(self._df, 
                                                   col, 
                                                   normalization_strategy)

        chunk_ids = self._df[self._df['chunk'] != -1]['chunk'].unique()

        self._chunk_ixs \
            = [(self._df[self._df['chunk'] == chunk_id].index[0],
                len(self._df[self._df['chunk'] == chunk_id]))
              for chunk_id in chunk_ids]
        

        # lists of chunk ids with chunk type 0 (tr) 1(val) 2(test)
        self._chunk_ids = {'tr': [], 'val': [], 'test': []}
        for id in chunk_ids:
            type = self._df[self._df['chunk'] == id]['chunk_type'].iloc[0]
            if type == ChunkType.TRAINING:
                self._chunk_ids['tr'].append(id)
            elif type == ChunkType.VALIDATION:
                self._chunk_ids['val'].append(id)
            elif type == ChunkType.TEST:
                self._chunk_ids['test'].append(id)

    # returns an iterator over the data chunks of the specified type with 
    def get_iterator(self, chunk_type: str) -> 'ChunkIt':
        if chunk_type not in ['tr', 'val', 'test']:
            raise ValueError(f"Chunk type {chunk_type} not supported.")
        return DataProvider.ChunkIt(self._df,
                                    self._required_columns,
                                    self._chunk_ids[chunk_type],
                                    self._chunk_ixs)

    def get_chunk_cnt(self, chunk_type: str) -> int:
        if chunk_type not in ['tr', 'val', 'test']:
            raise ValueError(f"Chunk type {chunk_type} not supported.")
        return len(self._chunk_ids[chunk_type])


    class ChunkIt:
        def __init__(self, 
                     df: pd.DataFrame, 
                     required_columns: list[str],
                     ids: list[int],
                     chunk_ixs: list[tuple[int, int]]):
            self._df = df
            self._chunk_ixs = chunk_ixs
            self._chunk_ids = ids
            self._required_columns = required_columns
            self._ix = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._ix == len(self._chunk_ids):
                raise StopIteration

            chunk_id = self._chunk_ids[self._ix]
            chunk_ix = self._chunk_ixs[chunk_id]

            self._ix += 1

            df = self._df.iloc[chunk_ix[0]:chunk_ix[0]+chunk_ix[1]][self._required_columns]

            return torch.tensor(df.values, dtype=torch.float32)
