
from typing import Tuple
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
    def _normalize_col(df: pd.DataFrame, col: str, strategy: str) \
        :
        if strategy == 'none':
            return df, {'type': 'none'}
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
            return df, {'type': 'warn', 'mean': mean, 'std': std, 'min': min_val, 'max': max_val}

        elif strategy == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
            return df, {'type': 'minmax', 'min': min_val, 'max': max_val}
        elif strategy == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            return df, {'type': 'zscore', 'mean': mean, 'std': std}
        else:
            raise ValueError(f"Normalization strategy {strategy} not supported!")

    def __init__(self, 
                 df: pd.DataFrame, 
                 required_columns: list[str],
                 normalization_strategy: str, 
                 window_size: int = 1):

        self._window_size = window_size

        self._df_original = df
        self._df_normalized = df.copy()

        # check if all required columns are present in the data
        missing_cols = [col for col in required_columns if col not in self._df_normalized]
        if len(missing_cols) > 0:
            raise ValueError(f"Columns {missing_cols} not found in data.")
        self._required_columns = required_columns



        # normalize the required columns
        self._normalization_meta = {}
        for col in self._required_columns:
            self._df_normalized, meta = DataProvider._normalize_col(self._df_normalized, 
                                                   col, 
                                                   normalization_strategy)
            self._normalization_meta[col] = meta

        chunk_ids = self._df_normalized[self._df_normalized['chunk'] != -1]['chunk'].unique()

        self._chunk_ixs \
            = [(self._df_normalized[self._df_normalized['chunk'] == chunk_id].index[0],
                len(self._df_normalized[self._df_normalized['chunk'] == chunk_id]))
              for chunk_id in chunk_ids]
        

        # lists of chunk ids with chunk type 0 (tr) 1(val) 2(test)
        self._chunk_ids = {'tr': [], 'val': [], 'test': []}
        for id in chunk_ids:
            type = self._df_normalized[self._df_normalized['chunk'] == id]['chunk_type'].iloc[0]
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
        return DataProvider.ChunkIt(self._df_normalized,
                                    self._df_original,
                                    self._required_columns,
                                    self._chunk_ids[chunk_type],
                                    self._chunk_ixs,
                                    self._window_size)

    def get_chunk_cnt(self, chunk_type: str) -> int:
        if chunk_type not in ['tr', 'val', 'test']:
            raise ValueError(f"Chunk type {chunk_type} not supported.")
        return len(self._chunk_ids[chunk_type])

    def get_normalization_info(self) -> dict:
        return self._normalization_meta

    class ChunkIt:
        def __init__(self, 
                     df_normalized: pd.DataFrame,
                     df_orginal: pd.DataFrame, 
                     required_columns: list[str],
                     ids: list[int],
                     chunk_ixs: list[tuple[int, int]],
                     window_size: int):
            self._df_normalized = df_normalized
            self._df_original = df_orginal
            self._chunk_ixs = chunk_ixs
            self._chunk_ids = ids
            self._required_columns = required_columns
            self._ix = 0
            self._window_size = window_size

        def __iter__(self):
            return self

        """Returns the next chunk of data and the original data."""
        def __next__(self) -> 'ChunkReader':
            if self._ix == len(self._chunk_ids):
                raise StopIteration

            chunk_id = self._chunk_ids[self._ix]
            chunk_ix = self._chunk_ixs[chunk_id]

            self._ix += 1

            df_n = self._df_normalized.iloc[chunk_ix[0]:chunk_ix[0]+chunk_ix[1]][self._required_columns]
            df_o = self._df_original.iloc[chunk_ix[0]:chunk_ix[0]+chunk_ix[1]]

            return ChunkReader(torch.tensor(df_n.values, dtype=torch.float32),
                               df_o, self._window_size)

class ChunkReader:
    def __init__(self, 
                chunk: torch.Tensor, 
                df: pd.DataFrame,
                window_size: int) -> None:
        self._df = df
        self._chunk = chunk
        self._window_size = window_size

        self._ix = 0

    def get_chunk(self) -> Tuple[torch.Tensor, pd.DataFrame]:
        return self._chunk, self._df

    def __len__(self) -> int:
        return len(self._chunk) - self._window_size + 1

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, pd.DataFrame]:
        if self._ix + self._window_size > len(self._chunk):
            raise StopIteration

        tensor_state = self._chunk[self._ix:self._ix+self._window_size]
        df_state = self._df.iloc[self._ix+self._window_size-1]

        self._ix += 1

        return tensor_state, df_state
