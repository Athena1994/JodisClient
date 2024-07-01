from dataclasses import dataclass
from typing import List, Tuple
from uuid import UUID

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from interfaces.trading_interfaces import Environment


class DataLoader:
    def load_history(self, asset: str, timespan: str)\
            -> DataFrame:
        pass


class DataChunk:
    def get(self, ix: int, cnt: int):
        pass


class DataStore:
    def set_loader(self, loader: DataLoader):
        pass

    def load_history(self, pair: str, timeframe: str):
        pass

    def get_chunk(self,
                  pair: str, timeframe: str,
                  ix_from: int, ix_to: int,
                  preamble_len: int, appendix: int) -> DataChunk:
        pass

@dataclass
class DataSplits:
    training_chunks: List[Tuple[int, int]]
    validation_chunks: List[DataChunk]
    test_chunks: List[DataChunk]

class DataSplitter:
    def split(self, df: pd.DataFrame) -> DataSplits:
        pass


class SimulatedEnvironment(Environment):
