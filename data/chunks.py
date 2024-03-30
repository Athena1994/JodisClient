import datetime

import numpy as np
import pandas as pd

import data.utils
from data.data_source import DataSource, CandleData
from data.utils import CandleInterval, get_open_close_av, correct_rate_for_fee


#from simulation.price_evaluation import get_open_close_av


class DataChunkIt:
    def __init__(self,
                 raw_data: np.ndarray,
                 time_series: np.array,
                 pre_len: int,
                 post_len: int):

        self._timestamps: np.ndarray[datetime.datetime] = time_series
        self._raw_data: np.ndarray = raw_data
        self._rich_data: np.ndarray = None

        self._first_unread_ix: int = 0
        self._current_ix_limit: int = 0

        self._pre_len: int = pre_len
        self._post_len: int = post_len

        self.reset()

    def __iter__(self):
        return self

    def __len__(self):
        return self._rich_data.shape[0] - self._pre_len

    def set_ix(self, ix):
        self._current_ix_limit = ix+1

    def __next__(self):
        if self._current_ix_limit == self._rich_data.shape[0]-1 - self._post_len:
            raise StopIteration()

        self._current_ix_limit += 1
        return self.get_current(True)

    def set_rich_data(self, data: np.ndarray):
        self._rich_data = data

    def get(self,
            ix_from: int = -1, cnt: int = -1,
            since: datetime.datetime = None,
            relative_zero: bool = True,
            restrict: bool = True,
            use_rich_data: bool = False) -> pd.DataFrame:

        if since is not None:  # use time stamp
            ix_from = sum(self._timestamps < since)
        elif ix_from == -1:  # use first unread index
            ix_from = self._first_unread_ix
        elif relative_zero:  # shift ix for chunk pre length
            ix_from += self._pre_len

        if cnt < 0:  # select all samples till current limit
            cnt = self._current_ix_limit - ix_from

        ix_till = ix_from + cnt

        if restrict and ix_till > self._current_ix_limit:
            raise Exception('Chunk restriction violated!')

        if use_rich_data:
            result = self._rich_data[ix_from:ix_till]
        else:
            result = self._raw_data[ix_from:ix_till]

        if ix_till > self._first_unread_ix:
            self._first_unread_ix = ix_till

        return result

    def get_current(self, use_rich_data: bool = False):
        if use_rich_data:
            return self._rich_data[self._current_ix_limit-1]
        else:
            return self._raw_data[self._current_ix_limit-1]

    def get_timestamp(self) -> datetime.datetime:
        return self._timestamps[self._current_ix_limit-1]

    def reset(self):
        self._current_ix_limit = self._pre_len+1

    def get_ix(self):
        return self._current_ix_limit-1


class DataChunk:

    def __init__(self, df: pd.DataFrame,
                 pre_length: int, post_length: int):
        self._df = df
        self._array = np.array(df)

        self._pre_len = pre_length
        self._post_len = post_length

    def __iter__(self) -> DataChunkIt:
        return DataChunkIt(raw_data=self._array,
                           time_series=self._array[:, CandleData.DATE],
                           pre_len=self._pre_len,
                           post_len=self._post_len)

    def get_df(self):
        return self._df


class ChunkDataSource(DataSource):
    def __init__(self, chunk: DataChunkIt):
        super().__init__()
        self._chunk = chunk
        self._fix_fee = 0
        self._rel_fee = 0

    def set_fees(self, fix, rel):
        self._fix_fee = fix
        self._rel_fee = rel

    def get_candles(self, asset: str, cur: str,
                    since: datetime.datetime,
                    timespan: str):
        return self._chunk.get(since=since)

    def get_current_candle(self, asset: str, cur: str, timespan: str):
        return self._chunk.get_current()

    def get_current_buy_price(self, asset: str, cur: str) -> float:
        candle = self.get_current_candle(asset, cur, CandleInterval.ONE_MINUTE)
        return get_open_close_av(candle[CandleData.OPEN], candle[CandleData.CLOSE])

    def get_current_sell_price(self, asset: str, cur: str) -> float:
        candle = self.get_current_candle(asset, cur, CandleInterval.ONE_MINUTE)
        return get_open_close_av(candle[CandleData.OPEN], candle[CandleData.CLOSE])


