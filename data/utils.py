import random
import numpy as np
import pandas as pd
import typing

from data.chunk_type import ChunkType


class CandleInterval:
    ONE_MINUTE = "ONE_MINUTE"
    TEN_MINUTES = "TEN_MINUTES"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"


# v: every split contains at least an extra of {pre_timespan} samples at the beginning, they might overlap
# min_sample_cnt: minimal length of each split excluding pre_timespan
# tr_val_ratio: ratio between training and validation size (val/tr)
# test set is always taken from the end with a minimal length of min_sample_cnt
def split_df(df: pd.DataFrame,
             pre_sample_cnt: int, min_sample_cnt: int, post_sample_cnt: int,
             tr_val_ratio: float) -> [(int, int, int)]:
    split_list = []
    ix = pre_sample_cnt
    remaining_cnt = len(df) - pre_sample_cnt - min_sample_cnt

    has_val_chunk = False

    while remaining_cnt > min_sample_cnt:
        # determine chunk size (adjacent chunks of same time are joined)
        # ensure every chunk has a different size to break time correlation
        chunk_len = min([int(min_sample_cnt * (1 + 0.5*random.random())), remaining_cnt])

        # ensure there is at least one validation chunk
        if not has_val_chunk and (remaining_cnt - chunk_len < min_sample_cnt):
            chunk_type = ChunkType.VALIDATION
        else:
            chunk_type = (ChunkType.TRAINING, ChunkType.VALIDATION)[random.random() <= tr_val_ratio]

        if chunk_type == ChunkType.VALIDATION:
            has_val_chunk = True

        # add new split
        start_ix = ix - pre_sample_cnt
        ix += chunk_len
        split_list += [(start_ix, ix+post_sample_cnt, chunk_type)]

        remaining_cnt -= chunk_len

    # add test chunk at end
    split_list += [(ix - pre_sample_cnt, len(df), ChunkType.TEST)]

    # join adjacent splits
    i = 0
    while i < len(split_list) - 1:
        if split_list[i][2] == split_list[i + 1][2]:
            split_list[i] = (split_list[i][0], split_list[i + 1][1], split_list[i + 1][2])
            del split_list[i + 1]
        else:
            i += 1

    return split_list


def prepare_feature_vec(df: pd.DataFrame,
                        ind_df: pd.DataFrame,
                        norms: typing.List[float],
                        raise_on_nan: bool) -> pd.DataFrame:

    # merge dataframes
    result_df = pd.DataFrame.merge(df, ind_df, left_index=True, right_index=True)
    df_norms = [1, -1, -1, -1, -1, 1]
    norms = df_norms + norms

    # convert date to time 0-1
    result_df['time'] = result_df['time'].dt.strftime('%H:%M:%S')
    result_df['time'] = pd.to_timedelta(result_df['time']).astype('timedelta64[s]').astype(float)/(60*60*24)

    # find and apply money offset and norm
    money_df = df.iloc[:, [n == -1 for n in df_norms]]
    min_val = pd.DataFrame.min(money_df).min()
    max_val = pd.DataFrame.max(money_df).max()
    av_money = (max_val+min_val)/2
    money_norm = (max_val-min_val)/2

    result_df.iloc[:, [n == -1 for n in norms]] -= av_money
    norms = [(money_norm if n == -1 else n) for n in norms]

    # normalize feature vec
    result_df = result_df.divide(norms, axis=1)

    # remove nan values
    nan_num = pd.DataFrame.max(pd.DataFrame.sum(np.isnan(result_df),
                                                axis=0))
    if raise_on_nan and nan_num:
        raise Exception('No nan values should be present at this time!')

    return np.array(result_df), av_money, money_norm


def correct_rate_for_fee(series, fix, rel, is_buy):
    fee = fix + (series - fix) * rel
    if is_buy:
        series += fee
    else:
        series -= fee
    return series


def get_open_close_av(open_series, close_series):
    return (open_series + close_series) / 2

