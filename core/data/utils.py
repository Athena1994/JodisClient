import random
import pandas as pd

from .technical_indicators.indicators import IndicatorDescription
from .technical_indicators.collection import IndicatorCollection

class ChunkType:
    TRAINING = 0
    VALIDATION = 1
    TEST = 2

class CandleInterval:
    ONE_MINUTE = "ONE_MINUTE"
    TEN_MINUTES = "TEN_MINUTES"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"

# v: splits a dataframe into chunks based on time differences
def split_time_chunks(df: pd.DataFrame) -> list[tuple[int, int]]:
    
    # ensure time column is in datetime format
    if df['time'].dtype != 'datetime64[ns]':
        df['time'] = pd.to_datetime(df['time'])

    time_diff = df['time'].diff()
    diff_av = time_diff.min() * 2

    split_indices = time_diff[time_diff > diff_av].index
    split_indices = split_indices.append(pd.Index([len(df)]))

    # make list of tuples with start index and lenght of each chunk
    split_list = []
    start_ix = 0
    for end_ix in split_indices:
        split_list.append((start_ix, end_ix - start_ix))
        start_ix = end_ix

    return split_list


def mark_chunks(df: pd.DataFrame, split_list: list[tuple[int, int]],
                chunk_size: int, skip_cnt: int) -> pd.DataFrame:

    df['chunk'] = -1
    chunk_id = 0
    for start_ix, block_len in split_list:
        for i in range((block_len-skip_cnt) // chunk_size):            
            df.loc[(skip_cnt + start_ix + i*chunk_size):(skip_cnt + start_ix + (i+1)*chunk_size-1), 
                   'chunk'] = chunk_id
            chunk_id += 1
    return df

def training_split(chunk_cnt: int, 
                   tr_val_ratio: float, 
                   test_chunk_cnt: int) -> list[int]:
    # create list with chunk_cnt elements with values 0, 1, 2
    # 0: training, 1: validation, 2: test
    # the values should be randomly distributed with a ratio of tr_val_ratio and a total of test_chunk_cnt test chunks
    tr_cnt = chunk_cnt - test_chunk_cnt
    val_cnt = int(tr_cnt * tr_val_ratio)
    tr_cnt -= val_cnt

    split_list = [ChunkType.TRAINING] * tr_cnt
    split_list += [ChunkType.VALIDATION] * val_cnt
    split_list += [ChunkType.TEST] * test_chunk_cnt
    random.shuffle(split_list)
    return split_list

### finds regions in the data where the indicator needs to be updated
### returns a list of tuples with the start index and the length of the region
def find_indicator_update_regions(data: pd.DataFrame, name: str, skip_cnt: int):
    update_list = []
    split_list = split_time_chunks(data)

    fresh = name not in data.columns # indicator was not yet added to df

    for (start_ix, frame_len) in split_list:
        if frame_len > skip_cnt:
            if fresh: 
                ix = start_ix
            else:
                val_list = data.loc[start_ix + skip_cnt: start_ix + frame_len,
                                    name]
                val_list = val_list[val_list.isnull()]

                # check if time frame is already complete
                if len(val_list) == 0:
                    continue

                ix = val_list.index.min() - skip_cnt
            update_list.append((ix, start_ix + frame_len - ix))

    return update_list
    

def apply_indicator(data: pd.DataFrame, required_indicator: dict):
    def convert(x, type): 
        if type == 'int':
            return int(x)
        elif type == 'float':
            return float(x)
        elif type == 'str':
            return str(x)
        raise Exception(f"Unknown type {type}")
 
    if 'name' not in required_indicator or 'params' not in required_indicator:
        raise Exception('Indicator must have a name and params field')

    indicator_desc = IndicatorCollection.get(required_indicator['name'])
    descriptors = indicator_desc.get_parameter_descriptions()

    indicator_data = {}

    for d in descriptors:
        if d.name not in required_indicator['params']:
            raise Exception('Indicator must have all required parameters!'
                            f"(Missing parameter {d.name})")
        indicator_data[d.name] = convert(required_indicator['params'][d.name], d.data_type)       

    indicator = indicator_desc.create_indicator(indicator_data)
    