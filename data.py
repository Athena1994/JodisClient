#! /home/falrach/miniconda3/envs/Trading/bin

import argparse
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt

from core.data.loader.sql_loader import SQLOHCLLoader
from core.data.utils import apply_indicator, assign_chunk_ids, split_time_chunks, training_split
from program.data_manager import DataManager


def plot_df(df: pd.DataFrame):
    # plot candles with matplotlib
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['open'], label='open')
    ax.plot(df['time'], df['close'], label='close')
    ax.plot(df['time'], df['high'], label='high')
    ax.plot(df['time'], df['low'], label='low')
    ax.legend()
    plt.show()


def print_statistics(data: pd.DataFrame):

    # print number of samples
    print(f"Number of samples: {len(data)}")
    # print time range
    print(f"Time range: {data['time'].min()} - {data['time'].max()}")

    time_chunks = split_time_chunks(data)
    # print number of chunks
    print(f"Number of chunks: {len(time_chunks)}")
    
    # print time frame and size of chunks
    for i, frame in enumerate(time_chunks):
        print(f"timeframe {i}: {data.loc[frame[0], 'time']} "
              f"- {data.loc[frame[0]+frame[1]-1, 'time']} ({frame[1]})")   

    TYPES = {-1: "UNDEFINED", 0: 'TRAINING', 1: 'VALIDATION', 2: 'TEST'}
    for i, chunk_id in enumerate(data['chunk'].unique()):
        chunk_data = data[data['chunk'] == chunk_id]
        print(f"Chunk {i}: {chunk_data['time'].min()} - {chunk_data['time'].max()} "
              f"({len(chunk_data)}) - {TYPES[chunk_data['chunk_type'].iloc[0]]}")
        

def load_cached_data(input: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(input)
    except FileNotFoundError:
        print(f"File {input} not found.")
        sys.exit(1)
    return data

def cache_data(data: pd.DataFrame, output: str):
    data.to_csv(output, index=False)

def fetch_and_proccess(currency: str, interval: str, chunk_size: int, ratio: float, test_num: int, skip_cnt: int):
    data = load_data(currency, interval)
    time_chunks_ixs = split_time_chunks(data)
    data = assign_chunk_ids(data, time_chunks_ixs, chunk_size, skip_cnt)
    chunk_cnt = data['chunk'].max() + 1
    split_list = training_split(chunk_cnt, ratio, test_num)
    data['chunk_type'] = -1
    for i, chunk_type in enumerate(split_list):
        data.loc[data['chunk'] == i, 'chunk_type'] = chunk_type
    return data

#def apply_indicator(data: pd.DataFrame, indicator: dict):
    

def read_and_apply_indicators(data: pd.DataFrame, agent_file: str):

    try:
        with open(agent_file, 'r') as f:
            indicators = json.load(f)['data'][0]['indicators']
    except FileNotFoundError:
        print(f"File {agent_file} not found.")
        sys.exit(1)

    for indicator_desc in indicators:
        apply_indicator(data, indicator_desc)

def load_file(file: str):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {file} not found.")
        sys.exit(1)

def main(argv):
    parser = argparse.ArgumentParser(description='Explore current data.')

    parser.add_argument('-f', '--fetch', action='store_true', help='Fetches data.')
    parser.add_argument('-i', '--indicator', type=str, help='File with indicator values to be calculated.')
    parser.add_argument('-u', '--update_split', action="store_true", help='Update chunks.')
    parser.add_argument('-r', '--reset_split', action="store_true", help='Update will reset chunk assignment.')

    parser.add_argument('-c', '--clear', action='store_true', help='Clears cached data.')

    parser.add_argument('data_cfg', type=str, help='Path to data configuration file.')
    
#    parser.add_argument('-h', '--help', action='store_true', help='Prints help message.')

    args = parser.parse_args()

    if 'help' in args:
        parser.print_help()
        sys.exit(0)

    dm = DataManager(load_file(args.data_cfg), clear_cache=args.clear)

    if args.fetch or args.clear:
        dm.fetch_assets()

    if args.update_split:
        dm.update_training_split(args.reset_split)

    if args.indicator:
        cfg = load_file(args.indicator)
        if 'data' not in cfg:
            print("No data key found in file.")
            sys.exit(1)
        dm.update_indicators(cfg['data'])

if __name__ == '__main__':
    main(sys.argv)