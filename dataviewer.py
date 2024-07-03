#! /home/falrach/miniconda3/envs/Trading/bin

import argparse
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt

from core.data.loader.sql_loader import SQLOHCLLoader
from core.data.utils import mark_chunks, split_time_chunks, training_split


SQL_USER = 'trader'
SQL_PW = 'trade_good'
SQL_SERVER = 'vserver'
SQL_DATABASE = 'trading_data'
SQL_TABLE = 'ohcl'

def plot_df(df: pd.DataFrame):
    # plot candles with matplotlib
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['open'], label='open')
    ax.plot(df['time'], df['close'], label='close')
    ax.plot(df['time'], df['high'], label='high')
    ax.plot(df['time'], df['low'], label='low')
    ax.legend()
    plt.show()

def load_data(currency: str, interval: str) -> tuple[pd.DataFrame]:
    loader = SQLOHCLLoader(SQL_USER, SQL_PW, SQL_SERVER, SQL_DATABASE, SQL_TABLE)
    data = loader.get(currency, interval)
    return data

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

def fetch_and_proccess(currency: str, interval: str, chunk_size: int, ratio: float, test_num: int):
    data = load_data(currency, interval)
    time_chunks_ixs = split_time_chunks(data)
    data = mark_chunks(data, time_chunks_ixs, chunk_size)
    chunk_cnt = data['chunk'].max() + 1
    split_list = training_split(chunk_cnt, ratio, test_num)
    data['chunk_type'] = -1
    for i, chunk_type in enumerate(split_list):
        data.loc[data['chunk'] == i, 'chunk_type'] = chunk_type
    return data

def apply_indicator(data: pd.DataFrame, indicator: dict):
    

def read_and_apply_indicators(data: pd.DataFrame, indicator_file: str):

    try:
        with open(indicator_file, 'r') as f:
            indicators = json.load(f)
    except FileNotFoundError:
        print(f"File {indicator_file} not found.")
        sys.exit(1)

    for indicator in indicators:

        data = indicator['fct'](data, indicator['params'])
    return data

def main(argv):
    parser = argparse.ArgumentParser(description='Explore current data.')

    parser.add_argument('--stats', action='store_true', help='Prints a statistics summary of stored data.')

    parser.add_argument('-p', '--pair', type=str, default="BTCEUR", help='Currency pair to explore.')
    parser.add_argument('-i', '--interval', type=str, default="ONE_MINUTE", help='Interval to explore.')
    parser.add_argument('-o', '--output', type=str, help='Output file for cached data.', default="data.csv")
    parser.add_argument('-s', '--size', type=int, help='chunk size', default=60*24)
    parser.add_argument('-q', '--ratio', type=float, help='validation split ratio', default=0.2)
    parser.add_argument('-t', '--test_num', type=float, help='number of test chunks', default=7)

    parser.add_argument('-c', '--cache', type=str, help='Input file with cached data.')

    parser.add_argument('--indicator', type=str, help='File width indicator values to be calculated.')

    parser.add_argument('--plot', type=int, help='Plots the selected chunk.')


    args = parser.parse_args()

    if args.cache:
        data = load_cached_data(args.cache)
    else:
        data = fetch_and_proccess(args.pair, args.interval, args.size, args.ratio, args.test_num)
        cache_data(data, args.output)

    if args.stats:
        print_statistics(data)

    if args.indicator:
        data = read_and_apply_indicators(data, args.indicator)
        cache_data(data, args.output)

    if args.plot:
        plot_df(data[data['chunk'] == args.plot])

if __name__ == '__main__':
    main(sys.argv)