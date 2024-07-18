
from dataclasses import dataclass
from enum import Enum
from typing import Self
import numpy as np
import pandas as pd


class Normalizer:
    class Strategy(Enum):
        NONE = 'none'
        MINMAX = 'minmax'
        ZSCORE = 'zscore'
        FORMULA = 'formula'

        @staticmethod
        def from_str(s: str) -> Self:
            if s == 'none':
                return Normalizer.Strategy.NONE
            elif s == 'minmax':
                return Normalizer.Strategy.MINMAX
            elif s == 'zscore':
                return Normalizer.Strategy.ZSCORE
            elif s == 'formula':
                return Normalizer.Strategy.FORMULA
            else:
                raise ValueError(f"Normalization strategy {s} not supported!")

    @dataclass
    class Stats:
        min: float
        max: float
        mean: float
        std: float

        @staticmethod
        def from_array(arr: np.ndarray) -> 'Normalizer.Stats':
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)

            return Normalizer.Stats(
                min = arr.min(),
                max = arr.max(),
                mean = arr.mean(),
                std = arr.std())

    def __init__(self):
        self._stats = {}

    def prepare(self, df: pd.DataFrame, conf: dict) -> None:
        
        def get_group_stats(df: pd.DataFrame, cols: list[str]):
            missing_cols = [col for col in cols if col not in df.columns]
            if len(missing_cols) > 0:
                raise ValueError(f"Columns {missing_cols} not found in "
                                  "dataframe")
            
            return Normalizer.Stats.from_array(
                np.concatenate(df[cols].values))
        
        if 'df_key' not in conf:
            raise ValueError("df_key not found in configuration")

        stats = {}

        groups = conf.get('groups', [])
        for group in groups:
            group_stats = get_group_stats(df, group)        
            for col in group:
                if col in stats:
                    raise ValueError(f"Stats for column {col} already "
                                     "exists")                                    
                stats[col] = group_stats

        for col in df.columns:
            if col not in stats:
                stats[col] = Normalizer.Stats.from_array(df[col].values)

        self._stats[conf['df_key']] = stats

    def normalize_df(self, df: pd.DataFrame, conf: dict) -> pd.DataFrame:
        default_strategy = Normalizer.Strategy.from_str(
            conf.get("default_strategy", "none")
        )
        default_params = conf.get("params", {})

        normalized_df = df.copy()

        if 'df_key' not in conf:
            default_df_key = None
        else:
            default_df_key = conf['df_key']
            if default_df_key not in self._stats:
                raise ValueError(f"Stats for key {default_df_key} not "
                                 "found")

        if default_strategy != Normalizer.Strategy.NONE \
        and default_df_key is None:
            raise ValueError("If a default strategy other than 'none' is "
                             "specified, 'df_key' must be provided")

        extra = conf.get('extra', {})
        for col in df.columns:
            df_key = default_df_key
            col_key = col
            strategy = default_strategy
            params = default_params

            if col in extra:
                field_conf = extra[col]
                
                df_key = field_conf.get('stats_df', df_key)
                col_key = field_conf.get('stats_col', col_key)

                if "strategy" in field_conf:
                    strategy = Normalizer.Strategy.from_str(
                        field_conf["strategy"])
                params = field_conf.get("params", params)

            if (strategy != Normalizer.Strategy.NONE 
                and strategy != Normalizer.Strategy.FORMULA)\
             and df_key is None:
                raise ValueError(f"Strategy {strategy} requires a stats key") 

            normalized_df[col] = self._normalize(
                data=df[col].values, 
                key=df_key, 
                col=col_key,
                strategy=strategy,
                params = params)
        return normalized_df

    def _normalize(self, 
                   data: np.ndarray, 
                   key: str, col: str, 
                   strategy: Strategy,
                   params: dict) \
                -> np.ndarray:
        
        if strategy == Normalizer.Strategy.NONE:
            return data.copy()

        if strategy == Normalizer.Strategy.FORMULA:
            if 'expression' not in params:
                raise ValueError("Expression not found in formula "
                                 f"params ({params})")
            x = data
            return eval(params['expression'])

        if key not in self._stats:
            if col not in self._stats[key]:
                raise ValueError(f"Stats for key {key} and column {col} not found")
        stats = self._stats[key][col]

        if strategy == Normalizer.Strategy.MINMAX:
            return (data - stats.min) / (stats.max - stats.min)
        
        if strategy == Normalizer.Strategy.ZSCORE:
            return (data - stats.mean) / stats.std
        
        raise ValueError(f"Normalization strategy {strategy} not supported!")
