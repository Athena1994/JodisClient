from dataclasses import dataclass
from enum import Enum
from typing import Self
import numpy as np
import pandas as pd

from utils.config_utils import assert_fields_in_dict


class Normalizer:
    """
    Normalizer class for data normalization.

    Configuration:
    - df_key (mandatory for calling prepare):
        Frame key under which to store statistics calculated during
        prepare.

    - groups (optional):
        List of groups of columns. Statistics will be calculated accross
        entire group.

    - default_strategy (optional, takes value 'none' if omitted):
        The default normalization strategy to be applied if not
        specified for a specific column.

    - params (optional):
        Additional parameters for normalization strategy.

    - extra (optional):
        Additional configuration for specific columns. Can override
        the default strategy, parameters and stats source.

    example config:
    {
        "df_key": "train",
        "groups": [["col1", "col2"]],
        "default_strategy": "minmax",
        "extra": {
            "col3": {
                "strategy": "zscore",
                "stats_df": "val",
                "stats_col": "col4"
            }
        }
    }

    Supported normalization strategies:
    - 'none': No normalization is applied.
    - 'minmax': Min-max normalization.
    - 'zscore': Z-score normalization.
    - 'formula': Custom formula-based normalization.

    """

    @dataclass
    class Config:
        @dataclass
        class Strategy:
            type: "Normalizer.Strategy"
            params: dict

            @staticmethod
            def from_dict(conf: dict) -> 'Normalizer.Config.Strategy':
                if conf is None:
                    return Normalizer.Config.Strategy(
                        Normalizer.Strategy.NONE, {})
                assert_fields_in_dict(conf, ['type'])
                return Normalizer.Config.Strategy(
                    Normalizer.Strategy.from_str(conf['type']),
                    conf.get('params', {}))

        @dataclass
        class Extra:
            column: str
            strategy: 'Normalizer.Config.Strategy'
            stats_df: str
            stats_col: str

            @staticmethod
            def from_dict(conf: dict) -> 'Normalizer.Config.Extra':
                assert_fields_in_dict(conf, ['strategy', 'column'])
                return Normalizer.Config.Extra(
                    conf['column'],
                    Normalizer.Config.Strategy.from_dict(conf['strategy']),
                    conf.get('stats_df', None),
                    conf.get('stats_col', None))

        df_key: str
        default_strategy: 'Normalizer.Config.Strategy'
        groups: list[list[str]]
        extra: list['Extra']

        @staticmethod
        def from_dict(conf: dict) -> 'Normalizer.Config':

            default_strategy \
                = Normalizer.Config.Strategy.from_dict(
                    conf.get('default_strategy', None))

            extra = [Normalizer.Config.Extra.from_dict(e)
                     for e in conf.get('extra', [])]

            return Normalizer.Config(
                conf.get('df_key', None),
                default_strategy,
                conf.get('groups', []),
                extra)

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

            if not np.issubdtype(arr.dtype, np.number):
                return None

            arr = arr[~np.isnan(arr)]

            return Normalizer.Stats(
                min=arr.min(),
                max=arr.max(),
                mean=arr.mean(),
                std=arr.std())

    def __init__(self):
        self._stats = {}

    def prepare(self, df: pd.DataFrame, cfg: Config) -> None:
        """
        Prepare the normalizer by calculating statistics for the specified
        DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame for which statistics are
                               calculated.
            conf (dict): The configuration dictionary.

        Raises:
            ValueError: If 'df_key' is not found in the configuration.
            ValueError: If columns specified in 'groups' are not found in the
                        DataFrame.
            ValueError: If statistics for a column already exist.
        """
        def get_group_stats(df: pd.DataFrame, cols: list[str]):
            missing_cols = [col for col in cols if col not in df.columns]
            if len(missing_cols) > 0:
                raise ValueError(f"Columns {missing_cols} not found in df")

            return Normalizer.Stats.from_array(
                np.concatenate(df[cols].values))

        if cfg.df_key is None:
            raise ValueError("df_key must be set")

        stats = {}

        for group in cfg.groups:
            group_stats = get_group_stats(df, group)
            for col in group:
                if col in stats:
                    raise ValueError(f"Stats for column {col} already exists")
                stats[col] = group_stats

        for col in df.columns:
            if col not in stats:
                stats[col] = Normalizer.Stats.from_array(df[col].values)

        self._stats[cfg.df_key] = stats

    def normalize_data(self, data: dict, cfg: Config) -> pd.DataFrame:
        """
        Normalize the specified DataFrame using the configured normalization
        strategies.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
            conf (dict): The configuration dictionary.

        Returns:
            pd.DataFrame: The normalized DataFrame.

        Raises:
            ValueError: If 'df_key' is not found in the configuration.
            ValueError: If statistics for the specified key are not found.
            ValueError: If a strategy other than 'none' is specified without
                        providing 'df_key'.
            ValueError: If a strategy requires a stats key but it is not
                        provided.
            ValueError: If the normalization strategy is not supported.
        """

        normalized_df = data.copy()

        if cfg.df_key is not None and cfg.df_key not in self._stats:
            raise ValueError(f"Stats for key {cfg.df_key} not found")

        if cfg.default_strategy.type != Normalizer.Strategy.NONE \
                and cfg.df_key is None:
            raise ValueError("If a default strategy other than 'none' is "
                             "specified, 'df_key' must be provided")

        extra_dict = {e.column: e for e in cfg.extra}

        for col in data.keys():
            df_key = cfg.df_key
            col_key = col
            strategy = cfg.default_strategy

            if col in extra_dict:
                field_conf = extra_dict[col]
                df_key = field_conf.stats_df or df_key
                col_key = field_conf.stats_col or col_key
                strategy = field_conf.strategy

            if (strategy.type != Normalizer.Strategy.NONE
                    and strategy.type != Normalizer.Strategy.FORMULA) \
                    and df_key is None:
                raise ValueError(f"Strategy {strategy} requires a stats key")

            if isinstance(data, pd.DataFrame):
                d = data[col].values
            else:
                d = data[col]

            normalized_df[col] = self._normalize(
                data=d,
                key=df_key,
                col=col_key,
                strategy=strategy.type,
                params=strategy.params)
        return normalized_df

    def _normalize(self,
                   data: np.ndarray,
                   key: str, col: str,
                   strategy: Strategy,
                   params: dict) \
            -> np.ndarray:
        """
        Normalize the data using the specified strategy.

        Args:
            data (np.ndarray): The data to be normalized.
            key (str): The key to identify the statistics.
            col (str): The column for which statistics are used.
            strategy (Strategy): The normalization strategy.
            params (dict): Additional parameters for the strategy.

        Returns:
            np.ndarray: The normalized data.

        Raises:
            ValueError: If the strategy is 'none', 'formula', or 'minmax' and
                        the statistics are not found.
            ValueError: If the strategy is not supported.
            ValueError: If the expression is not found in the formula
                        parameters.
        """
        if strategy == Normalizer.Strategy.NONE:
            return data.copy()

        if strategy == Normalizer.Strategy.FORMULA:
            if 'expression' not in params:
                raise ValueError("Expression not found in formula params "
                                 f"({params})")

            x = data  # noqa (x is used in the eval)
            return eval(params['expression'])

        if key not in self._stats:
            if col not in self._stats[key]:
                raise ValueError(f"Stats for key {key} and column {col} not "
                                 "found")
        stats = self._stats[key][col]

        if strategy == Normalizer.Strategy.MINMAX:
            return (data - stats.min) / (stats.max - stats.min)

        if strategy == Normalizer.Strategy.ZSCORE:
            return (data - stats.mean) / stats.std

        raise ValueError(f"Normalization strategy {strategy} not supported!")
