import copy
from abc import ABC, abstractmethod

import json
import typing

from pandas import DataFrame, Series

"""
    Specific indicator logic and parameter descriptions are wrapped in 
    `IndicatorPrototype` implementations. These provide a matching 
    `IndicatorDesc` object, which in turn can be used to instantiate actual 
    indicator objects.     
"""


class Indicator:
    def __init__(self,
                 name: str,
                 params: dict,
                 fct: typing.Callable[[dict, DataFrame], Series],
                 skip_num: int,
                 norm_factor: float):
        self._name = name
        self._params = params
        self._fct = fct
        self._skip_num = skip_num
        self._norm_factor = norm_factor

    def apply_to_df(self, df: DataFrame, col: str, ix: int, cnt: int) -> None:
        def normalize(data: Series, factor: float) -> Series:
            if factor == -1:
                return data
            return data / factor
        def set_series(data: Series, df: DataFrame, col: str, i: int, l: int):
            df.loc[i + self._skip_num -1: i + l, col] = data[self._skip_num-1:]

        result = self._fct(self._params, df.iloc[ix: ix+cnt])

        if isinstance(result, Series):     
            set_series(normalize(result, self._norm_factor), 
                       df, col, ix, cnt)
        elif isinstance(result, tuple):
            set_series(normalize(result[0], self._norm_factor[0]), 
                       df, col, ix, cnt)
            for i, res in enumerate(result[1:]):
                set_series(normalize(res, self._norm_factor[i+1]), 
                           df, f"{col}_{i+2}", ix, cnt)

    def get_skip_cnt(self) -> int:
        return self._skip_num

    def get_norm_factor(self) -> float:
        return self._norm_factor
    
    def __hash__(self):
        return hash(self._name) + hash(json.dumps(self._params, sort_keys=True))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Indicator):
            return False
        return self._name == value._name and self._params == value._params


class IndicatorParameterDescription(typing.NamedTuple):
    name: str
    min_val: float
    max_val: float
    default_val: float
    data_type: str


class IndicatorDescription:
    def __init__(self, name: str,
                 params: typing.List[IndicatorParameterDescription],
                 fct: typing.Callable[[dict, DataFrame], Series],
                 norm_factor: float = 1,
                 skip_field: str = None):
        self._name = name
        self._params_description = copy.deepcopy(params)
        self._fct = fct
        self._skip_parameter = skip_field
        self._norm_factor = norm_factor

    def get_parameter_descriptions(self) \
            -> typing.List[IndicatorParameterDescription]:
        return copy.deepcopy(self._params_description)

    def create_indicator(self, parameter_values: dict) -> Indicator:
        skip_cnt = 0 if self._skip_parameter is None \
            else parameter_values[self._skip_parameter]

        return Indicator(self._name,
                         parameter_values,
                         self._fct,
                         skip_cnt,
                         self._norm_factor)

    def get_value_cnt(self) -> int:
        if isinstance(self._norm_factor, tuple):
            return len(self._norm_factor)
        else:
            return 1

class IndicatorPrototype(ABC):
    def __init__(self,
                 name: str,
                 params: typing.Collection[IndicatorDescription] = [],
                 skip_field: str = None,
                 norm_factor: float = 1):
        super().__init__()
        self._params = params
        self._name = name
        self._skip_field = skip_field
        self._norm_factor = norm_factor

    def get_descriptor(self) -> IndicatorDescription:
        return IndicatorDescription(
            self._name,
            self._params,
            self.calculate,
            self._norm_factor,
            self._skip_field
        )

    @abstractmethod
    def calculate(self, params: dict, df: DataFrame) -> Series:
        pass

    def get_parameter_descriptions(self) \
        -> typing.Collection[IndicatorParameterDescription]:
        return self._params