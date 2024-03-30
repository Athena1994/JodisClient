import typing
from typing import Tuple
from abc import abstractmethod

from numpy import ndarray
from pandas import DataFrame

from technical_indicators import indicators, collection
from technical_indicators.indicators import Indicator
from data.utils import prepare_feature_vec
from trading.interface import TradingInterface
from ai.decision_arbiter import DecisionArbiter, RandomArbiter


class Agent:

    @abstractmethod
    def select_action(self, ti: TradingInterface, state) -> int:
        yield

    @abstractmethod
    def add_indicators_to_df(self, df: DataFrame) -> DataFrame:
        yield

    @abstractmethod
    def get_lookback(self) -> int:
        yield


class BaseIndicatorAgent(Agent):

    def __init__(self,
                 indicator_list: typing.List[Indicator],
                 arbiter: DecisionArbiter):
        self._indicators = indicator_list
        self._lookback = 60
        self._extra = 60  # nan barrier
        self._arbiter = arbiter

        self._df = None
        self._last_date = None

    def get_lookback(self) -> int:
        return self._lookback

    def add_indicators_to_df(self, df: DataFrame) -> Tuple[ndarray,
                                                           Tuple[float, float]]:

        feature_vec, norm = indicators.apply(df, self._indicators)
        history_feature_vec, av, money_norm \
            = prepare_feature_vec(df=df,
                                  ind_df=feature_vec,
                                  norms=norm,
                                  raise_on_nan=False)
        return history_feature_vec, (av, money_norm)

    def select_action(self, ti: TradingInterface, state):

        feature_vec, state_vec = ti.get_state()

        action_id = self._arbiter.decide(feature_vec, state_vec)
        return action_id


class SimpleIndicatorAgent(BaseIndicatorAgent):

    def __init__(self, arbiter: DecisionArbiter):
        self._descriptors = collection.get_all_indicators()
        self._params = []
        for desc in self._descriptors:
            desc_params = desc.get_parameter_descriptions()
            keys = desc_params.keys()
            values = [v[2] for v in desc_params.values()]
            self._params += [dict(map(lambda i, j: (i, j), keys, values))]

        self._indicators = indicators.instantiate(self._descriptors,
                                                  self._params)

        super().__init__(self._indicators, arbiter)

