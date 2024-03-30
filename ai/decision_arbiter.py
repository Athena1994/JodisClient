import random
from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd

from ai.utils import Actions


class DecisionArbiter:
    @abstractmethod
    def decide(self, feature_vec: pd.DataFrame) -> int:
        yield


class RandomArbiter(DecisionArbiter):
    def __init__(self):
        super().__init__()

    def decide(self,
               feature_vec: np.array,
               state_vec: np.array) -> int:
        r = random.random()
        if r < 0.01:
            return Actions.BUY
        elif r > 0.99:
            return Actions.SELL

        return Actions.WAIT


#       r = random.random()
#        if r < 0.01:
#            return Actions.BUY
#        elif r > 0.99:
#            return Actions.SELL
#
#        return Actions.WAIT

