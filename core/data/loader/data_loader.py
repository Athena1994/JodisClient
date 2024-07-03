from abc import abstractmethod
from datetime import datetime

from pandas import DataFrame

class OHCLLoader:
    def __init__(self):
        pass

    @abstractmethod
    def get(self, pair: str, interval: str, earliest: datetime=None, last: datetime=None) -> dict:
        yield
