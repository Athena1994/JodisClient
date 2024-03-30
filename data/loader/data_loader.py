from abc import abstractmethod


class OHCLLoader:
    def __init__(self):
        pass

    @abstractmethod
    def get(self, pair, interval, earliest=None, last=None):
        yield
