from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from interfaces.datatypes import TransactionReceipt, TransactionDetails


@dataclass
class DataConfiguration:
    pass


class TradingHistoryData:

    def as_dataframe(self) -> pd.DataFrame:
        pass

    def as_ndarray(self) -> np.ndarray:
        pass

    def as_tensor(self) -> torch.Tensor:
        pass


class Wallet:

    def get_asset(self, asset: str) -> float:
        pass

    def apply_transaction(self, transaction: TransactionDetails) \
            -> TransactionReceipt:
        pass


class Environment:

    class TransactionFailedError(Exception):
        pass

    def perform_transaction(self, details: TransactionDetails)\
            -> TransactionReceipt:
        pass

    def create_transaction_details(self,
                                   source_asset: str,
                                   source_amount: float,
                                   target_asset: str) -> TransactionDetails:
        pass

    def get_current_rate(self, source_asset: str, target_asset: str) -> float:
        pass

    def get_timestamp(self) -> datetime:
        pass

    def get_wallet(self) -> Wallet:
        pass

    def get_trading_history(self,
                            pair: str,
                            timeframe: str,
                            count: int,
                            config: DataConfiguration) -> TradingHistoryData:
        pass
