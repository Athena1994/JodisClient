from dataclasses import dataclass
from datetime import datetime

from interfaces.enums import TransactionResult


@dataclass
class OHCLData:
    open: float
    close: float
    high: float
    low: float
    timestamp: datetime


@dataclass
class TransactionDetails:
    source_asset: str
    source_amount: float

    target_asset: str
    target_amount: float

    relative_fee: float
    fix_fee: float

    exchange_rate: float
    effective_rate: float


@dataclass
class TransactionReceipt:
    details: TransactionDetails
    result: TransactionResult
    timestamp: datetime
