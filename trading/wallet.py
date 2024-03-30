import copy
import datetime
import typing

import numpy as np


class ReceiptItems:
    DATE = 'date'
    SOURCE_ASSET = 'source_asset'
    SOURCE_AMOUNT = 'source_amount'
    TARGET_ASSET = 'target_asset'
    TARGET_AMOUNT = 'target_amount'
    RATE = 'exchange_rate'
    FEE = 'fee'
    EFFECTIVE_RATE = 'effective_rate'


class Wallet:
    def __init__(self, init_assets: dict = {}):
        self.assets = init_assets
        self.history = []
        self._verbosity = 0

    def get_verbosity(self):
        return self._verbosity

    def set_verbosity(self, val: int):
        self._verbosity = val

    def get_inv_history(self, cnt: int = -1):
        if cnt == -1:
            return self.history[::-1]

        total = len(self.history)
        if total == 0:
            return []
        if total < cnt:
            return self.history[::-1]
        return self.history[-1:-(cnt+1):-1]

    def get_open_position(self, asset: str) -> typing.Tuple[datetime.datetime, float]:
        if self.get_asset(asset) == 0:
            return None, 0

        # get all receipts with given asset
        receipt = next(iter([r for r in self.history[::-1] if r[ReceiptItems.TARGET_ASSET] == asset
                                                           or r[ReceiptItems.SOURCE_ASSET] == asset]))
        # check whether last order bought or sold asset
        if receipt is None or receipt[ReceiptItems.SOURCE_ASSET] == asset:
            return None, 0
        else:
            return receipt[ReceiptItems.DATE], receipt[ReceiptItems.RATE]

    def _modify(self, cur, amount):
        if cur not in self.assets:
            self.assets[cur] = 0

        if self.assets[cur] + amount < 0:
            raise Exception(f'wallet does not hold enough of asset "{cur}" '
                            f'(present: {self.assets[cur]} | change: {amount}')

        self.assets[cur] += amount

    def exec_transaction(self,
                         source_asset: str, source_amount: float,
                         target_asset: str, target_amount: float,
                         exchange_rate: float, fee: float,
                         timestamp: datetime) -> tuple:
        if target_amount == 0 and self._verbosity > 0:
            print('invalid transaction')
            return None

        self._modify(source_asset, -source_amount)
        self._modify(target_asset, target_amount)

        effective_rate = source_amount/target_amount if target_amount != 0 else np.nan

        receipt = {
               ReceiptItems.DATE: timestamp,
               ReceiptItems.SOURCE_ASSET: source_asset,
               ReceiptItems.SOURCE_AMOUNT: source_amount,
               ReceiptItems.TARGET_ASSET: target_asset,
               ReceiptItems.TARGET_AMOUNT: target_amount,
               ReceiptItems.RATE: exchange_rate,
               ReceiptItems.FEE: fee,
               ReceiptItems.EFFECTIVE_RATE: effective_rate
           }

        if self._verbosity > 0:
            print(receipt)

        self.history += [receipt]

        if len(self.history) > 2:
            self.history = self.history[-2:]

        return receipt

    def get_asset(self, cur: str) -> float:
        if cur not in self.assets:
            self.assets[cur] = 0

        return self.assets[cur]

    @staticmethod
    def load_from_state(state: dict):
        wallet = Wallet(state['assets'])
        wallet.history = state['history']
        return wallet

    def get_state(self) -> dict:
        return {'assets': copy.deepcopy(self.assets),
                'history': copy.deepcopy(self.history)}

    def set_state(self, state: dict):
        self.assets = copy.deepcopy(state['assets'])
        self.history = copy.deepcopy(state['history'])

