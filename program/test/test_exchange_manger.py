from unittest import TestCase

from pandas import Series

from core.data.data_provider import Sample
from core.simulation.state_manager import StateManager
from program.exchange_manager import ExchangeDirection, StateSourcedExchanger


class TestExchanger(TestCase):
    def test_state_source_exchanger(self):
        sm = StateManager()

        exchanger = StateSourcedExchanger(sm, {
            'pairs': [
                {
                    'asset': 'BTC',
                    'currency': 'USD',
                    'fee': {
                        'relative': 0.1,
                        'fixed': 1
                    },
                    'candle_src': 'test'
                }
            ]
        })

        with self.assertRaises(Exception):
            exchanger.get_exchange_details('USD', 'ETH')

        sm.update_samples({
            'test': Sample(None, Series({'close': 100, 'open': 100}))})

        with self.assertRaises(Exception):
            exchanger.get_exchange_details('USD', 'ETH')

        with self.assertRaises(Exception):
            exchanger.get_exchange_details('EUR', 'BTC')

        details = exchanger.get_exchange_details('USD', 'BTC')
        self.assertEqual(details.buy_rate, 100)
        self.assertEqual(details.sell_rate, 100)
        self.assertEqual(details.fee.relative, 0.1)
        self.assertEqual(details.fee.fixed, 1)

        receipt = exchanger.prepare_exchange('USD',
                                             'BTC',
                                             1000,
                                             ExchangeDirection.BUY)

        self.assertEqual(receipt.currency_balance, -1000)
        self.assertEqual(receipt.fee, 100.9)
        self.assertEqual(receipt.asset_balance, 8.991)
        self.assertEqual(receipt.currency, 'USD')
        self.assertEqual(receipt.asset, 'BTC')

        receipt = exchanger.prepare_exchange('USD',
                                             'BTC',
                                             10,
                                             ExchangeDirection.SELL)
        self.assertEqual(receipt.currency_balance, 899.1)
        self.assertEqual(receipt.fee, 100.9)
        self.assertEqual(receipt.asset_balance, -10)
        self.assertEqual(receipt.currency, 'USD')
        self.assertEqual(receipt.asset, 'BTC')
