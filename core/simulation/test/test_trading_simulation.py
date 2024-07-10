

import unittest

import pandas as pd
from pandas import DataFrame, Series

from core.data.data_provider import DataProvider
from core.simulation.trading_simulation import TradingSimulation


class TestTradingSimulation(unittest.TestCase):

    def test_init(self):
        
        df = DataFrame(
            {
                "date": pd.date_range(start='1/1/2021', periods=100),
                "up": range(100),
                "down": range(100, 0, -1),
                "close": range(100, 0, -1),
                "chunk": [i // 10 for i in range(100)],
                "chunk_type": [0]*60 + [1]*20 + [2]*20, 
            }
        )
        conf = {
            "general": {
                "initial_balance": 1000
            }
        }

        data_provider = DataProvider(df, ['up', 'down'], 'zscore', 5)
        sim = TradingSimulation(data_provider, conf)

        try:
            sim.get_next_state()
            self.fail("Should have raised an exception")
        except Exception:
            pass

        sim.start_session('tr')
        for i in range(6*6):
            v = sim.get_next_state()
            self.assertIsNotNone(v)
            tensor, series, state, episode = v
        self.assertIsNone(sim.get_next_state())
        sim.start_session('tr')
        for i in range(6*6):
            v = sim.get_next_state()
            self.assertIsNotNone(v)
            tensor, series, state, episode = v

        sim.start_session('tr')
        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(state['money'], 1000)
        self.assertEqual(state['asset'], 0)
        self.assertIsNone(sim.sell(series))
        self.assertEqual(state['money'], 1000)
        self.assertEqual(state['asset'], 0)
        
        self.assertEqual(sim.buy(Series({'close': 100})), 10)
        self.assertEqual(state['money'], 0)
        self.assertEqual(state['asset'], 10)
        self.assertIsNone(sim.buy(series))
        
        self.assertEqual(sim.sell(Series({'close': 90})), 900)
        
        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(state['money'], 1000)        
        self.assertEqual(state['asset'], 0)

        sim.buy(series)
        for i in range(4):
            tensor, series, state, episode = sim.get_next_state()
            self.assertEqual(state['money'], 0)
            self.assertNotEqual(state['asset'], 0)
            self.assertEqual(episode, 0)

        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 1)
        self.assertEqual(state['money'], 1000)
        self.assertEqual(state['asset'], 0)
            

            
        sim.start_session('val')
        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 0)
        self.assertEqual(state['money'], 1000)
        self.assertEqual(state['asset'], 0)
        self.assertEqual(sim.buy(Series({'close': 100})), 10)

        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 0)
        self.assertEqual(state['money'], 0)
        self.assertEqual(state['asset'], 10)
        self.assertEqual(sim.sell(Series({'close': 90})), 900)
        
        for i in range(4):
            tensor, series, state, episode = sim.get_next_state()
            self.assertEqual(episode, 0)
            self.assertEqual(state['money'], 900)
            self.assertEqual(state['asset'], 0)

        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 1)
        self.assertEqual(state['money'], 1000)
        self.assertEqual(state['asset'], 0)



        sim.start_session('test')
        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 0)
        self.assertEqual(state['money'], 1000)
        self.assertEqual(state['asset'], 0)
        self.assertEqual(sim.buy(Series({'close': 100})), 10)

        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 0)
        self.assertEqual(state['money'], 0)
        self.assertEqual(state['asset'], 10)
        self.assertEqual(sim.sell(Series({'close': 90})), 900)
        
        for i in range(4):
            tensor, series, state, episode = sim.get_next_state()
            self.assertEqual(episode, 0)
            self.assertEqual(state['money'], 900)
            self.assertEqual(state['asset'], 0)

        tensor, series, state, episode = sim.get_next_state()
        self.assertEqual(episode, 1)
        self.assertEqual(state['money'], 900)
        self.assertEqual(state['asset'], 0)
