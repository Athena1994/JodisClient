

import unittest

import pandas as pd
from pandas import DataFrame, Series

from core.simulation.sample_provider import SampleProvider
from program.asset_provider import AssetProvider


class TestExperienceProvider(unittest.TestCase):

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

        data_provider = AssetProvider(df, ['up', 'down'], 'zscore', 5)
        sim = SampleProvider(data_provider, conf)

        try:
            sim.get_next_sample()
            self.fail("Should have raised an exception")
        except Exception:
            pass

        sim.start_session('tr')
        for i in range(6*6):
            state = sim.get_next_sample()
            self.assertIsNotNone(state)

        self.assertIsNone(sim.get_next_sample())
        sim.start_session('tr')
        for i in range(6*6):
            state = sim.get_next_sample()
            self.assertIsNotNone(state)

        sim.start_session('tr')
        state = sim.get_next_sample()
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)
        self.assertIsNone(sim.sell(state.context))
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)

        self.assertEqual(sim.buy(Series({'close': 100})), 10)
        state = sim.get_current_state()
        self.assertEqual(state.simulation['money'], 0)
        self.assertEqual(state.simulation['asset'], 10)
        self.assertIsNone(sim.buy(state.context))

        self.assertEqual(sim.sell(Series({'close': 90})), 900)

        state = sim.get_next_sample()
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)

        sim.buy(state.context)
        for i in range(4):
            state = sim.get_next_sample()
            self.assertEqual(state.simulation['money'], 0)
            self.assertNotEqual(state.simulation['asset'], 0)
            self.assertEqual(state.episode, 0)

        state = sim.get_next_sample()
        self.assertEqual(state.episode, 1)
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)

        sim.start_session('val')
        state = sim.get_next_sample()
        self.assertEqual(state.episode, 0)
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)
        self.assertEqual(sim.buy(Series({'close': 100})), 10)

        state = sim.get_next_sample()
        self.assertEqual(state.episode, 0)
        self.assertEqual(state.simulation['money'], 0)
        self.assertEqual(state.simulation['asset'], 10)
        self.assertEqual(sim.sell(Series({'close': 90})), 900)

        for i in range(4):
            state = sim.get_next_sample()
            self.assertEqual(state.episode, 0)
            self.assertEqual(state.simulation['money'], 900)
            self.assertEqual(state.simulation['asset'], 0)

        state = sim.get_next_sample()
        self.assertEqual(state.episode, 1)
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)

        sim.start_session('test')
        state = sim.get_next_sample()
        self.assertEqual(state.episode, 0)
        self.assertEqual(state.simulation['money'], 1000)
        self.assertEqual(state.simulation['asset'], 0)
        self.assertEqual(sim.buy(Series({'close': 100})), 10)

        state = sim.get_next_sample()
        self.assertEqual(state.episode, 0)
        self.assertEqual(state.simulation['money'], 0)
        self.assertEqual(state.simulation['asset'], 10)
        self.assertEqual(sim.sell(Series({'close': 90})), 900)

        for i in range(4):
            state = sim.get_next_sample()
            self.assertEqual(state.episode, 0)
            self.assertEqual(state.simulation['money'], 900)
            self.assertEqual(state.simulation['asset'], 0)

        state = sim.get_next_sample()
        self.assertEqual(state.episode, 1)
        self.assertEqual(state.simulation['money'], 900)
        self.assertEqual(state.simulation['asset'], 0)
