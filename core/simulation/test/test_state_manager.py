

import pandas as pd
import torch
from core.data.data_provider import ChunkType, Sample
from core.data.normalizer import Normalizer
from core.simulation.state_manager import StateManager, StateProvider
import unittest


class TestStateManager(unittest.TestCase):

    def test_state_manager(self):
        sm = StateManager()

        self.assertIsNone(sm.get_context())
        self.assertIsNone(sm.get_samples())

        samples = {'a': Sample(None, {'c': 1}),
                   'b': Sample(None, {'d': 2})}
        sm.update_samples(samples)

        s = sm.get_samples()
        self.assertIsNotNone(s)
        self.assertEqual(s['a'].context['c'], 1)
        self.assertEqual(s['b'].context['d'], 2)

        samples['a'] = Sample(None, {'c': 3})
        self.assertEqual(s['a'].context['c'], 1)
        s = sm.get_samples()
        self.assertEqual(s['a'].context['c'], 1)

        s['a'] = Sample(None, {'c': 3})
        s = sm.get_samples()
        self.assertEqual(s['a'].context['c'], 1)

        context = {'a': 1, 'b': 2}
        sm.reset_state(context)

        self.assertIsNone(sm.get_samples())

        c2 = sm.get_context()
        self.assertIsNotNone(c2)
        self.assertEqual(c2['a'], 1)
        self.assertEqual(c2['b'], 2)

        context['a'] = 3
        self.assertEqual(c2['a'], 1)
        c2 = sm.get_context()
        self.assertEqual(c2['a'], 1)

        c2['a'] = 3
        c2 = sm.get_context()
        self.assertEqual(c2['a'], 1)

        sm.update_samples(samples)
        self.assertIsNotNone(sm.get_samples())
        self.assertIsNotNone(sm.get_context())

        context = {'a': 4, 'b': 5}
        sm.update_context(context)
        self.assertIsNotNone(sm.get_samples())
        self.assertIsNotNone(sm.get_context())
        c2 = sm.get_context()
        self.assertIsNotNone(c2)
        self.assertEqual(c2['a'], 4)
        self.assertEqual(c2['b'], 5)

        context['a'] = 3
        self.assertEqual(c2['a'], 4)
        c2 = sm.get_context()
        self.assertEqual(c2['a'], 4)

        c2['a'] = 3
        c2 = sm.get_context()
        self.assertEqual(c2['a'], 4)

    def test_state_provider(self):
        sm = StateManager()
        normalizer = Normalizer()
        conf = StateProvider.Config.from_dict({
            "normalizer": {
                "default_strategy": None,
                "extra": [
                    {
                        "column": "a",
                        "strategy": {"type": "minmax"},
                        "stats_df": "norm_df",
                        "stats_col": "val",
                    },
                ]
            },
            "include": ["a", "b"]
        })

        df = pd.DataFrame({'val': range(5), 'b': [5]*5})
        normalizer.prepare(df, Normalizer.Config.from_dict({
            'df_key': 'norm_df'
        }))

        context = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

        sp = StateProvider(sm, normalizer, conf)
        self.assertEqual(sp.get_iterator(ChunkType.TRAINING), sp)
        self.assertEqual(sp.get_iterator(ChunkType.VALIDATION), sp)
        self.assertEqual(sp.get_iterator(ChunkType.TEST), sp)

        self.assertIsNone(next(sp))

        sm.update_context(context)

        s = next(sp)
        self.assertIsNotNone(s)
        self.assertDictEqual(s.context, context)
        self.assertEqual(s.tensor.shape, torch.Size([2]))
        self.assertEqual(s.tensor[0], context['a'] / 4)
        self.assertEqual(s.tensor[1], context['b'])

        context['a'] = 2
        context['b'] = 3
        sm.update_context(context)
        s = sp.update_sample(s)
        self.assertEqual(s.tensor[0], context['a'] / 4)
        self.assertEqual(s.tensor[1], context['b'])
