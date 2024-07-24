

import unittest

import torch

from core.nn.dynamic_nn import DynamicNN


class TestDynamicNN(unittest.TestCase):

    CONF = {
        "input": {
            "input_window": 128,
            "data": [{
                "key": "ts",
                "type": "asset",
                "params": {
                    "asset": {
                        "name": "BTCEUR",
                        "source": "default",
                        "interval": "ONE_MINUTE"
                    },

                    "include": ["volume", "open", "high", "close", "low"],

                    "indicators": [
                        {
                            "name": "AwesomeOscillator",
                            "params": {
                                "short_period": 5,
                                "long_period": 34
                            }
                        }
                    ],

                    "normalizer": {
                        "df_key": "default",
                        "default_strategy": {
                            "type": "zscore"
                        },
                        "groups": [["open", "high", "close", "low"]]
                    }
                }
            }, {
                "key": "meta",
                "type": "state",
                "params": {
                    "include": [
                        "balance",
                        "position_open"
                    ],
                    "normalizer": {
                        "default_strategy": None,

                        "extra": [
                            {
                                "column": "balance",
                                "strategy": {
                                    "type": "formula",
                                    "params": {
                                        "expression": "np.log(x)/np.log(1.10)"
                                    }
                                }
                            }
                        ]
                    }
                }
            }]
        },

        "nn": {
            "units": [
                {
                    "name": "LSTM",
                    "input": "ts",
                    "type": "LSTM",
                    "params": {
                        "num_layers": 1,
                        "hidden_size": 128
                    }
                },
                {
                    "name": "fc",
                    "input": ["meta", "LSTM"],
                    "type": "Sequence",
                    "params": [
                        {"type": "ReLU"},
                        {"type": "Dropout", "p": 0.5},
                        {"type": "Linear", "size": 256}
                    ]
                },
                {
                    "name": "classifier",
                    "input": "fc",
                    "type": "Sequence",
                    "params": [
                        {"type": "ReLU"},
                        {"type": "Dropout", "p": 0.5},
                        {"type": "Linear", "size": 128},
                        {"type": "ReLU"},
                        {"type": "Dropout", "p": 0.5},
                        {"type": "Linear", "size": 3}
                    ]
                }
            ],
            "output": "classifier"
        }
    }

    def test_init(self):

        nn_conf = DynamicNN.Config.from_dict(self.CONF['nn'])
        input_conf = DynamicNN.Config.Input.from_dict(self.CONF['input'])

        dnn = DynamicNN(nn_conf, input_conf)

        self.assertEqual(len(dnn._units), 3)

        # --- lstm

        self.assertEqual(dnn._units[0]['name'], 'LSTM')
        self.assertEqual(dnn._units[0]['concat'], False)
        self.assertTrue(isinstance(dnn._units[0]['module'],
                                   torch.nn.LSTM))
        self.assertEqual(dnn._units[0]['module'].input_size, 6)
        self.assertEqual(dnn._units[0]['module'].hidden_size, 128)

        # --- fc

        self.assertEqual(dnn._units[1]['name'], 'fc')
        self.assertEqual(dnn._units[1]['concat'], True)
        self.assertTrue(isinstance(dnn._units[1]['module'],
                                   torch.nn.Sequential))
        self.assertTrue(isinstance(dnn._units[1]['module'][0],
                                   torch.nn.ReLU))
        self.assertTrue(isinstance(dnn._units[1]['module'][1],
                                   torch.nn.Dropout))
        self.assertEqual(dnn._units[1]['module'][1].p, 0.5)
        self.assertTrue(isinstance(dnn._units[1]['module'][2], torch.nn.Linear))
        self.assertEqual(dnn._units[1]['module'][2].in_features, 130)
        self.assertEqual(dnn._units[1]['module'][2].out_features, 256)

        # --- classifier

        self.assertEqual(dnn._units[2]['name'], 'classifier')
        self.assertEqual(dnn._units[2]['concat'], False)
        self.assertTrue(isinstance(dnn._units[2]['module'],
                                   torch.nn.Sequential))

        self.assertTrue(isinstance(dnn._units[2]['module'][0], torch.nn.ReLU))

        self.assertTrue(isinstance(dnn._units[2]['module'][1],
                                   torch.nn.Dropout))
        self.assertEqual(dnn._units[2]['module'][1].p, 0.5)

        self.assertTrue(isinstance(dnn._units[2]['module'][2], torch.nn.Linear))
        self.assertEqual(dnn._units[2]['module'][2].in_features, 256)
        self.assertEqual(dnn._units[2]['module'][2].out_features, 128)

        self.assertTrue(isinstance(dnn._units[2]['module'][3], torch.nn.ReLU))

        self.assertTrue(isinstance(dnn._units[2]['module'][4],
                                   torch.nn.Dropout))
        self.assertEqual(dnn._units[2]['module'][4].p, 0.5)

        self.assertTrue(isinstance(dnn._units[2]['module'][5], torch.nn.Linear))
        self.assertEqual(dnn._units[2]['module'][5].in_features, 128)
        self.assertEqual(dnn._units[2]['module'][5].out_features, 3)

    def test_forward(self):

        nn_conf = DynamicNN.Config.from_dict(self.CONF['nn'])
        input_cfg = DynamicNN.Config.Input.from_dict(self.CONF['input'])

        dnn = DynamicNN(nn_conf, input_cfg)

        # --- input

        ts = torch.rand(5, 128, 6)
        meta = torch.rand(5, 2)

        # --- forward

        out = dnn({'ts': ts, 'meta': meta})
        self.assertEqual(out.size(), (5, 3))
